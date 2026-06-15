"""Build the Run-7 curriculum: balanced single 3-way + SYMMETRIC pairwise contrastives.

Why this exists (what Run 6 proved and what still broke)
--------------------------------------------------------
Run 6 interleaved a single CLAUDE-vs-CHATGPT contrastive pair into the 3/3/3 single-text
curriculum. It WORKED for its target — the Run-5 CHATGPT collapse was cured (CHATGPT train
recall stayed healthy 0.86-0.94) and val rose 0.245 -> 0.324. BUT the collapse merely
ROTATED: GEMINI collapsed instead (GEMINI recall 0.39 -> 0.00, predicted-GEMINI -> 0.01 by
step 27) while the global 3-way prediction prior oscillated CLAUDE-heavy (steps 11-18) then
CHATGPT-heavy (steps 19-27). Diagnosis: the contrastive pair only protects the ONE boundary
it covers (C-vs-CHATGPT); GEMINI had no contrastive protection and the global 3-way prior is
still unconstrained, so the abandoned class simply moved.

Fix (this builder): cover EVERY boundary with its own contrastive pair —
  CLAUDE-vs-CHATGPT, CLAUDE-vs-GEMINI, CHATGPT-vs-GEMINI.
Each pair shows TWO texts (A and B) with DIFFERENT authors drawn from exactly that boundary's
two providers (random A/B order), restricts the label space to those two providers (no escape
to the third), and is scored with the UNCHANGED plain-binary pair reward (1.0 iff BOTH slots
correct -> non-hackable; all-wrong groups stay zero-advantage-filtered, no shaped escape).
Because a constant/position-prior policy is always exactly 50% wrong on a different-author
pair, every pair's rollout group stays MIXED -> survives the zero-advantage filter ->
a restoring gradient flows on THAT boundary every step. Covering all three boundaries
protects all three classes symmetrically, so no class can be abandoned (the rubber-duck
critique of the Run-6 result recommended exactly this symmetric coverage, raising the
contrastive fraction to ~50%).

Layout: every consecutive 12 on-disk rows == one training step ==
  6 single 3-way rows (strict 2 CLAUDE / 2 CHATGPT / 2 GEMINI, trainable-middle nc1..3,
    difficulty/length/strip matched across classes — kept because val/eval is single-text
    3-way, so the single rows keep the training distribution matched to evaluation)
  + 6 pairwise-contrastive rows (2x C-vs-CHATGPT + 2x C-vs-GEMINI + 2x CHATGPT-vs-GEMINI).
Per step the label exposure is balanced 6/6/6 (singles 2/2/2 + pairs 4/4/4: each provider
appears in exactly two of the three boundaries, 2 pairs each => 4).

Output: data/blog_author_id_pairwise/{train,val}. EVERY row carries a per-row ``prompt``
column ([system,user] messages) so a single env + single binary reward serves every task
type (the env sets system_prompt=None when a prompt column is present). info carries
task_type ("3way"|"pair"), pair_kind (e.g. "CLAUDE-GEMINI"), nc, stripped. Val is the
unchanged single-text 3-way set (204 rows).
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset, load_from_disk

# Single source of truth for the prompt text (must match the env / eval baseline).
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]
        / "deps" / "research-environments" / "environments" / "blog_author_id"),
)
from blog_author_id import SYSTEM_PROMPT_3WAY, make_pair_system_prompt  # noqa: E402

SRC = Path(__file__).resolve().parent / "blog_author_id_3way"
# OUT and the per-boundary pair mix are env-configurable so one builder produces both
# the balanced curriculum (blog_author_id_diverse: 2/2/2 pairs) and a curriculum that
# concentrates contrastive pressure on the hard CHATGPT-vs-GEMINI boundary
# (blog_author_id_pvg: 1 CvP / 2 CvG / 3 PvG). The total pairs/step is held at 6 in both
# (6 singles + 6 pairs = 12 prompts/step) so batch_size 144 / group_size 12 is unchanged.
OUT = Path(__file__).resolve().parent / os.environ.get("BLOG_OUT", "blog_author_id_diverse")
PREDS = Path("/home/ubuntu/blogger/blog-eval/logs/multimodel3way_predictions.jsonl")

PROVIDERS = ["CLAUDE", "CHATGPT", "GEMINI"]
# Per-step: 6 single 3-way (2/2/2) + 6 pairs = 12 prompts (~50% contrastive).
THREEWAY_PER_CLASS = 2
BOUNDARIES = [("CLAUDE", "CHATGPT"), ("CLAUDE", "GEMINI"), ("CHATGPT", "GEMINI")]
# Per-step pair count per boundary (env-overridable). Default 2/2/2 = balanced. The
# CHATGPT-vs-GEMINI ("PvG") boundary is the empirically hard one, so a PvG-weighted
# build (1/2/3) is selectable via PAIRS_CVP/PAIRS_CVG/PAIRS_PVG without forking the file.
PAIRS_PER_BOUNDARY = {
    ("CLAUDE", "CHATGPT"): int(os.environ.get("PAIRS_CVP", "2")),
    ("CLAUDE", "GEMINI"): int(os.environ.get("PAIRS_CVG", "2")),
    ("CHATGPT", "GEMINI"): int(os.environ.get("PAIRS_PVG", "2")),
}
N_STEPS = 56
TRAINABLE_NC = {1, 2, 3}
STRIP_FRAC = 0.40
PAIR_MAXLEN = 14000           # chars per text in a pair (two fit the 16k window w/ room)
SEED = 0
LENGTH_RANK = {"short": 0, "long": 1}


def strip_formatting(t: str) -> str:
    """Remove markdown/formatting cues while preserving wording (matches the
    content-signal probe that scored ~100% val acc on stripped text)."""
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", t)
    t = re.sub(r"(?m)^\s*[-*+]\s+", "", t)
    t = re.sub(r"(?m)^\s*\d+\.\s+", "", t)
    t = t.replace("**", "").replace("__", "")
    t = re.sub(r"(?<!\*)\*(?!\*)", "", t)
    t = t.replace("\u2014", ", ").replace("\u2013", "-")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def load_passrates() -> dict[tuple[str, str, str, str], int]:
    rates: dict[tuple[str, str, str, str], int] = {}
    for line in PREDS.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("split") != "train" or r.get("markdown") != "keep":
            continue
        rates[(r["category"], r["topic"], r["source_model"], r["length"])] = r["n_correct"]
    return rates


def nc_of(info: dict, rates: dict) -> int:
    return rates.get((info["category"], info["topic"], info["source_model"], info["length"]), 2)


# ----------------------- single 3-way stream (Run-5 recipe, PER_CLASS=2) -----------------
def build_pools(train: Dataset, rates: dict) -> dict[str, list[dict]]:
    pools: dict[str, list[dict]] = {p: [] for p in PROVIDERS}
    for i, row in enumerate(train):
        info = row["info"]
        nc = nc_of(info, rates)
        if nc not in TRAINABLE_NC:
            continue
        pools[row["answer"]].append(
            {"src": i, "gold": row["answer"], "nc": nc,
             "lrank": LENGTH_RANK.get(info["length"], 0)}
        )
    return pools


def deal_class(pool: list[dict], rng: random.Random) -> list[list[dict]]:
    # Uniform sample across the whole trainable-middle pool so the EXPANDED
    # diverse categories (all defaulted to nc=2) are sampled in proportion to
    # availability. The previous round-robin-over-nc dealing drew ~1/3 of singles
    # from each nc bucket, under-sampling nc=2 (where all the new humanities/
    # social-science data lives) and starving the single-text stream of exactly
    # the register diversity it needs to generalize to the held-out categories.
    need = N_STEPS * THREEWAY_PER_CLASS
    pool = list(pool)
    rng.shuffle(pool)
    seq = [pool[i % len(pool)] for i in range(need)]
    seq = [d for _, d in sorted(enumerate(seq), key=lambda x: (x[1]["lrank"], x[0]))]
    return [seq[s * THREEWAY_PER_CLASS:(s + 1) * THREEWAY_PER_CLASS] for s in range(N_STEPS)]


def assign_strips_3way(order_by_step: list[list[dict]], rng: random.Random) -> set[int]:
    """~STRIP_FRAC of three-way positions stripped, balanced per class within each step."""
    n_steps = len(order_by_step)
    target = STRIP_FRAC * THREEWAY_PER_CLASS
    lo = int(target)
    n_hi = int(round((target - lo) * n_steps))
    strip_pos: set[int] = set()
    for p in PROVIDERS:
        counts = [lo + 1] * n_hi + [lo] * (n_steps - n_hi)
        rng.shuffle(counts)
        for step, k in zip(order_by_step, counts):
            picks = [d["_pos"] for d in step if d["gold"] == p]
            rng.shuffle(picks)
            strip_pos.update(picks[:k])
    return strip_pos


# ----------------------------- pairwise-contrastive stream -------------------------------
def build_pair_assignments(train: Dataset, rng: random.Random) -> list[dict]:
    """Build, per boundary, N_STEPS*PAIRS_PER_BOUNDARY contrastive pairs. Each pair = one
    SHORT blog from each of the boundary's two providers (truncated to PAIR_MAXLEN), random
    A/B order with the A-slot gold balanced 50/50 so position carries no exploitable prior.

    Returns a flat list ordered so that step s gets, for each boundary in BOUNDARIES,
    PAIRS_PER_BOUNDARY consecutive pairs (the materializer slices PAIRS_PER_STEP per step)."""
    short: dict[str, list[int]] = {p: [] for p in PROVIDERS}
    for i, row in enumerate(train):
        if row["answer"] in short and row["info"].get("length") == "short":
            short[row["answer"]].append(i)
    for k in short:
        rng.shuffle(short[k])

    cursors = {p: 0 for p in PROVIDERS}

    def take(prov: str) -> int:
        pool = short[prov]
        idx = pool[cursors[prov] % len(pool)]
        cursors[prov] += 1
        return idx

    # Per boundary, build N_STEPS*PAIRS_PER_BOUNDARY[b] pairs with a 50/50 A-slot balance.
    per_boundary: dict[tuple[str, str], list[dict]] = {}
    for (p1, p2) in BOUNDARIES:
        need = N_STEPS * PAIRS_PER_BOUNDARY[(p1, p2)]
        a_is_p1_flags = [True] * (need // 2) + [False] * (need - need // 2)
        rng.shuffle(a_is_p1_flags)
        pairs: list[dict] = []
        for a_is_p1 in a_is_p1_flags:
            s1 = take(p1)
            s2 = take(p2)
            a_gold = p1 if a_is_p1 else p2
            b_gold = p2 if a_is_p1 else p1
            a_src = s1 if a_is_p1 else s2
            b_src = s2 if a_is_p1 else s1
            pairs.append({
                "p1": p1, "p2": p2, "pair_kind": f"{p1}-{p2}",
                "a_src": a_src, "b_src": b_src,
                "a_gold": a_gold, "b_gold": b_gold,
                "a_strip": rng.random() < STRIP_FRAC,
                "b_strip": rng.random() < STRIP_FRAC,
            })
        per_boundary[(p1, p2)] = pairs

    # Flatten so step s = [boundary0 x cnt0, boundary1 x cnt1, boundary2 x cnt2].
    flat: list[dict] = []
    for s in range(N_STEPS):
        for b in BOUNDARIES:
            cnt = PAIRS_PER_BOUNDARY[b]
            seg = per_boundary[b][s * cnt:(s + 1) * cnt]
            flat.extend(seg)
    return flat


def make_3way_prompt(text: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT_3WAY},
            {"role": "user", "content": text}]


def make_pair_prompt(p1: str, p2: str, a_text: str, b_text: str) -> list[dict]:
    user = f"### TEXT A\n{a_text}\n\n### TEXT B\n{b_text}"
    return [{"role": "system", "content": make_pair_system_prompt(p1, p2)},
            {"role": "user", "content": user}]


def prep_pair_text(train: Dataset, src: int, strip: bool) -> str:
    t = train[src]["question"]
    if strip:
        t = strip_formatting(t)
    return t[:PAIR_MAXLEN]


def main() -> None:
    rng = random.Random(SEED)
    rates = load_passrates()
    train = load_from_disk(str(SRC / "train"))
    val = load_from_disk(str(SRC / "val"))

    pools = build_pools(train, rates)
    print("[pairwise] trainable-middle (nc1..3) pool sizes:",
          {p: len(pools[p]) for p in PROVIDERS})

    dealt = {p: deal_class(pools[p], rng) for p in PROVIDERS}

    # single 3-way positions per step (interleaved C/P/G) with flat positions for strips.
    order_by_step: list[list[dict]] = []
    pos = 0
    for s in range(N_STEPS):
        step: list[dict] = []
        for j in range(THREEWAY_PER_CLASS):
            for p in PROVIDERS:
                d = dict(dealt[p][s][j])
                d["_pos"] = pos
                pos += 1
                step.append(d)
        order_by_step.append(step)
    strip_pos = assign_strips_3way(order_by_step, rng)

    pairs_per_step = sum(PAIRS_PER_BOUNDARY.values())  # 6 in both builds
    pairs = build_pair_assignments(train, rng)
    assert len(pairs) == N_STEPS * pairs_per_step, (len(pairs), N_STEPS * pairs_per_step)

    # Materialize: per step emit the 6 single 3-way rows then the 6 pair rows.
    rows = {"prompt": [], "answer": [], "info": []}
    pair_iter = iter(pairs)
    for s in range(N_STEPS):
        for d in order_by_step[s]:
            base = train[d["src"]]
            text = base["question"]
            stripped = d["_pos"] in strip_pos
            if stripped:
                text = strip_formatting(text)
            info = dict(base["info"])
            info.update(task_type="3way", nc=d["nc"], stripped=stripped)
            rows["prompt"].append(make_3way_prompt(text))
            rows["answer"].append(d["gold"])
            rows["info"].append(info)
        for _ in range(pairs_per_step):
            pr = next(pair_iter)
            a_text = prep_pair_text(train, pr["a_src"], pr["a_strip"])
            b_text = prep_pair_text(train, pr["b_src"], pr["b_strip"])
            rows["prompt"].append(make_pair_prompt(pr["p1"], pr["p2"], a_text, b_text))
            rows["answer"].append(f"A={pr['a_gold']};B={pr['b_gold']}")
            rows["info"].append({
                "task_type": "pair",
                "pair_kind": pr["pair_kind"],
                "stripped": pr["a_strip"] or pr["b_strip"],
                "a_gold": pr["a_gold"], "b_gold": pr["b_gold"],
            })
    train_out = Dataset.from_dict(rows)

    # Val: unchanged single-text 3-way, prompt column for env-path parity.
    vrows = {"prompt": [], "answer": [], "info": []}
    for row in val:
        info = dict(row["info"])
        info.update(task_type="3way", stripped=False)
        vrows["prompt"].append(make_3way_prompt(row["question"]))
        vrows["answer"].append(row["answer"])
        vrows["info"].append(info)
    val_out = Dataset.from_dict(vrows)

    # ---- verification / report ----
    failures = []
    print(f"\n[pairwise] {N_STEPS} steps; {len(train_out)} train rows "
          f"({N_STEPS*THREEWAY_PER_CLASS*3} single 3-way + {N_STEPS*pairs_per_step} pair)")
    print("  step | 3way gold C/P/G | pair kinds (count) | strip3 C/P/G")
    for s in range(N_STEPS):
        chunk = train_out[s * 12:(s + 1) * 12]
        infos = chunk["info"]; answers = chunk["answer"]
        gold3 = Counter(a for a, i in zip(answers, infos) if i["task_type"] == "3way")
        if not (gold3["CLAUDE"] == gold3["CHATGPT"] == gold3["GEMINI"] == THREEWAY_PER_CLASS):
            failures.append((s, "3way-imbalance", dict(gold3)))
        kinds = Counter(i.get("pair_kind") for i in infos if i["task_type"] == "pair")
        npair = sum(kinds.values())
        if npair != pairs_per_step:
            failures.append((s, "pair-count", npair))
        # every boundary present exactly its configured count
        for (p1, p2) in BOUNDARIES:
            if kinds.get(f"{p1}-{p2}", 0) != PAIRS_PER_BOUNDARY[(p1, p2)]:
                failures.append((s, "boundary-count", (f"{p1}-{p2}", kinds.get(f"{p1}-{p2}"))))
        # every pair must be exactly its two providers, different authors
        for i in infos:
            if i["task_type"] != "pair":
                continue
            p1, p2 = i["pair_kind"].split("-")
            if {i["a_gold"], i["b_gold"]} != {p1, p2}:
                failures.append((s, "pair-mismatch", i))
        nc_ok = all(i.get("nc") in TRAINABLE_NC for i in infos if i["task_type"] == "3way")
        if not nc_ok:
            failures.append((s, "non-trainable-nc", None))
        strip3 = {p: sum(1 for a, i in zip(answers, infos)
                         if i["task_type"] == "3way" and a == p and i["stripped"])
                  for p in PROVIDERS}
        if s < 4 or s >= N_STEPS - 2:
            kinds_str = " ".join(f"{k.split('-')[0][0]}{k.split('-')[1][0]}:{v}"
                                 for k, v in sorted(kinds.items()))
            print(f"  {s:3d}  | {gold3['CLAUDE']}/{gold3['CHATGPT']}/{gold3['GEMINI']} | "
                  f"{kinds_str} | "
                  f"{strip3['CLAUDE']}/{strip3['CHATGPT']}/{strip3['GEMINI']}")

    all_info = train_out["info"]; all_ans = train_out["answer"]
    nstrip = sum(1 for i in all_info if i["stripped"])
    print("\n[pairwise] global:")
    print("  3way gold:", Counter(a for a, i in zip(all_ans, all_info) if i["task_type"] == "3way"))
    pair_infos = [i for i in all_info if i["task_type"] == "pair"]
    print("  pair kinds:", Counter(i["pair_kind"] for i in pair_infos))
    # label exposure across pairs (each provider should be balanced)
    pair_label_exposure = Counter()
    a_slot = Counter()
    for i in pair_infos:
        pair_label_exposure[i["a_gold"]] += 1
        pair_label_exposure[i["b_gold"]] += 1
        a_slot[i["a_gold"]] += 1
    print("  pair label exposure (A+B):", dict(pair_label_exposure))
    print("  pair A-slot gold (should be ~balanced per boundary):", dict(a_slot))
    print(f"  stripped rows: {nstrip} ({nstrip/len(train_out):.0%})")

    if failures:
        raise SystemExit(f"\n[pairwise] FAILED: {len(failures)} bad steps: {failures[:10]}")
    print(f"\n[pairwise] OK: every step = 2/2/2 single 3-way + pairs "
          f"{dict((f'{a[0]}{b[0]}', n) for (a, b), n in PAIRS_PER_BOUNDARY.items())}.")

    OUT.mkdir(parents=True, exist_ok=True)
    train_out.save_to_disk(str(OUT / "train"))
    val_out.save_to_disk(str(OUT / "val"))
    print(f"\n[pairwise] wrote {len(train_out)} train / {len(val_out)} val to {OUT}")


if __name__ == "__main__":
    main()
