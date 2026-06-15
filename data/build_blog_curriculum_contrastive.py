"""Build the Run-6 curriculum: strict-balanced 3-way + paired-CONTRASTIVE C-vs-P.

Why this exists (what Run 5 proved and what still broke)
-------------------------------------------------------
Run 5's strict 4/4/4 balanced 3-way curriculum fixed per-step GOLD imbalance and
balanced the active-gradient mass, and train reward climbed 0.33 -> 0.625 without a
hard collapse. BUT per-step rollout analysis showed the gain was HACKED: the policy
aced CLAUDE+GEMINI and ABANDONED the confusable CHATGPT (train CHATGPT recall fell
52% at init -> 0% by step 19; the prediction prior went C46/P3/G46). Mechanism: once
CHATGPT recall -> ~0, every CHATGPT prompt yields an all-wrong rollout group, which the
GRPO zero-advantage filter drops -> NO corrective gradient -> an ABSORBING collapse.
Strict 4/4/4 balances the gold but cannot stop this: getting CLAUDE(4/4)+GEMINI(4/4)
and CHATGPT(0/4) still scores 8/12 = 0.67. prime-rl's loss has no reference-policy KL
and no entropy bonus, so nothing anchors the policy to the (better-balanced) base.

Fix (this builder): interleave a paired-CONTRASTIVE auxiliary task that makes
class-abandonment structurally impossible and keeps the hard boundary's gradient alive.
Each pair prompt shows TWO texts (A and B) — exactly one CLAUDE and one CHATGPT in random
order — and asks the model to assign each. Because the two authors differ, a constant
label is >=50% wrong, so the group stays MIXED (survives the zero-advantage filter ->
gradient keeps flowing) and earning reward REQUIRES distinguishing CLAUDE from CHATGPT.
Reward stays plain binary (1.0 iff BOTH assignments correct) so it is non-hackable.

Layout: every consecutive 12 on-disk rows == one training step ==
  9 three-way rows (strict 3 CLAUDE / 3 CHATGPT / 3 GEMINI, trainable-middle nc1..3,
    difficulty/length/strip matched across classes — same recipe as Run 5)
  + 3 paired-contrastive rows (each = 1 CLAUDE + 1 CHATGPT short blog, random A/B order).
So per step the 3-way gold is balanced 3/3/3 AND three extra CLAUDE-vs-CHATGPT
discriminations inject gradient on exactly the boundary that collapsed.

Output: data/blog_author_id_contrastive/{train,val}. EVERY row carries a per-row
``prompt`` column ([system,user] messages) so a single env + single binary reward serves
both task types (the env sets system_prompt=None when a prompt column is present).
info carries task_type ("3way"|"pair"), nc, stripped. Val is unchanged 3-way (204 rows).
"""

from __future__ import annotations

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
from blog_author_id import SYSTEM_PROMPT_3WAY, SYSTEM_PROMPT_PAIR  # noqa: E402

import json  # noqa: E402

SRC = Path(__file__).resolve().parent / "blog_author_id_3way"
OUT = Path(__file__).resolve().parent / "blog_author_id_contrastive"
PREDS = Path("/home/ubuntu/blogger/blog-eval/logs/multimodel3way_predictions.jsonl")

PROVIDERS = ["CLAUDE", "CHATGPT", "GEMINI"]
THREEWAY_PER_CLASS = 3        # 3/3/3 -> 9 three-way prompts per step
PAIRS_PER_STEP = 3            # + 3 contrastive C-vs-P prompts -> 12 prompts/step
N_STEPS = 28
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


# ----------------------- 3-way stream (Run-5 recipe, PER_CLASS=3) -----------------------
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
    need = N_STEPS * THREEWAY_PER_CLASS
    by_nc: dict[int, list[dict]] = defaultdict(list)
    for d in pool:
        by_nc[d["nc"]].append(d)
    for nc in by_nc:
        rng.shuffle(by_nc[nc])
    cursors = {nc: 0 for nc in by_nc}
    ncs = sorted(by_nc)
    seq: list[dict] = []
    k = 0
    while len(seq) < need:
        nc = ncs[k % len(ncs)]
        bucket = by_nc[nc]
        seq.append(bucket[cursors[nc] % len(bucket)])
        cursors[nc] += 1
        k += 1
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


# ----------------------------- paired-contrastive stream -----------------------------
def build_pair_assignments(train: Dataset, rng: random.Random) -> list[dict]:
    """Build N_STEPS*PAIRS_PER_STEP contrastive pairs, each = one CLAUDE + one CHATGPT
    SHORT blog (truncated to PAIR_MAXLEN), with random A/B order and per-blog strip."""
    short = {"CLAUDE": [], "CHATGPT": []}
    for i, row in enumerate(train):
        if row["answer"] in short and row["info"].get("length") == "short":
            short[row["answer"]].append(i)
    for k in short:
        rng.shuffle(short[k])
    need = N_STEPS * PAIRS_PER_STEP
    # Balance the A-slot gold exactly 50/50 so position carries no exploitable prior.
    a_claude_flags = [True] * (need // 2) + [False] * (need - need // 2)
    rng.shuffle(a_claude_flags)
    pairs: list[dict] = []
    ci = pi = 0
    for idx in range(need):
        c_src = short["CLAUDE"][ci % len(short["CLAUDE"])]; ci += 1
        p_src = short["CHATGPT"][pi % len(short["CHATGPT"])]; pi += 1
        a_is_claude = a_claude_flags[idx]
        pairs.append({
            "task": "pair",
            "a_src": c_src if a_is_claude else p_src,
            "b_src": p_src if a_is_claude else c_src,
            "a_gold": "CLAUDE" if a_is_claude else "CHATGPT",
            "b_gold": "CHATGPT" if a_is_claude else "CLAUDE",
            "a_strip": rng.random() < STRIP_FRAC,
            "b_strip": rng.random() < STRIP_FRAC,
        })
    return pairs


def make_3way_prompt(text: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT_3WAY},
            {"role": "user", "content": text}]


def make_pair_prompt(a_text: str, b_text: str) -> list[dict]:
    user = f"### TEXT A\n{a_text}\n\n### TEXT B\n{b_text}"
    return [{"role": "system", "content": SYSTEM_PROMPT_PAIR},
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
    print("[contrastive] trainable-middle (nc1..3) pool sizes:",
          {p: len(pools[p]) for p in PROVIDERS})

    dealt = {p: deal_class(pools[p], rng) for p in PROVIDERS}

    # 3-way positions per step (interleaved C/P/G) with flat positions for strip selection.
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

    pairs = build_pair_assignments(train, rng)

    # Materialize: per step emit the 9 three-way rows then the 3 pair rows.
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
        for _ in range(PAIRS_PER_STEP):
            pr = next(pair_iter)
            a_text = prep_pair_text(train, pr["a_src"], pr["a_strip"])
            b_text = prep_pair_text(train, pr["b_src"], pr["b_strip"])
            rows["prompt"].append(make_pair_prompt(a_text, b_text))
            rows["answer"].append(f"A={pr['a_gold']};B={pr['b_gold']}")
            rows["info"].append({
                "task_type": "pair",
                "stripped": pr["a_strip"] or pr["b_strip"],
                "a_gold": pr["a_gold"], "b_gold": pr["b_gold"],
            })
    train_out = Dataset.from_dict(rows)

    # Val: unchanged 3-way, prompt column for env-path parity.
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
    print(f"\n[contrastive] {N_STEPS} steps; {len(train_out)} train rows "
          f"({N_STEPS*THREEWAY_PER_CLASS*3} three-way + {N_STEPS*PAIRS_PER_STEP} pair)")
    print("  step | 3way gold C/P/G | pair A=C/A=P | strip3 C/P/G")
    for s in range(N_STEPS):
        chunk = train_out[s * 12:(s + 1) * 12]
        infos = chunk["info"]; answers = chunk["answer"]
        gold3 = Counter(a for a, i in zip(answers, infos) if i["task_type"] == "3way")
        if not (gold3["CLAUDE"] == gold3["CHATGPT"] == gold3["GEMINI"] == THREEWAY_PER_CLASS):
            failures.append((s, "3way-imbalance", dict(gold3)))
        npair = sum(1 for i in infos if i["task_type"] == "pair")
        if npair != PAIRS_PER_STEP:
            failures.append((s, "pair-count", npair))
        a_is_c = sum(1 for i in infos if i["task_type"] == "pair" and i["a_gold"] == "CLAUDE")
        # every pair must be exactly one CLAUDE + one CHATGPT
        for i in infos:
            if i["task_type"] == "pair" and {i["a_gold"], i["b_gold"]} != {"CLAUDE", "CHATGPT"}:
                failures.append((s, "pair-not-CvP", i))
        nc_ok = all(i.get("nc") in TRAINABLE_NC for i in infos if i["task_type"] == "3way")
        if not nc_ok:
            failures.append((s, "non-trainable-nc", None))
        strip3 = {p: sum(1 for a, i in zip(answers, infos)
                         if i["task_type"] == "3way" and a == p and i["stripped"])
                  for p in PROVIDERS}
        if s < 6 or s >= N_STEPS - 2:
            print(f"  {s:3d}  | {gold3['CLAUDE']}/{gold3['CHATGPT']}/{gold3['GEMINI']} | "
                  f"{a_is_c}/{PAIRS_PER_STEP - a_is_c} | "
                  f"{strip3['CLAUDE']}/{strip3['CHATGPT']}/{strip3['GEMINI']}")

    all_info = train_out["info"]; all_ans = train_out["answer"]
    nstrip = sum(1 for i in all_info if i["stripped"])
    print("\n[contrastive] global:")
    print("  3way gold:", Counter(a for a, i in zip(all_ans, all_info) if i["task_type"] == "3way"))
    npairs = sum(1 for i in all_info if i["task_type"] == "pair")
    pair_a = Counter(i["a_gold"] for i in all_info if i["task_type"] == "pair")
    print(f"  pairs: {npairs}; A-slot gold: {dict(pair_a)} (should be ~balanced)")
    print(f"  stripped rows: {nstrip} ({nstrip/len(train_out):.0%})")

    if failures:
        raise SystemExit(f"\n[contrastive] FAILED: {len(failures)} bad steps: {failures[:10]}")
    print("\n[contrastive] OK: every step = 3/3/3 three-way + 3 CLAUDE-vs-CHATGPT pairs.")

    OUT.mkdir(parents=True, exist_ok=True)
    train_out.save_to_disk(str(OUT / "train"))
    val_out.save_to_disk(str(OUT / "val"))
    print(f"\n[contrastive] wrote {len(train_out)} train / {len(val_out)} val to {OUT}")


if __name__ == "__main__":
    main()
