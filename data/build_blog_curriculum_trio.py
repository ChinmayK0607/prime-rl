"""Build a TRIOWISE contrastive 3-way train curriculum (Run 15B).

Why this exists: every prior run's bottleneck is the unconstrained global 3-way class
MARGINAL drifting (entropy keeps all classes alive but does NOT balance them; pairs restore
a per-boundary gradient but each pair only ever exposes TWO of the three labels, so the
GLOBAL marginal is still free to drift). A triowise task closes that gap: each prompt shows
THREE texts (A/B/C), each authored by one of CLAUDE/CHATGPT/GEMINI with the SAME provider
allowed to repeat, and the reward is the FRACTION of the three slots assigned correctly (per-slot
accuracy, multiplicatively gated on a substantive <reason_why>). Per-slot (not all-or-nothing) is
deliberate: all-or-nothing exact-match is sparsest exactly on the under-predicted class, so it
yields NO corrective gradient where drift is worst (drift self-reinforces). Per-slot accuracy keeps
the rollout group MIXED (rollouts vary in #correct) so a restoring gradient flows even before any
rollout nails all three. It stays non-hackable: the only within-group variance axis IS partial
correctness (monotonic in true accuracy; for uniform-marginal trios no constant/escape strategy
beats E=1/3 per slot). Because a trio spans the whole label space at realistic class frequencies,
raising slot accuracy REQUIRES tracking the true class frequencies -- attacking the drift bottleneck.

Composition (the user's hard requirement): trio author compositions MUST be MIXED and must NOT
be biased toward "3 distinct labels" (which would teach a permutation shortcut). We sample each
of the 3 slots i.i.d. uniformly over {CLAUDE, CHATGPT, GEMINI}, which yields ~22% 3-distinct,
~67% two-of-one-plus-one (e.g. CHATGPT/CLAUDE/CHATGPT), ~11% 3-same. This makes both the
"always 3 distinct" and the "constant class" strategies losing strategies (a constant label is
wrong on any 2+1 or 3-distinct trio), and keeps the gold marginal uniform (matching val).

Per step (12 prompts, matching batch_size 96 / group_size 8): 6 single 3-way (2/2/2, the
val-matching task that always provides a difficulty spread so the step is never starved) + 6
trios. Single-text stream and strip augmentation are reused verbatim from the pairwise builder.
Val is copied through unchanged (single-text 3-way, eval-only).
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

from datasets import Dataset, load_from_disk

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]
        / "deps" / "research-environments" / "environments" / "blog_author_id"),
)
from blog_author_id import SYSTEM_PROMPT_3WAY, make_trio_system_prompt  # noqa: E402

SRC = Path(__file__).resolve().parent / "blog_author_id_3way"
OUT = Path(__file__).resolve().parent / os.environ.get("BLOG_OUT", "blog_author_id_trio")
PREDS = Path("/home/ubuntu/blogger/blog-eval/logs/multimodel3way_predictions.jsonl")

PROVIDERS = ["CLAUDE", "CHATGPT", "GEMINI"]
THREEWAY_PER_CLASS = 2                       # 6 single 3-way per step (2/2/2)
TRIOS_PER_STEP = int(os.environ.get("TRIOS_PER_STEP", "6"))
N_STEPS = 56
TRAINABLE_NC = {1, 2, 3}
STRIP_FRAC = 0.40
TRIO_MAXLEN = 9000                           # chars/text (3 fit the 16k window w/ room)
SEED = 0
LENGTH_RANK = {"short": 0, "long": 1}


def strip_formatting(t: str) -> str:
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
    need = N_STEPS * THREEWAY_PER_CLASS
    pool = list(pool)
    rng.shuffle(pool)
    seq = [pool[i % len(pool)] for i in range(need)]
    seq = [d for _, d in sorted(enumerate(seq), key=lambda x: (x[1]["lrank"], x[0]))]
    return [seq[s * THREEWAY_PER_CLASS:(s + 1) * THREEWAY_PER_CLASS] for s in range(N_STEPS)]


def assign_strips_3way(order_by_step: list[list[dict]], rng: random.Random) -> set[int]:
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


# ----------------------------- triowise stream -------------------------------------------
def build_trio_assignments(train: Dataset, rng: random.Random) -> list[dict]:
    """Build N_STEPS*TRIOS_PER_STEP trios. Each trio = three SHORT blogs whose providers are
    drawn i.i.d. uniformly over {C,P,G} (mixed composition, no 3-distinct bias, uniform gold
    marginal). Texts truncated to TRIO_MAXLEN; per-text strip augmentation at STRIP_FRAC."""
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

    trios: list[dict] = []
    for _ in range(N_STEPS * TRIOS_PER_STEP):
        golds = [rng.choice(PROVIDERS) for _ in range(3)]  # i.i.d. uniform per slot
        srcs = [take(g) for g in golds]
        strips = [rng.random() < STRIP_FRAC for _ in range(3)]
        comp = "distinct" if len(set(golds)) == 3 else ("same" if len(set(golds)) == 1 else "twoone")
        trios.append({"golds": golds, "srcs": srcs, "strips": strips, "comp": comp})
    return trios


def make_3way_prompt(text: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT_3WAY},
            {"role": "user", "content": text}]


def make_trio_prompt(a_text: str, b_text: str, c_text: str) -> list[dict]:
    user = f"### TEXT A\n{a_text}\n\n### TEXT B\n{b_text}\n\n### TEXT C\n{c_text}"
    return [{"role": "system", "content": make_trio_system_prompt()},
            {"role": "user", "content": user}]


def prep_text(train: Dataset, src: int, strip: bool) -> str:
    t = train[src]["question"]
    if strip:
        t = strip_formatting(t)
    return t[:TRIO_MAXLEN]


def main() -> None:
    rng = random.Random(SEED)
    rates = load_passrates()
    train = load_from_disk(str(SRC / "train"))
    val = load_from_disk(str(SRC / "val"))

    pools = build_pools(train, rates)
    print("[trio] trainable-middle (nc1..3) pool sizes:",
          {p: len(pools[p]) for p in PROVIDERS})

    dealt = {p: deal_class(pools[p], rng) for p in PROVIDERS}

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

    trios = build_trio_assignments(train, rng)
    assert len(trios) == N_STEPS * TRIOS_PER_STEP

    rows = {"prompt": [], "answer": [], "info": []}
    trio_iter = iter(trios)
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
        for _ in range(TRIOS_PER_STEP):
            tr = next(trio_iter)
            texts = [prep_text(train, tr["srcs"][k], tr["strips"][k]) for k in range(3)]
            ga, gb, gc = tr["golds"]
            rows["prompt"].append(make_trio_prompt(*texts))
            rows["answer"].append(f"A={ga};B={gb};C={gc}")
            rows["info"].append({
                "task_type": "trio",
                "comp": tr["comp"],
                "stripped": any(tr["strips"]),
                "a_gold": ga, "b_gold": gb, "c_gold": gc,
            })
    train_out = Dataset.from_dict(rows)

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
    step_len = THREEWAY_PER_CLASS * 3 + TRIOS_PER_STEP
    print(f"\n[trio] {N_STEPS} steps; {len(train_out)} train rows "
          f"({N_STEPS*THREEWAY_PER_CLASS*3} single 3-way + {N_STEPS*TRIOS_PER_STEP} trio)")
    for s in range(N_STEPS):
        chunk = train_out[s * step_len:(s + 1) * step_len]
        infos = chunk["info"]; answers = chunk["answer"]
        gold3 = Counter(a for a, i in zip(answers, infos) if i["task_type"] == "3way")
        if not (gold3["CLAUDE"] == gold3["CHATGPT"] == gold3["GEMINI"] == THREEWAY_PER_CLASS):
            failures.append((s, "3way-imbalance", dict(gold3)))
        ntrio = sum(1 for i in infos if i["task_type"] == "trio")
        if ntrio != TRIOS_PER_STEP:
            failures.append((s, "trio-count", ntrio))

    all_info = train_out["info"]; all_ans = train_out["answer"]
    trio_infos = [i for i in all_info if i["task_type"] == "trio"]
    comp = Counter(i["comp"] for i in trio_infos)
    n_trio = len(trio_infos)
    # gold marginal across all trio slots
    slot_marg = Counter()
    for i in trio_infos:
        slot_marg[i["a_gold"]] += 1
        slot_marg[i["b_gold"]] += 1
        slot_marg[i["c_gold"]] += 1
    nstrip = sum(1 for i in all_info if i["stripped"])
    print("\n[trio] global:")
    print("  3way gold:", Counter(a for a, i in zip(all_ans, all_info) if i["task_type"] == "3way"))
    print("  trio composition:", dict(comp),
          f"=> distinct {comp['distinct']/n_trio:.0%} / twoone {comp['twoone']/n_trio:.0%} "
          f"/ same {comp['same']/n_trio:.0%}")
    print("  trio slot gold marginal (should be ~uniform):", dict(slot_marg))
    print(f"  stripped rows: {nstrip} ({nstrip/len(train_out):.0%})")

    # the user's hard requirement: 2+1 must be present and 3-distinct must NOT dominate.
    if comp["twoone"] <= comp["distinct"]:
        failures.append(("global", "trio-distinct-bias", dict(comp)))
    if comp["twoone"] == 0:
        failures.append(("global", "no-twoone-trios", dict(comp)))

    if failures:
        raise SystemExit(f"\n[trio] FAILED: {len(failures)} bad: {failures[:10]}")
    print("\n[trio] OK: every step = 2/2/2 single 3-way + 6 trios; mixed composition "
          "(2+1 majority, no 3-distinct bias), uniform gold marginal.")

    OUT.mkdir(parents=True, exist_ok=True)
    train_out.save_to_disk(str(OUT / "train"))
    val_out.save_to_disk(str(OUT / "val"))
    print(f"\n[trio] wrote {len(train_out)} train / {len(val_out)} val to {OUT}")


if __name__ == "__main__":
    main()
