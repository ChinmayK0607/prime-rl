"""Build the STRICT per-step-balanced 3-way curriculum (stability-first redesign).

Why this exists
---------------
Two prior runs (aug, hardpair) showed a CLASS-PRIOR OSCILLATION rather than learning:
the model's marginal prediction prior chased the recent-majority gold class. Root cause
(grounded in per-step rollout analysis + a rubber-duck critique):

  1. Per-step GOLD imbalance. GRPO normalizes advantage WITHIN each group (8 rollouts of
     one prompt), but the per-step GRADIENT is the average over all 12 groups in the
     step. If a step has more CLAUDE-gold groups, that step over-reinforces CLAUDE; with
     optimizer momentum the prior chases whatever class recently dominated. The old
     builder's loose guard (max>=6 or min==0) permitted 5/3/4-type steps, and the 3
     hard-pair rows (gold CLAUDE/CHATGPT) skewed each step further.

  2. Active-GRADIENT imbalance (subtler, flagged by the critique). Only groups with
     0 < n_correct < 8 produce gradient (zero-advantage filter drops all-wrong and
     all-correct groups). The base-model difficulty is wildly asymmetric:
        CLAUDE  nc: {0:59, 1:53, 2:20, 3:5}   (hardest; 43% all-wrong at init)
        CHATGPT nc: {0:18, 1:44, 2:48, 3:21, 4:6}
        GEMINI  nc: {0:11, 1:44, 2:50, 3:28, 4:4}
     With naive 4/4/4 over ALL difficulties, ~43% of CLAUDE groups sample all-wrong and
     get filtered, so CLAUDE contributes far less ACTIVE gradient than CHATGPT/GEMINI ->
     CLAUDE under-learned / unstable even with balanced gold labels.

Fix (this builder):
  - PURE 3-way (no hard-pair auxiliary; it broke balance and flipped the bias).
  - Every consecutive 12 on-disk rows == one training step == EXACTLY 4 CLAUDE + 4
    CHATGPT + 4 GEMINI gold (strict; verified, hard-fails otherwise).
  - Draw each class's 4 prompts from its TRAINABLE-MIDDLE pool (n_correct in 1..3) so
    every group is expected to be mixed -> balanced ACTIVE-gradient mass across classes.
    (nc0 = base always-wrong -> would sample all-wrong and be filtered = dead weight that
    unbalances; nc4 = trivial all-correct -> also filtered. Both excluded.)
  - Within each step, match DIFFICULTY (stratified nc mix), LENGTH (short/long), and
    AUGMENTATION (strip flag) across the three classes so no class is systematically
    easier/longer/stripped in any step (de-correlates the per-step gradient).
  - 40% formatting-strip augmentation, distributed evenly across classes & steps.
  - Gentle global short->long drift applied symmetrically to all three classes.

Output: data/blog_author_id_balanced/{train,val} with the proven 3-way schema
(question / answer / info; NO per-row prompt column, so the env prepends
SYSTEM_PROMPT_3WAY exactly like the offline baseline). info carries
nc / stripped / task_type="3way" for analysis.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset, load_from_disk

SRC = Path(__file__).resolve().parent / "blog_author_id_3way"
OUT = Path(__file__).resolve().parent / "blog_author_id_balanced"
PREDS = Path("/home/ubuntu/blogger/blog-eval/logs/multimodel3way_predictions.jsonl")

PROVIDERS = ["CLAUDE", "CHATGPT", "GEMINI"]
PER_CLASS = 4                 # per step per class -> 12-prompt step, strict 4/4/4
N_STEPS = 28                  # 28*4=112 picks/class; CLAUDE middle pool=78 (=>~1.4x
                              # reuse), CHATGPT 113 / GEMINI 122 (reuse-free).
TRAINABLE_NC = {1, 2, 3}      # exclude nc0 (always-wrong->filtered) and nc4 (trivial)
STRIP_FRAC = 0.40
SEED = 0
LENGTH_RANK = {"short": 0, "long": 1}


def strip_formatting(t: str) -> str:
    """Remove markdown/formatting cues while preserving wording (matches the
    content-signal probe that scored ~100% val acc on stripped text)."""
    t = re.sub(r"```.*?```", " ", t, flags=re.S)          # code fences
    t = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", t)            # headers
    t = re.sub(r"(?m)^\s*[-*+]\s+", "", t)                 # bullets
    t = re.sub(r"(?m)^\s*\d+\.\s+", "", t)                 # numbered lists
    t = t.replace("**", "").replace("__", "")              # bold
    t = re.sub(r"(?<!\*)\*(?!\*)", "", t)                  # italics
    t = t.replace("\u2014", ", ").replace("\u2013", "-")   # em/en dash -> neutral
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)                         # collapse blank lines
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


def build_pools(train: Dataset, rates: dict) -> dict[str, list[dict]]:
    """Per-class trainable-middle descriptors, sorted by (nc, length) so a stratified
    deal can pull a balanced difficulty/length mix into every step."""
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
    """Deal a class pool into N_STEPS buckets of PER_CLASS each (with controlled reuse
    when the pool is smaller than N_STEPS*PER_CLASS). Stratify by nc so each step gets a
    matched difficulty mix; apply a gentle global short->long drift across steps.

    Returns a list of N_STEPS lists, each of length PER_CLASS.
    """
    need = N_STEPS * PER_CLASS
    # Round-robin across nc buckets to interleave difficulty, shuffling within a bucket.
    by_nc: dict[int, list[dict]] = defaultdict(list)
    for d in pool:
        by_nc[d["nc"]].append(d)
    for nc in by_nc:
        rng.shuffle(by_nc[nc])
    # Build a difficulty-interleaved sequence, cycling buckets; reuse wraps each bucket.
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
    # Gentle global short->long drift: stable-sort the dealt sequence by length only,
    # preserving the difficulty interleave as the secondary order.
    seq = [d for _, d in sorted(enumerate(seq), key=lambda x: (x[1]["lrank"], x[0]))]
    return [seq[s * PER_CLASS:(s + 1) * PER_CLASS] for s in range(N_STEPS)]


def assign_strips(order_by_step: list[list[dict]], rng: random.Random) -> set[int]:
    """Pick which (flattened) positions are formatting-stripped to ~STRIP_FRAC overall,
    balanced PER CLASS within each step so strip status never correlates with a class.

    PER_CLASS is small (4), so a fixed per-step count can only be 25% or 50%. To average
    STRIP_FRAC we give each class a per-step strip count that mixes floor/ceil across
    steps (e.g. 0.40 -> some steps strip 2, others 1), independently per class."""
    n_steps = len(order_by_step)
    target = STRIP_FRAC * PER_CLASS                 # e.g. 1.6
    lo = int(target)                                # 1
    n_hi = int(round((target - lo) * n_steps))      # #steps that strip lo+1
    strip_pos: set[int] = set()
    for p in PROVIDERS:
        counts = [lo + 1] * n_hi + [lo] * (n_steps - n_hi)
        rng.shuffle(counts)
        for step, k in zip(order_by_step, counts):
            picks = [d["_pos"] for d in step if d["gold"] == p]
            rng.shuffle(picks)
            strip_pos.update(picks[:k])
    return strip_pos


def materialize(order: list[dict], train: Dataset, strip_pos: set[int]) -> Dataset:
    rows = {"question": [], "answer": [], "info": []}
    for d in order:
        base = train[d["src"]]
        text = base["question"]
        stripped = d["_pos"] in strip_pos
        if stripped:
            text = strip_formatting(text)
        info = dict(base["info"])
        info["task_type"] = "3way"
        info["nc"] = d["nc"]
        info["stripped"] = stripped
        rows["question"].append(text)
        rows["answer"].append(d["gold"])
        rows["info"].append(info)
    return Dataset.from_dict(rows)


def build_val(val: Dataset) -> Dataset:
    rows = {"question": [], "answer": [], "info": []}
    for row in val:
        info = dict(row["info"])
        info["task_type"] = "3way"
        info["stripped"] = False
        rows["question"].append(row["question"])
        rows["answer"].append(row["answer"])
        rows["info"].append(info)
    return Dataset.from_dict(rows)


def main() -> None:
    rng = random.Random(SEED)
    rates = load_passrates()
    train = load_from_disk(str(SRC / "train"))
    val = load_from_disk(str(SRC / "val"))

    pools = build_pools(train, rates)
    print("[balanced] trainable-middle (nc1..3) pool sizes:",
          {p: len(pools[p]) for p in PROVIDERS})

    dealt = {p: deal_class(pools[p], rng) for p in PROVIDERS}

    # Interleave classes within each step; assign flat positions as we go.
    order_by_step: list[list[dict]] = []
    order: list[dict] = []
    pos = 0
    for s in range(N_STEPS):
        step: list[dict] = []
        for j in range(PER_CLASS):
            for p in PROVIDERS:                 # CLAUDE, CHATGPT, GEMINI round-robin
                d = dict(dealt[p][s][j])
                d["_pos"] = pos
                pos += 1
                step.append(d)
                order.append(d)
        order_by_step.append(step)

    strip_pos = assign_strips(order_by_step, rng)
    train_out = materialize(order, train, strip_pos)
    val_out = build_val(val)

    # ---- verification / report (hard-fail on any per-step imbalance) ----
    print(f"\n[balanced] {N_STEPS} steps; {len(order)} train rows")
    print("  step | gold C/P/G | nc(C,P,G means) | short C/P/G | strip C/P/G")
    failures = []
    for s, step in enumerate(order_by_step):
        gold = Counter(d["gold"] for d in step)
        if not (gold["CLAUDE"] == gold["CHATGPT"] == gold["GEMINI"] == PER_CLASS):
            failures.append((s, "gold-imbalance", dict(gold)))
        ncmean = {p: round(sum(d["nc"] for d in step if d["gold"] == p) / PER_CLASS, 2)
                  for p in PROVIDERS}
        short = {p: sum(1 for d in step if d["gold"] == p and d["lrank"] == 0)
                 for p in PROVIDERS}
        strip = {p: sum(1 for d in step if d["gold"] == p and d["_pos"] in strip_pos)
                 for p in PROVIDERS}
        if any(d["nc"] not in TRAINABLE_NC for d in step):
            failures.append((s, "non-trainable-nc", None))
        if s < 8 or s >= N_STEPS - 2:
            print(f"  {s:3d}  | {gold['CLAUDE']}/{gold['CHATGPT']}/{gold['GEMINI']} | "
                  f"{ncmean['CLAUDE']},{ncmean['CHATGPT']},{ncmean['GEMINI']} | "
                  f"{short['CLAUDE']}/{short['CHATGPT']}/{short['GEMINI']} | "
                  f"{strip['CLAUDE']}/{strip['CHATGPT']}/{strip['GEMINI']}")

    print("\n[balanced] global:")
    print("  gold:", Counter(d["gold"] for d in order))
    print("  nc:", dict(sorted(Counter(d["nc"] for d in order).items())))
    nstrip = sum(1 for r in train_out if r["info"]["stripped"])
    print(f"  stripped: {nstrip} ({nstrip/len(train_out):.0%})")
    # reuse report
    for p in PROVIDERS:
        used = [d["src"] for d in order if d["gold"] == p]
        print(f"  {p}: {len(used)} picks / {len(set(used))} unique "
              f"({len(used)/max(1,len(set(used))):.2f}x reuse)")
    short_pos = [i for i, d in enumerate(order) if d["lrank"] == 0]
    if short_pos:
        print(f"  short rows mean position: {sum(short_pos)/len(short_pos):.1f} "
              f"of {len(order)} (<{len(order)/2:.0f} => short->long drift)")

    if failures:
        raise SystemExit(f"\n[balanced] FAILED: {len(failures)} bad steps: {failures[:10]}")
    print("\n[balanced] OK: every step is strict 4/4/4 gold, all nc in 1..3.")

    OUT.mkdir(parents=True, exist_ok=True)
    train_out.save_to_disk(str(OUT / "train"))
    val_out.save_to_disk(str(OUT / "val"))
    print(f"\n[balanced] wrote {len(train_out)} train / {len(val_out)} val to {OUT}")


if __name__ == "__main__":
    main()
