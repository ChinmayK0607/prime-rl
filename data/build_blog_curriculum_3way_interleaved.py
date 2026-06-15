"""Build a BATCH-AWARE interleaved 3-way train curriculum.

Why this exists: the strict easy->hard sort (build_blog_curriculum_3way.py)
COLLAPSED GRPO training. It clustered same-difficulty prompts, so once the run
reached the n_correct=0 tail whole 12-prompt steps were all-wrong groups (zero
advantage) and `Trainable` cratered to ~1%. Val fell below baseline.

A GRPO group (group_size=8 rollouts of ONE prompt) only yields a non-zero
advantage when the rollouts DISAGREE (some right, some wrong). So every training
step needs a spread of difficulties (especially the trainable middle, n_correct
1..3) AND provider balance (12 prompts split ~4/4/4 to avoid class-collapse).

Construction (each consecutive block of `BATCH_PROMPTS` on-disk rows == one
training step under PRIME_RL_PRESERVE_DATA_ORDER=1):
  - Split rows by provider (CLAUDE/CHATGPT/GEMINI). Counts are 137/137/137.
  - For EACH provider independently, stratify its rows across the W windows so
    every window receives that provider's proportional n_correct mix: within
    each n_correct bucket, order short-before-long (gentle short->long drift),
    then deal bucket row j to window floor(j * W / bucket_size).
  - Window w = provider_C[w] + provider_G[w] + provider_P[w], interleaved
    round-robin. => each ~12-row window is 4/4/4 provider-balanced and carries
    the global n_correct mix (~{0:2.6,1:4.1,2:3.5,3:1.6,4:0.3}), i.e. >=7
    middle-difficulty (trainable) prompts per step.

The result is a gentle short->long, mixed-difficulty, provider-balanced order.
Val is copied through unchanged (eval-only).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset, load_from_disk

SRC = Path(__file__).resolve().parent / "blog_author_id_3way"
OUT = Path(__file__).resolve().parent / "blog_author_id_3way_interleaved"
PREDS = Path("/home/ubuntu/blogger/blog-eval/logs/multimodel3way_predictions.jsonl")

LENGTH_RANK = {"short": 0, "long": 1}
PROVIDER_ORDER = ["CLAUDE", "CHATGPT", "GEMINI"]
BATCH_PROMPTS = 12  # orchestrator batch_size 96 / group_size 8


def load_passrates() -> dict[tuple[str, str, str, str], int]:
    """(category, topic, source_model, length) -> n_correct in 0..4 (keep, train)."""
    rates: dict[tuple[str, str, str, str], int] = {}
    for line in PREDS.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("split") != "train" or r.get("markdown") != "keep":
            continue
        rates[(r["category"], r["topic"], r["source_model"], r["length"])] = r["n_correct"]
    return rates


def stratify_provider(rows: list[tuple[int, int, int]], n_windows: int) -> list[list[int]]:
    """rows = list of (row_index, n_correct, length_rank) for ONE provider.

    Return a per-window list of row indices: each window gets this provider's
    proportional n_correct mix, with short-before-long drift across windows.
    """
    by_nc: dict[int, list[tuple[int, int]]] = defaultdict(list)  # nc -> [(lrank, idx)]
    for idx, nc, lrank in rows:
        by_nc[nc].append((lrank, idx))

    windows: list[list[int]] = [[] for _ in range(n_windows)]
    # Deal each difficulty bucket evenly across windows, short blogs first so the
    # earliest windows lean short (gentle short->long curriculum drift).
    for nc in sorted(by_nc):  # ascending nc; order within bucket is what matters
        bucket = sorted(by_nc[nc])  # (lrank asc => short first, then idx)
        size = len(bucket)
        for j, (_lrank, idx) in enumerate(bucket):
            w = (j * n_windows) // size
            windows[w].append(idx)
    return windows


def build_order(train: Dataset, rates: dict[tuple[str, str, str, str], int]) -> list[int]:
    n_windows = (len(train) + BATCH_PROMPTS - 1) // BATCH_PROMPTS

    per_provider: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for i, row in enumerate(train):
        info = row["info"]
        key = (info["category"], info["topic"], info["source_model"], info["length"])
        nc = rates.get(key, 2)  # neutral if unmeasured
        lrank = LENGTH_RANK.get(info["length"], 0)
        per_provider[row["answer"]].append((i, nc, lrank))

    prov_windows = {p: stratify_provider(per_provider[p], n_windows) for p in PROVIDER_ORDER}

    order: list[int] = []
    for w in range(n_windows):
        # Round-robin C/G/P within the window so the on-disk block is balanced.
        cursors = {p: 0 for p in PROVIDER_ORDER}
        lists = {p: prov_windows[p][w] for p in PROVIDER_ORDER}
        remaining = sum(len(v) for v in lists.values())
        while remaining:
            for p in PROVIDER_ORDER:
                if cursors[p] < len(lists[p]):
                    order.append(lists[p][cursors[p]])
                    cursors[p] += 1
                    remaining -= 1
    return order


def main() -> None:
    rates = load_passrates()
    print(f"[interleave] loaded {len(rates)} measured train pass-rates")

    train = load_from_disk(str(SRC / "train"))
    val = load_from_disk(str(SRC / "val"))

    order = build_order(train, rates)
    assert len(order) == len(train), (len(order), len(train))
    assert len(set(order)) == len(train), "duplicate indices in interleave order"

    train_sorted = train.select(order)

    def row_nc(row) -> int:
        info = row["info"]
        return rates.get((info["category"], info["topic"], info["source_model"], info["length"]), 2)

    # Per-window composition report + degeneracy guard.
    print(f"\n[interleave] per-window composition (first 15 of {(len(train)+11)//12} windows):")
    print("  win | providers C/G/P | n_correct {0:..4:} | #middle(nc1-3) | #short")
    bad = []
    n_windows = (len(train) + BATCH_PROMPTS - 1) // BATCH_PROMPTS
    for w in range(n_windows):
        block = train_sorted.select(range(w * 12, min((w + 1) * 12, len(train_sorted))))
        prov = Counter(block["answer"])
        nc = Counter(row_nc(r) for r in block)
        middle = sum(nc.get(k, 0) for k in (1, 2, 3))
        nshort = sum(1 for r in block if r["info"]["length"] == "short")
        c, g, p = prov.get("CLAUDE", 0), prov.get("GEMINI", 0), prov.get("CHATGPT", 0)
        if w < 15:
            ncs = "/".join(str(nc.get(k, 0)) for k in range(5))
            print(f"  {w:3d} | {c}/{g}/{p}           | {ncs}        | {middle:2d}            | {nshort}")
        # Guard: every full window should be provider-balanced and signal-rich.
        if len(block) == 12:
            if max(c, g, p) >= 7 or min(c, g, p) == 0:
                bad.append((w, "provider-skew", (c, g, p)))
            if middle < 6:
                bad.append((w, "low-signal", middle))

    # Global stats.
    print("\n[interleave] global:")
    print("  provider:", Counter(train_sorted["answer"]))
    print("  n_correct:", dict(sorted(Counter(row_nc(r) for r in train_sorted).items())))
    short_idx = [i for i, r in enumerate(train_sorted) if r["info"]["length"] == "short"]
    print(f"  short rows mean position: {sum(short_idx)/len(short_idx):.1f} (of {len(train_sorted)}) "
          f"=> <{len(train_sorted)/2:.0f} confirms short->long drift")

    if bad:
        print(f"\n[interleave] WARNING: {len(bad)} degenerate windows: {bad[:10]}")
    else:
        print("\n[interleave] OK: all full windows provider-balanced (<7,>0) and >=6 middle-difficulty.")

    OUT.mkdir(parents=True, exist_ok=True)
    train_sorted.save_to_disk(str(OUT / "train"))
    val.save_to_disk(str(OUT / "val"))
    print(f"\n[interleave] wrote {len(train_sorted)} train / {len(val)} val to {OUT}")


if __name__ == "__main__":
    main()
