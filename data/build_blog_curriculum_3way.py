"""Reorder the 3-way train split into a curriculum (easy -> hard).

Reads the prebuilt provider split (data/blog_author_id_3way/{train,val}) and the
offline 3-way pass@k predictions (per-blog n_correct in 0..4, markdown=keep,
split=train) and writes a curriculum-ordered train split to
data/blog_author_id_3way_curriculum/{train,val}.

Curriculum ordering (honoring the request "shorter to longer" + "highest
pass-rate to lowest"):
  primary   : length stage  short (0) before long (1)
  secondary : measured pass-rate, n_correct DESCENDING (4 -> 0), easy first
  tertiary  : provider round-robin within each (length, n_correct) tier so every
              consecutive batch stays provider-balanced (avoids feeding a step a
              single-provider batch, which pushes class-collapse).

Train rows that have no measured pass-rate (shouldn't happen) sort as n_correct=2
(neutral). Val is copied through unchanged (it is eval-only, never trained).

The on-disk order is what prime-rl consumes when launched with
PRIME_RL_PRESERVE_DATA_ORDER=1 (TrainSource skips its shuffles).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, load_from_disk

SRC = Path(__file__).resolve().parent / "blog_author_id_3way"
OUT = Path(__file__).resolve().parent / "blog_author_id_3way_curriculum"
PREDS = Path("/home/ubuntu/blogger/blog-eval/logs/multimodel3way_predictions.jsonl")

LENGTH_RANK = {"short": 0, "long": 1}
PROVIDER_ORDER = ["CLAUDE", "CHATGPT", "GEMINI"]


def load_passrates() -> dict[tuple[str, str, str, str], int]:
    """(category, topic, source_model, length) -> n_correct in 0..4 (keep, train).

    Length is part of the key because a blog often has both a short and a long
    version sharing the same (category, topic, source_model)."""
    rates: dict[tuple[str, str, str, str], int] = {}
    for line in PREDS.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("split") != "train" or r.get("markdown") != "keep":
            continue
        rates[(r["category"], r["topic"], r["source_model"], r["length"])] = r["n_correct"]
    return rates


def curriculum_order(train: Dataset, rates: dict[tuple[str, str, str, str], int]) -> list[int]:
    """Return train row indices in curriculum order."""
    # Bucket rows by (length_rank, -n_correct) tier, preserving provider info.
    tiers: dict[tuple[int, int], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for i, row in enumerate(train):
        info = row["info"]
        key = (info["category"], info["topic"], info["source_model"], info["length"])
        n_correct = rates.get(key, 2)  # neutral if unmeasured
        lrank = LENGTH_RANK.get(info["length"], 0)
        tiers[(lrank, -n_correct)][row["answer"]].append(i)

    order: list[int] = []
    for tier_key in sorted(tiers):  # (length asc, -n_correct asc => n_correct desc)
        by_prov = tiers[tier_key]
        cursors = {p: 0 for p in PROVIDER_ORDER}
        remaining = sum(len(v) for v in by_prov.values())
        # Round-robin across providers within the tier.
        while remaining:
            for p in PROVIDER_ORDER:
                lst = by_prov.get(p, [])
                if cursors[p] < len(lst):
                    order.append(lst[cursors[p]])
                    cursors[p] += 1
                    remaining -= 1
    return order


def main() -> None:
    rates = load_passrates()
    print(f"[curriculum] loaded {len(rates)} measured train pass-rates")

    train = load_from_disk(str(SRC / "train"))
    val = load_from_disk(str(SRC / "val"))

    order = curriculum_order(train, rates)
    assert len(order) == len(train), (len(order), len(train))
    assert len(set(order)) == len(train), "duplicate indices in curriculum order"

    train_sorted = train.select(order)

    # Report the resulting curriculum so it can be eyeballed.
    print("[curriculum] first 15 rows (length / n_correct / provider):")
    for i in range(min(15, len(train_sorted))):
        info = train_sorted[i]["info"]
        key = (info["category"], info["topic"], info["source_model"], info["length"])
        print(
            f"  {i:3d}  {info['length']:5s}  nc={rates.get(key, 2)}  {train_sorted[i]['answer']}"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    train_sorted.save_to_disk(str(OUT / "train"))
    val.save_to_disk(str(OUT / "val"))
    print(f"\n[curriculum] wrote {len(train_sorted)} train / {len(val)} val to {OUT}")


if __name__ == "__main__":
    main()
