"""Add formatting-strip augmentation to the interleaved 3-way train split.

Motivation: the blogs are ~97% separable by FORMATTING alone (em-dash, markdown,
length). If the model is allowed to bank only on those surface cues, the learned
<reason_why> stays shallow. Stripping markdown/formatting from a fraction of the
TRAIN inputs forces the policy onto deeper stylistic/lexical signal (which is also
~100% separable, so stripped rows stay learnable, not noise). This makes the
emergent, RL-learned reasoning more likely to reflect real style rather than a
single shortcut.

- Loads the already curriculum-ordered interleaved train (preserves batch-aware
  provider/difficulty windows).
- Strips a seeded STRIP_FRAC of train rows; records info["stripped"] (bool) so
  analysis/harvest can separate the two regimes.
- Val is copied through UNCHANGED (formatted) so val tracks the same baseline and
  eval stays cheap; a stripped-val ablation can be run offline later.

Output: data/blog_author_id_3way_aug/{train,val}
"""

from __future__ import annotations

import random
import re
from pathlib import Path

from datasets import load_from_disk

SRC = Path(__file__).resolve().parent / "blog_author_id_3way_interleaved"
OUT = Path(__file__).resolve().parent / "blog_author_id_3way_aug"
STRIP_FRAC = 0.40
SEED = 0


def strip_formatting(t: str) -> str:
    """Remove markdown/formatting cues while preserving wording.

    Identical normalization to the content-signal probe that scored ~100% val
    accuracy on stripped text (so stripped rows remain learnable)."""
    t = re.sub(r"```.*?```", " ", t, flags=re.S)        # code fences
    t = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", t)          # headers
    t = re.sub(r"(?m)^\s*[-*+]\s+", "", t)               # bullets
    t = re.sub(r"(?m)^\s*\d+\.\s+", "", t)               # numbered lists
    t = t.replace("**", "").replace("__", "")            # bold
    t = re.sub(r"(?<!\*)\*(?!\*)", "", t)                # italics *
    t = t.replace("\u2014", ", ").replace("\u2013", "-")  # em/en dash -> neutral
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)                        # collapse blank lines
    return t.strip()


def main() -> None:
    train = load_from_disk(str(SRC / "train"))
    val = load_from_disk(str(SRC / "val"))

    rng = random.Random(SEED)
    n = len(train)
    strip_idx = set(rng.sample(range(n), int(round(n * STRIP_FRAC))))

    def transform(row, idx):
        info = dict(row["info"])
        if idx in strip_idx:
            row["question"] = strip_formatting(row["question"])
            info["stripped"] = True
        else:
            info["stripped"] = False
        row["info"] = info
        return row

    train_aug = train.map(transform, with_indices=True)

    # val: mark stripped=False, keep text unchanged.
    def mark_val(row):
        info = dict(row["info"])
        info["stripped"] = False
        row["info"] = info
        return row

    val_aug = val.map(mark_val)

    # Report.
    n_strip = sum(1 for r in train_aug if r["info"]["stripped"])
    import statistics as st
    flen = [len(r["question"]) for r in train_aug if not r["info"]["stripped"]]
    slen = [len(r["question"]) for r in train_aug if r["info"]["stripped"]]
    print(f"[aug] train {len(train_aug)}: stripped {n_strip} ({n_strip/len(train_aug):.0%}), formatted {len(train_aug)-n_strip}")
    print(f"[aug] median chars: formatted={st.median(flen):.0f}  stripped={st.median(slen):.0f}")
    # spot-check provider balance is preserved among stripped rows
    from collections import Counter
    print(f"[aug] stripped provider balance: {dict(Counter(r['answer'] for r in train_aug if r['info']['stripped']))}")
    print(f"[aug] sample stripped text head:\n  {next(r['question'] for r in train_aug if r['info']['stripped'])[:200]!r}")

    OUT.mkdir(parents=True, exist_ok=True)
    train_aug.save_to_disk(str(OUT / "train"))
    val_aug.save_to_disk(str(OUT / "val"))
    print(f"\n[aug] wrote {len(train_aug)} train / {len(val_aug)} val to {OUT}")


if __name__ == "__main__":
    main()
