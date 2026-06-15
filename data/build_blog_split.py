"""Build train/val split for the blog author-detection RL task.

Val = the 3 hardest categories (lowest baseline pass@1: history, politics,
economics-systems) held out *entirely*. This makes the validation set both the
hardest domains and a true generalization test (topics never seen in training),
so an improvement on val reflects real learning rather than memorization.

Output: HF datasets saved to disk under data/blog_author_id/{train,val} with
columns question / answer / info, ready for a verifiers SingleTurnEnv.
"""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, load_dataset

DATASET_ID = "anonymousNeurIPS2026submission4281/copilot-sdk-blogs"
OUT_DIR = Path(__file__).resolve().parent / "blog_author_id"

# Held-out validation categories (3 lowest baseline pass@1).
VAL_CATEGORIES = {"history", "politics", "economics-systems"}

MODEL_TO_LABEL = {"claude-opus-4.8": "CLAUDE", "gpt-5.5": "CHATGPT"}


def to_record(rec: dict) -> dict:
    return {
        "question": rec["content"],
        "answer": MODEL_TO_LABEL[rec["model"]],
        "info": {
            "category": rec["category"],
            "topic": rec["topic"],
            "source_model": rec["model"],
        },
    }


def summarize(name: str, rows: list[dict]) -> None:
    from collections import Counter

    labels = Counter(r["answer"] for r in rows)
    cats = Counter(r["info"]["category"] for r in rows)
    print(f"[{name}] n={len(rows)}  labels={dict(labels)}")
    for c, n in sorted(cats.items()):
        print(f"    {c:34s} {n}")


def main() -> None:
    ds = load_dataset(DATASET_ID, split="train")
    train_rows, val_rows = [], []
    for rec in ds:
        out = to_record(rec)
        if rec["category"] in VAL_CATEGORIES:
            val_rows.append(out)
        else:
            train_rows.append(out)

    summarize("train", train_rows)
    summarize("val", val_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(train_rows).save_to_disk(str(OUT_DIR / "train"))
    Dataset.from_list(val_rows).save_to_disk(str(OUT_DIR / "val"))
    print(f"\nSaved to {OUT_DIR}/train and {OUT_DIR}/val")


if __name__ == "__main__":
    main()
