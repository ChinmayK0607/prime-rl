"""Build an expanded train/val split from the locally generated blog corpus.

Unlike ``build_blog_split.py`` (which pulls the 126-blog HuggingFace release),
this reads the fresh corpus under ``blogrl/blogs`` so the RL task gets ~2x the
data. Only the ``short`` and default (no-suffix) length buckets are used: both
sit around ~4k tokens (~3000 words), matching the original task length and
leaving room for an 8k completion inside the 16384 sequence budget. The ``long``
bucket (p50 ~9k tokens) is skipped because prompt+completion would overflow.

Val = the 3 hardest categories (history, politics, economics-systems) held out
*entirely*, identical to the original split, so val measures generalization to
unseen domains rather than memorization.

Output: HF datasets saved to disk under data/blog_author_id_v2/{train,val} with
columns question / answer / info, ready for the verifiers SingleTurnEnv.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from datasets import Dataset

BLOG_ROOT = Path("/home/ubuntu/blogger/blogrl/blogs")
OUT_DIR = Path(__file__).resolve().parent / "blog_author_id_v2"

VAL_CATEGORIES = {"history", "politics", "economics-systems"}
MODEL_TO_LABEL = {"claude-opus-4.8": "CLAUDE", "gpt-5.5": "CHATGPT"}
# Length buckets that fit the 16384 / 8192-completion budget.
LENGTH_SUFFIXES = {"": "default", "__short": "short"}


def strip_frontmatter(text: str) -> str:
    """Remove the leading YAML frontmatter block.

    Each blog starts with a ``---`` fenced block that includes ``model:
    <author>`` — passing that into the prompt leaks the gold label, so it must
    be stripped before the blog body is used as the question.
    """
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            newline = text.find("\n", end + 1)
            if newline != -1:
                return text[newline + 1 :].lstrip("\n")
    return text


def collect_records() -> list[dict]:
    rows: list[dict] = []
    for model, label in MODEL_TO_LABEL.items():
        for suffix, length_name in LENGTH_SUFFIXES.items():
            for path in sorted(BLOG_ROOT.glob(f"*/*/{model}{suffix}.md")):
                category = path.parts[-3]
                topic = path.parts[-2]
                body = strip_frontmatter(path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "question": body,
                        "answer": label,
                        "info": {
                            "category": category,
                            "topic": topic,
                            "source_model": model,
                            "length": length_name,
                        },
                    }
                )
    return rows


def summarize(name: str, rows: list[dict]) -> None:
    labels = Counter(r["answer"] for r in rows)
    cats = Counter(r["info"]["category"] for r in rows)
    print(f"[{name}] n={len(rows)}  labels={dict(labels)}")
    for c, n in sorted(cats.items()):
        print(f"    {c:34s} {n}")


def main() -> None:
    rows = collect_records()
    train_rows = [r for r in rows if r["info"]["category"] not in VAL_CATEGORIES]
    val_rows = [r for r in rows if r["info"]["category"] in VAL_CATEGORIES]

    summarize("train", train_rows)
    summarize("val", val_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(train_rows).save_to_disk(str(OUT_DIR / "train"))
    Dataset.from_list(val_rows).save_to_disk(str(OUT_DIR / "val"))
    print(f"\nSaved to {OUT_DIR}/train and {OUT_DIR}/val")


if __name__ == "__main__":
    main()
