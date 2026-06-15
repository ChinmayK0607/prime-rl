"""Build a 3-way (provider) train/val split from the local blog corpus.

Task: predict the *provider* of a blog — CLAUDE / CHATGPT / GEMINI — with the
two Gemini variants (gemini-3.1-pro-preview, gemini-3.5-flash) collapsed into a
single GEMINI class. Zero-shot Qwen3.5-9B collapses to predicting GEMINI for
~91% of inputs, so this split is balanced *by provider* (random = 0.333 and the
"always GEMINI" strategy only scores 0.333) to give RL a real signal to break
that bias.

- Uses the short + long length buckets (per request). Frontmatter is stripped
  (it leaks the model). Blog bodies longer than MAX_BODY_TOKENS are dropped so
  prompt + completion fit the 16384 training sequence budget.
- Val = the 3 hardest categories (history, politics, economics-systems) held out
  entirely, balanced by provider; train = the other 6 categories, balanced by
  provider (the larger classes, especially GEMINI, are subsampled to the per-
  split minimum so all three providers have equal counts).

Output: HF datasets under data/blog_author_id_3way/{train,val} with columns
question / answer / info (answer in {CLAUDE, CHATGPT, GEMINI}).
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

BLOG_ROOT = Path("/home/ubuntu/blogger/blogrl/blogs")
OUT_DIR = Path(__file__).resolve().parent / "blog_author_id_3way"
MODEL_ID = "Qwen/Qwen3.5-9B"

# These three categories are held out of training ENTIRELY and saved as a separate
# `val_ood` split, used only as an offline cross-register (out-of-distribution) stress
# test. They are NOT the primary validation metric.
OOD_CATEGORIES = {"history", "politics", "economics-systems"}
# Primary in-distribution validation: hold out this fraction of TOPICS within every
# other category (all provider/length variants of a held-out topic go to val). This
# keeps every category/register in TRAIN while preventing exact-blog AND topic leakage
# (a topic is never split across train and val), so val measures provider-style learning
# on unseen topics rather than cross-register generalization.
HOLDOUT_TOPIC_FRAC = 0.15
LENGTHS = ("__short", "__long")
MAX_BODY_TOKENS = 11800  # leaves room for the 4096-token completion + prompt scaffold

MODEL_TO_PROVIDER = {
    "claude-opus-4.8": "CLAUDE",
    "gpt-5.5": "CHATGPT",
    "gemini-3.1-pro-preview": "GEMINI",
    "gemini-3.5-flash": "GEMINI",
}
SEED = 0


def strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            newline = text.find("\n", end + 1)
            if newline != -1:
                return text[newline + 1 :].lstrip("\n")
    return text


def collect() -> list[dict]:
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    rows: list[dict] = []
    dropped = 0
    for model, provider in MODEL_TO_PROVIDER.items():
        for length in LENGTHS:
            for path in sorted(BLOG_ROOT.glob(f"*/*/{model}{length}.md")):
                body = strip_frontmatter(path.read_text(encoding="utf-8"))
                if len(tok(body).input_ids) > MAX_BODY_TOKENS:
                    dropped += 1
                    continue
                rows.append(
                    {
                        "question": body,
                        "answer": provider,
                        "info": {
                            "category": path.parts[-3],
                            "topic": path.parts[-2],
                            "source_model": model,
                            "length": length.strip("_"),
                        },
                    }
                )
    print(f"[collect] kept {len(rows)} blogs, dropped {dropped} over {MAX_BODY_TOKENS} tokens")
    return rows


def balance_by_provider(rows: list[dict], rng: random.Random) -> list[dict]:
    by_prov: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_prov[r["answer"]].append(r)
    n = min(len(v) for v in by_prov.values())
    out: list[dict] = []
    for prov, rs in by_prov.items():
        rng.shuffle(rs)
        out.extend(rs[:n])
    rng.shuffle(out)
    return out


def summarize(name: str, rows: list[dict]) -> None:
    from collections import Counter

    print(f"[{name}] n={len(rows)}  providers={dict(Counter(r['answer'] for r in rows))}")
    print(f"       lengths={dict(Counter(r['info']['length'] for r in rows))}")
    print(f"       src_models={dict(Counter(r['info']['source_model'] for r in rows))}")


def main() -> None:
    rng = random.Random(SEED)
    rows = collect()

    indist = [r for r in rows if r["info"]["category"] not in OOD_CATEGORIES]
    ood = [r for r in rows if r["info"]["category"] in OOD_CATEGORIES]

    # Topic-level holdout within the in-distribution categories.
    cat_topics: dict[str, set] = defaultdict(set)
    for r in indist:
        cat_topics[r["info"]["category"]].add(r["info"]["topic"])
    heldout_topics: set = set()
    for cat in sorted(cat_topics):
        topics = sorted(cat_topics[cat])
        rng.shuffle(topics)
        k = max(1, round(len(topics) * HOLDOUT_TOPIC_FRAC))
        for t in topics[:k]:
            heldout_topics.add((cat, t))

    def is_heldout(r: dict) -> bool:
        return (r["info"]["category"], r["info"]["topic"]) in heldout_topics

    train = [r for r in indist if not is_heldout(r)]
    val = [r for r in indist if is_heldout(r)]

    # Sanity: no topic leaks across train/val.
    train_topics = {(r["info"]["category"], r["info"]["topic"]) for r in train}
    val_topics = {(r["info"]["category"], r["info"]["topic"]) for r in val}
    assert not (train_topics & val_topics), "topic leakage between train and val!"

    train = balance_by_provider(train, rng)
    val = balance_by_provider(val, rng)
    ood = balance_by_provider(ood, rng)

    summarize("train", train)
    summarize("val", val)
    summarize("val_ood", ood)
    print(f"[split] held out {len(heldout_topics)} topics across "
          f"{len(cat_topics)} in-distribution categories")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(train).save_to_disk(str(OUT_DIR / "train"))
    Dataset.from_list(val).save_to_disk(str(OUT_DIR / "val"))
    Dataset.from_list(ood).save_to_disk(str(OUT_DIR / "val_ood"))
    print(f"\nSaved to {OUT_DIR}/train, {OUT_DIR}/val and {OUT_DIR}/val_ood")


if __name__ == "__main__":
    main()
