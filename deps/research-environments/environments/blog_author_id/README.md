# blog-author-id

Single-turn forensic classification: given a ~3000-word blog post, decide whether
it was written by **Claude** (`claude-opus-4.8`) or **ChatGPT** (`gpt-5.5`).
Thinking is OFF — the policy answers directly.

## Task

- **Input**: one blog post (the `question` column).
- **Output**: a `<reason_why>` block (2-4 sentences) followed by an `<answer>`
  tag containing `CLAUDE` or `CHATGPT` plus a confidence line.
- **Reward**: binary exact match between the parsed `<answer>` label and the gold
  author (`correct_answer`, weight 1.0). `parsed_ok` is a weight-0 diagnostic that
  reports whether a parseable label was produced at all.

The parser (`extract_label`) reads the `<answer>` tag; an answer that names both
labels or neither counts as a miss. With no tag it falls back to scanning the tail
of the completion where the conclusion lives.

## Data

Splits are HuggingFace datasets saved to disk under
`<repo>/data/blog_author_id/{train,val}` (override the base dir with the
`BLOG_DATA_DIR` env var, or pass `data_dir=...` to `load_environment`). Columns:
`question`, `answer` (`CLAUDE`/`CHATGPT`), `info` (`category`, `topic`,
`source_model`).

The split is built by `<repo>/data/build_blog_split.py`. **Validation holds out
the 3 hardest categories entirely** (`history`, `politics`, `economics-systems`)
so val measures generalization to unseen domains rather than memorization:
train = 84 examples, val = 42 examples, both balanced 50/50.

## Arguments

`load_environment(split="train", data_dir=None)`

- `split`: `"train"` or `"val"` (any subdir saved under the data dir).
- `data_dir`: optional override for the dataset base directory.

## Usage

```bash
# single verbose rollout
uv run vf-eval --env blog-author-id -d -v -n1 -r1

# more rollouts, saved for analysis
uv run vf-eval --env blog-author-id -n5 -r4 -s
```

The reward is verifiable and binary, which makes this a clean GRPO target: the
offline pass@4 baseline shows a large pass@1 (~56%) -> pass@4 (~94%) gap, i.e.
substantial headroom for RL to convert into pass@1 gains.
