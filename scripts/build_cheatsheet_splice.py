"""Build the cheatsheet teacher-prompt splice for OPCD/RLSD (experiments E2/E3).

The teacher in OPCD/RLSD is the SAME policy conditioned on the train-derived
CHEATSHEET in-context. At rollout time the student is prompted PLAIN; the teacher
must score the same completion under a cheatsheet-augmented prompt. Because the
env appends CHEATSHEET_TRAIN to the END of the system message, the plain and
cheat prompts differ ONLY by a contiguous block inside the system segment, with
an IDENTICAL tail (user blog + generation prompt).

This script renders plain and cheat full prompts for several blogs, derives the
differing leading system prefixes as the longest common LEADING prefix across
blogs, and verifies the exact splice identity ``cheat_prefix + plain_tail ==
cheat_prompt`` per blog before writing ``{plain_prefix_ids, cheat_prefix_ids}``
for the orchestrator to splice at train time (see compute_teacher_logprobs).

    uv run python scripts/build_cheatsheet_splice.py \
        --out data/cheatsheet_splice_3way.json
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer

ENV_DIR = Path(__file__).resolve().parent.parent / "deps/research-environments/environments/blog_author_id"
sys.path.insert(0, str(ENV_DIR))
from blog_author_id import (  # noqa: E402
    CHEATSHEET_TRAIN,
    SYSTEM_PROMPT_3WAY,
    SYSTEM_PROMPT_3WAY_ANSWERONLY,
)

MODEL = "Qwen/Qwen3.5-9B"
DATA = Path(__file__).resolve().parent.parent / "data/blog_author_id_3way_v2"


def render(tok, system, blog):
    text = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": blog}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return tok.encode(text, add_special_tokens=False)


def common_prefix_len(seqs):
    n = min(len(s) for s in seqs)
    for i in range(n):
        col = seqs[0][i]
        if any(s[i] != col for s in seqs):
            return i
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cheatsheet_splice_3way.json")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--answer_only", action="store_true",
                    help="Use the answer-only system prompt (must match the rollout env).")
    ap.add_argument("--n_probe", type=int, default=12)
    args = ap.parse_args()

    sys_prompt = SYSTEM_PROMPT_3WAY_ANSWERONLY if args.answer_only else SYSTEM_PROMPT_3WAY
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    ds = load_from_disk(str(DATA / "train"))
    blogs = [ds[i]["question"] for i in range(min(args.n_probe, len(ds)))]
    # Synthetic short/edge blogs to stress prefix blog-independence.
    blogs += ["short.", "A different opening sentence entirely, with punctuation!", "x"]

    plain_full = [render(tok, sys_prompt, b) for b in blogs]
    cheat_full = [render(tok, sys_prompt + CHEATSHEET_TRAIN, b) for b in blogs]

    # The blog-independent leading system region is the longest common leading
    # prefix across blogs (the blogs diverge right after the system block).
    plain_prefix = plain_full[0][: common_prefix_len(plain_full)]
    cheat_prefix = cheat_full[0][: common_prefix_len(cheat_full)]

    # Verify the EXACT identity the orchestrator splice relies on: replacing the
    # plain prefix with the cheat prefix reconstructs the cheat prompt token-for-token.
    for blog, pf, cf in zip(blogs, plain_full, cheat_full):
        if list(pf[: len(plain_prefix)]) != list(plain_prefix):
            raise RuntimeError(f"plain prefix is not a leading prefix of a plain prompt (blog={blog[:40]!r}).")
        spliced = list(cheat_prefix) + list(pf[len(plain_prefix):])
        if spliced != list(cf):
            raise RuntimeError(
                "Splice identity FAILED: cheat_prefix + plain_tail != cheat prompt — the chat "
                f"template/tokenizer does not splice cleanly (blog={blog[:40]!r}). Splice unsafe."
            )

    plain_prefix = list(plain_prefix)
    cheat_prefix = list(cheat_prefix)
    out = {
        "model": args.model,
        "answer_only": args.answer_only,
        "plain_prefix_ids": plain_prefix,
        "cheat_prefix_ids": cheat_prefix,
        "plain_prefix_len": len(plain_prefix),
        "cheat_prefix_len": len(cheat_prefix),
        "cheatsheet_tokens": len(cheat_prefix) - len(plain_prefix),
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"WROTE {outp}  (splice identity verified on {len(blogs)} blogs)")
    print(f"  plain prefix {len(plain_prefix)} tok -> cheat prefix {len(cheat_prefix)} tok "
          f"(+{len(cheat_prefix) - len(plain_prefix)} cheatsheet tokens)")
    div = next((i for i in range(min(len(plain_prefix), len(cheat_prefix)))
                if plain_prefix[i] != cheat_prefix[i]), len(plain_prefix))
    delta = len(cheat_prefix) - len(plain_prefix)
    print(f"  cheatsheet inserted at token {div}; starts: "
          f"{tok.decode(cheat_prefix[div:div + min(delta, 40)])!r}")


if __name__ == "__main__":
    main()
