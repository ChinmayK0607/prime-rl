"""Full-corpus pass@k for the 3-way provider task (train + val + val_ood).

Runs the PLAIN 3-way prompt (no cheatsheet, thinking OFF) and samples k=4 per blog,
then reports pass@1 / pass@4, per-provider recall, prediction distribution, parse-fail
and truncation rates for EACH split of ``data/blog_author_id_3way_v2``. Uses the env's
own system prompt + ``<answer>`` extractor so the numbers match the RL eval exactly
(zero leakage: the cheatsheet never enters the prompt).

    uv run python scripts/passk_fullcorpus.py --splits train,val,val_ood --k 4 --tp 8

Designed to run in the GAP between training runs (it loads its own vLLM instance and
needs the GPUs free).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from datasets import load_from_disk

ENV_DIR = Path(__file__).resolve().parent.parent / "deps/research-environments/environments/blog_author_id"
sys.path.insert(0, str(ENV_DIR))
from blog_author_id import (  # noqa: E402
    SYSTEM_PROMPT_3WAY,
    SYSTEM_PROMPT_3WAY_LEXICAL,
    _ANSWER_RE,
    _LABEL_RE_3WAY,
)

MODEL = "Qwen/Qwen3.5-9B"
DATA = Path(__file__).resolve().parent.parent / "data/blog_author_id_3way_v2"
LABELS = ["CLAUDE", "CHATGPT", "GEMINI"]
PROMPTS = {"default": SYSTEM_PROMPT_3WAY, "lexical": SYSTEM_PROMPT_3WAY_LEXICAL}


def extract_label(text: str) -> str:
    """Env-identical: a standalone label in <answer> (or the tail) or "" on miss."""
    m = _ANSWER_RE.search(text)
    body = m.group(1).upper() if m else text[-400:].upper()
    labels = set(_LABEL_RE_3WAY.findall(body))
    return next(iter(labels)) if len(labels) == 1 else ""


def render(tok, blog, system_prompt):
    return tok.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": blog}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,val,val_ood")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--prompt_variant", choices=list(PROMPTS), default="default")
    ap.add_argument("--out", default="blog-eval/results/passk_fullcorpus.json")
    args = ap.parse_args()
    system_prompt = PROMPTS[args.prompt_variant]

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, trust_remote_code=True, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem, max_model_len=16384)
    tok = llm.get_tokenizer()
    sp = SamplingParams(n=args.k, temperature=args.temp, top_p=args.top_p, max_tokens=args.max_tokens)

    results = {}
    for split in args.splits.split(","):
        split = split.strip()
        ds = load_from_disk(str(DATA / split))
        prompts = [render(tok, ds[i]["question"], system_prompt) for i in range(len(ds))]
        golds = [ds[i]["answer"] for i in range(len(ds))]
        outs = llm.generate(prompts, sp, use_tqdm=True)

        n = len(ds)
        sample_correct = 0
        sample_total = 0
        passk_hit = 0
        parse_fail = 0
        trunc = 0
        preds = Counter()
        per_prov = {L: {"n": 0, "pass1_num": 0, "pass1_den": 0, "passk": 0} for L in LABELS}
        for gold, out in zip(golds, outs):
            per_prov[gold]["n"] += 1
            any_correct = False
            for comp in out.outputs:
                pred = extract_label(comp.text)
                preds[pred or "<none>"] += 1
                sample_total += 1
                per_prov[gold]["pass1_den"] += 1
                if pred == "":
                    parse_fail += 1
                if comp.finish_reason == "length":
                    trunc += 1
                if pred == gold:
                    sample_correct += 1
                    per_prov[gold]["pass1_num"] += 1
                    any_correct = True
            if any_correct:
                passk_hit += 1
                per_prov[gold]["passk"] += 1

        res = {
            "n_blogs": n,
            "k": args.k,
            "pass1": round(sample_correct / sample_total, 4),
            f"pass{args.k}": round(passk_hit / n, 4),
            "parse_fail_rate": round(parse_fail / sample_total, 4),
            "truncation_rate": round(trunc / sample_total, 4),
            "pred_dist": dict(preds),
            "per_provider": {
                L: {
                    "n": per_prov[L]["n"],
                    "pass1": round(per_prov[L]["pass1_num"] / per_prov[L]["pass1_den"], 4) if per_prov[L]["pass1_den"] else None,
                    f"pass{args.k}": round(per_prov[L]["passk"] / per_prov[L]["n"], 4) if per_prov[L]["n"] else None,
                }
                for L in LABELS
            },
        }
        results[split] = res
        print(f"\n=== {split}  n={n}  k={args.k}  (plain, temp={args.temp}) ===")
        print(f"  pass1={res['pass1']}  pass{args.k}={res[f'pass{args.k}']}  "
              f"parse_fail={res['parse_fail_rate']}  trunc={res['truncation_rate']}")
        for L in LABELS:
            p = res["per_provider"][L]
            print(f"  {L:<8} n={p['n']:<5} pass1={p['pass1']}  pass{args.k}={p[f'pass{args.k}']}")
        print(f"  pred_dist={res['pred_dist']}")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({"model": args.model, "prompt_variant": args.prompt_variant, "splits": results}, indent=2))
    print(f"\nWROTE {outp}")


if __name__ == "__main__":
    main()
