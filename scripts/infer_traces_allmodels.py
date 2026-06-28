"""Plain-prompt trace inference for ONE Hub model on val + val_ood.

Run once per model (fresh process => clean vLLM/GPU). Downloads the model from the Hub to a
local temp dir, generates one completion per example under the PLAIN 3-way prompt (the same
SYSTEM_PROMPT_3WAY + <answer> extractor as every other eval, so numbers are comparable across
the whole method ladder), writes per-split JSONL traces + a summary.json, then deletes the
downloaded weights.

Usage:
  python scripts/infer_traces_allmodels.py --repo CK0607/<name> --short <name> [--out_dir blog-eval/traces]
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd  # noqa: F401  (datasets backend)

ENV_DIR = Path(__file__).resolve().parent.parent / "deps/research-environments/environments/blog_author_id"
sys.path.insert(0, str(ENV_DIR))
from blog_author_id import SYSTEM_PROMPT_3WAY, _ANSWER_RE, _LABEL_RE_3WAY  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data/blog_author_id_3way_v2"
LABELS = ["CLAUDE", "CHATGPT", "GEMINI"]


def extract_label(text: str) -> str:
    m = _ANSWER_RE.search(text)
    body = m.group(1).upper() if m else text[-400:].upper()
    labels = set(_LABEL_RE_3WAY.findall(body))
    return next(iter(labels)) if len(labels) == 1 else ""


def render(tok, blog):
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT_3WAY}, {"role": "user", "content": blog}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--short", required=True)
    ap.add_argument("--out_dir", default="blog-eval/traces")
    ap.add_argument("--splits", default="val,val_ood")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--tmp_root", default="tmp_models")
    ap.add_argument("--render_model", default="Qwen/Qwen3.5-9B",
                    help="Tokenizer used for chat-template rendering (finetunes share the base "
                         "template/vocab; the pushed repos may not carry a chat_template).")
    args = ap.parse_args()

    from datasets import load_from_disk
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    render_tok = AutoTokenizer.from_pretrained(args.render_model, trust_remote_code=True)
    if render_tok.chat_template is None:
        raise SystemExit(f"render_model {args.render_model} has no chat_template")

    local_dir = Path(args.tmp_root) / args.short
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    is_local = Path(args.repo).is_dir()
    if is_local:
        local_dir = Path(args.repo)
        print(f"[{args.short}] using local checkpoint {local_dir} (no download)", flush=True)
    else:
        print(f"[{args.short}] downloading {args.repo} -> {local_dir}", flush=True)
        snapshot_download(repo_id=args.repo, local_dir=str(local_dir),
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "tokenizer*", "*.model"])

    out_model = Path(args.out_dir) / args.short
    out_model.mkdir(parents=True, exist_ok=True)
    try:
        llm = LLM(model=str(local_dir), trust_remote_code=True, tensor_parallel_size=args.tp,
                  gpu_memory_utilization=args.gpu_mem, max_model_len=16384)
        sp = SamplingParams(n=1, temperature=args.temp, top_p=args.top_p, max_tokens=args.max_tokens)

        summary = {"repo": args.repo, "short": args.short, "prompt": "plain_SYSTEM_PROMPT_3WAY",
                   "render_model": args.render_model,
                   "sampling": {"temp": args.temp, "top_p": args.top_p, "max_tokens": args.max_tokens},
                   "splits": {}}
        for split in args.splits.split(","):
            split = split.strip()
            ds = load_from_disk(str(DATA / split))
            prompts = [render(render_tok, ds[i]["question"]) for i in range(len(ds))]
            golds = [ds[i]["answer"] for i in range(len(ds))]
            outs = llm.generate(prompts, sp, use_tqdm=True)

            n = len(ds)
            correct = parse_fail = trunc = 0
            preds = Counter()
            per_prov = {L: {"n": 0, "correct": 0} for L in LABELS}
            jsonl_path = out_model / f"{split}.jsonl"
            with open(jsonl_path, "w") as fh:
                for i, (gold, out) in enumerate(zip(golds, outs)):
                    comp = out.outputs[0]
                    pred = extract_label(comp.text)
                    is_trunc = comp.finish_reason == "length"
                    is_corr = pred == gold
                    preds[pred or "<none>"] += 1
                    per_prov[gold]["n"] += 1
                    if pred == "":
                        parse_fail += 1
                    if is_trunc:
                        trunc += 1
                    if is_corr:
                        correct += 1
                        per_prov[gold]["correct"] += 1
                    fh.write(json.dumps({
                        "idx": i, "split": split, "gold": gold, "pred": pred,
                        "correct": is_corr, "truncated": is_trunc,
                        "finish_reason": comp.finish_reason, "n_tokens": len(comp.token_ids),
                        "completion": comp.text, "blog": ds[i]["question"],
                    }) + "\n")

            res = {
                "n": n,
                "accuracy": round(correct / n, 4),
                "parse_fail_rate": round(parse_fail / n, 4),
                "truncation_rate": round(trunc / n, 4),
                "pred_dist": dict(preds),
                "per_provider": {L: {"n": per_prov[L]["n"],
                                     "accuracy": round(per_prov[L]["correct"] / per_prov[L]["n"], 4)
                                     if per_prov[L]["n"] else None} for L in LABELS},
                "jsonl": str(jsonl_path),
            }
            summary["splits"][split] = res
            print(f"[{args.short}] {split}: acc={res['accuracy']} trunc={res['truncation_rate']} "
                  f"parse_fail={res['parse_fail_rate']} preds={res['pred_dist']}", flush=True)

        (out_model / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[{args.short}] WROTE {out_model}/summary.json", flush=True)
    finally:
        if not is_local:
            shutil.rmtree(local_dir, ignore_errors=True)
            print(f"[{args.short}] deleted {local_dir}", flush=True)
        else:
            print(f"[{args.short}] kept local checkpoint {local_dir}", flush=True)


if __name__ == "__main__":
    main()
