"""Eval SFT warmup checkpoints on val + val_ood with the PLAIN prompt (no cheatsheet).

Loads each checkpoint once, evals both splits. Mirrors RL eval sampling
(temp 0.7, top_p 0.95, max 4096, thinking OFF). Target: >0.5, ideally >=0.674.
"""
import argparse, json, sys
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style_probe_eval import SYSTEM_PROMPT_3WAY, LABELS, parse_pred

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "deps/research-environments/environments/blog_author_id"))
from blog_author_id import CHEATSHEET_TRAIN

DATA = "/home/ubuntu/blogger/prime-rl/data/blog_author_id_3way_v2"


def eval_split(llm, tok, split, sp, cheatsheet=False):
    ds = load_from_disk(f"{DATA}/{split}")
    sys_prompt = SYSTEM_PROMPT_3WAY + (CHEATSHEET_TRAIN if cheatsheet else "")
    prompts = []
    for r in ds:
        msgs = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": r["question"]}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                       add_generation_prompt=True, enable_thinking=False))
    outs = llm.generate(prompts, sp)
    n = len(ds); correct = 0; none = 0
    per = {l: [0, 0] for l in LABELS}
    pred_share = {l: 0 for l in LABELS}; pred_share["NONE"] = 0
    for r, o in zip(ds, outs):
        p = parse_pred(o.outputs[0].text); g = r["answer"]
        per[g][1] += 1; pred_share[p] = pred_share.get(p, 0) + 1
        ok = (p == g); correct += ok; per[g][0] += ok
        if p == "NONE": none += 1
    acc = correct / n
    macro = sum(per[l][0]/max(per[l][1],1) for l in LABELS) / 3
    res = {"split": split, "n": n, "acc": round(acc,4), "macro_recall": round(macro,4),
           "none": none,
           "per_class": {l: {"recall": round(per[l][0]/max(per[l][1],1),3),
                             "pred_share": round(pred_share.get(l,0)/n,3)} for l in LABELS}}
    print(f"\n=== {split} n={n} === ACC={acc:.3f} MACRO={macro:.3f} NONE={none} ({none/n:.1%})")
    for l in LABELS:
        c, t = per[l]
        print(f"  {l}: recall {c}/{t}={c/max(t,1):.3f} pred_share={pred_share.get(l,0)/n:.3f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--out", default="")
    ap.add_argument("--cheatsheet", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=16384,
              gpu_memory_utilization=0.9, trust_remote_code=True, enforce_eager=False)
    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=4096, seed=0)

    out = {"model": args.model, "tag": args.tag, "cheatsheet": args.cheatsheet, "results": []}
    for split in ["val", "val_ood"]:
        out["results"].append(eval_split(llm, tok, split, sp, cheatsheet=args.cheatsheet))
    print("\nJSON " + json.dumps(out))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
