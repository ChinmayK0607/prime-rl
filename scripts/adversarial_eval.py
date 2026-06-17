"""Adversarial / robustness eval for the SFT provider classifier.

Hypothesis under test: did the SFT model learn genuine authorial *voice*, or a
brittle surface shortcut (GEMINI=LaTeX/numbered sections, CHATGPT=bold+lists,
CLAUDE=em-dashes/first-person)? Each perturbation strips a class of surface cue
while PRESERVING the author's actual word choice -> the gold label stays valid.
If accuracy survives the 'hard' combo, the model learned real style; if it
craters, we found the shortcut (and RL/harder-SFT regains scope there).

Perturbations (all label-preserving):
  raw         control, untouched
  strip_md    remove markdown headers/bold/italic/blockquote/code/lists/LaTeX/rules/tables
  norm_punct  smart quotes->straight, em/en-dash->'-', ellipsis->..., drop emphasis chars
  lower       lowercase everything (kills Title-Case / header capitalization tells)
  first800    only first 800 chars (forces decision on a short prefix)
  sent_shuffle  shuffle sentence order (destroys structure/position, keeps lexis/voice)
  hard        strip_md + norm_punct + lower + sent_shuffle (all of the above)
"""
import argparse, json, re, random, sys
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style_probe_eval import SYSTEM_PROMPT_3WAY, LABELS, parse_pred

DATA = "/home/ubuntu/blogger/prime-rl/data/blog_author_id_3way_v2"


def strip_md(t):
    t = re.sub(r"```.*?```", " ", t, flags=re.S)          # fenced code
    t = re.sub(r"`([^`]*)`", r"\1", t)                      # inline code
    t = re.sub(r"\$\$.*?\$\$", " ", t, flags=re.S)          # display math
    t = re.sub(r"\$[^$\n]*\$", " ", t)                      # inline math
    t = re.sub(r"\\\[.*?\\\]", " ", t, flags=re.S)
    t = re.sub(r"\\\(.*?\\\)", " ", t, flags=re.S)
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)     # headers
    t = re.sub(r"^\s*>+\s?", "", t, flags=re.M)             # blockquotes
    t = re.sub(r"^\s*[-*+]\s+", "", t, flags=re.M)          # bullet lists
    t = re.sub(r"^\s*\d+[.)]\s+", "", t, flags=re.M)        # numbered lists
    t = re.sub(r"^\s*[-*_]{3,}\s*$", "", t, flags=re.M)     # hr
    t = t.replace("|", " ")                                  # table pipes
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)                # bold
    t = re.sub(r"\*([^*]+)\*", r"\1", t)                    # italic *
    t = re.sub(r"__([^_]+)__", r"\1", t)
    t = re.sub(r"_([^_]+)_", r"\1", t)                      # italic _
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)          # links
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def norm_punct(t):
    for a, b in [("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"), ("—", "-"),
                 ("–", "-"), ("…", "..."), ("•", " "), ("→", " "), ("\u00a0", " ")]:
        t = t.replace(a, b)
    t = re.sub(r"[*_#>`]", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def sent_shuffle(t, seed=0):
    parts = re.split(r"(?<=[.!?])\s+", t)
    rng = random.Random(seed)
    rng.shuffle(parts)
    return " ".join(parts)


PERTS = {
    "raw": lambda t: t,
    "strip_md": strip_md,
    "norm_punct": norm_punct,
    "lower": lambda t: t.lower(),
    "first800": lambda t: t[:800],
    "sent_shuffle": lambda t: sent_shuffle(t),
    "hard": lambda t: sent_shuffle(norm_punct(strip_md(t)).lower()),
}


def eval_one(llm, tok, ds, fn, sp):
    prompts = []
    for r in ds:
        q = fn(r["question"])
        msgs = [{"role": "system", "content": SYSTEM_PROMPT_3WAY},
                {"role": "user", "content": q}]
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
    return {"n": n, "acc": round(correct/n, 4),
            "macro": round(sum(per[l][0]/max(per[l][1],1) for l in LABELS)/3, 4),
            "none": none,
            "per_class": {l: round(per[l][0]/max(per[l][1],1), 3) for l in LABELS},
            "pred_share": {l: round(pred_share.get(l,0)/n, 3) for l in LABELS+["NONE"]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="val_ood")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--perts", default="")  # comma list; empty=all
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ds = load_from_disk(f"{DATA}/{args.split}")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=16384,
              gpu_memory_utilization=0.9, trust_remote_code=True)
    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=4096, seed=0)

    names = [p for p in (args.perts.split(",") if args.perts else PERTS) if p]
    res = {"model": args.model, "split": args.split, "n": len(ds), "perts": {}}
    for name in names:
        r = eval_one(llm, tok, ds, PERTS[name], sp)
        res["perts"][name] = r
        print(f"[{name:12s}] acc={r['acc']:.3f} macro={r['macro']:.3f} "
              f"none={r['none']:3d} per_class={r['per_class']} pred={r['pred_share']}")
    print("\nJSON " + json.dumps(res))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print("wrote " + args.out)


if __name__ == "__main__":
    main()
