"""E0 — catastrophic-forgetting / capability baseline (eval only).

Compares BASE Qwen3.5-9B vs the SFT-to-1.000 checkpoint (sft_warmup/step_180) on
UNRELATED-domain text the SFT never touched (gsm8k math + generic English), to quantify
how much the blog-classification SFT damaged general capability. This is the yardstick the
OPSD ladder (E1-E4) must beat: a method that internalizes the tells WITHOUT this forgetting
cost is strictly better.

Two metrics, both zero-leakage wrt the blog task:
  (1) teacher-forced perplexity on held-out general text (gsm8k test + generic paragraphs)
  (2) greedy capability generations on a fixed probe set (qualitative side-by-side)

Runs one model at a time (bf16, single GPU) to keep memory modest.
"""

import argparse
import json
import math
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BASE = "Qwen/Qwen3.5-9B"
SFT = "/home/ubuntu/blogger/prime-rl/outputs/sft_warmup/weights/step_180"

# Generic English paragraphs (diverse domains, none about AI-provenance/blog classification).
GENERIC = [
    "The migration patterns of Arctic terns span nearly the entire globe; each year the birds "
    "travel from their breeding grounds in the Arctic to the Antarctic and back, covering a "
    "round-trip distance that is among the longest of any animal on Earth.",
    "In thermodynamics, the second law states that the total entropy of an isolated system can "
    "never decrease over time, and is constant if and only if all processes are reversible. "
    "Isolated systems spontaneously evolve toward thermodynamic equilibrium.",
    "The Treaty of Westphalia, signed in 1648, is often cited by historians as a foundational "
    "moment in the development of the modern state system, establishing principles of "
    "territorial sovereignty that continue to shape international relations.",
    "A balanced diet provides the body with the nutrients it needs to function correctly. To get "
    "the nutrition it requires, most of a person's daily calories should come from fresh fruits, "
    "vegetables, whole grains, legumes, nuts, and lean proteins.",
]

# Fixed capability probes (unrelated to the blog task).
PROBES = [
    ("math", "What is 47 times 23? Show your work briefly."),
    ("math", "If a train travels 60 miles in 1.5 hours, what is its average speed in miles per hour?"),
    ("factual", "Name the capital of Australia and one fact about it."),
    ("factual", "Who wrote the play 'Hamlet' and roughly when?"),
    ("reasoning", "A farmer has 17 sheep. All but 9 run away. How many are left? Explain."),
    ("code", "Write a one-line Python expression that returns the sum of squares from 1 to n."),
]


def load_gsm8k_text(n: int) -> list[str]:
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        out = []
        for i in range(min(n, len(ds))):
            out.append(ds[i]["question"].strip() + "\n" + ds[i]["answer"].strip())
        return out
    except Exception as e:
        print(f"[warn] gsm8k load failed ({e}); using generic only")
        return []


@torch.no_grad()
def perplexity(model, tok, texts: list[str]) -> float:
    """Mean per-token perplexity over the given texts (each scored independently)."""
    total_nll, total_tok = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(model.device)
        if ids.shape[1] < 2:
            continue
        out = model(ids, labels=ids)
        # HF returns mean NLL over (seq_len-1) tokens; reweight to accumulate a corpus-level mean.
        ntok = ids.shape[1] - 1
        total_nll += out.loss.item() * ntok
        total_tok += ntok
    return math.exp(total_nll / total_tok) if total_tok else float("nan")


@torch.no_grad()
def generate(model, tok, prompt: str) -> str:
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(ids, max_new_tokens=128, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def run_model(name: str, path: str, ppl_texts: list[str]) -> dict:
    print(f"\n==== loading {name}: {path} ====")
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to("cuda:0").eval()
    ppl = perplexity(model, tok, ppl_texts)
    print(f"{name} perplexity (general text) = {ppl:.3f}")
    gens = []
    for cat, p in PROBES:
        g = generate(model, tok, p)
        gens.append({"cat": cat, "prompt": p, "output": g})
        print(f"  [{cat}] {p}\n    -> {g[:180]}")
    del model
    torch.cuda.empty_cache()
    return {"perplexity": ppl, "generations": gens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_gsm8k", type=int, default=150)
    ap.add_argument("--out", default="probe_results/e0_forgetting.json")
    args = ap.parse_args()

    ppl_texts = GENERIC + load_gsm8k_text(args.n_gsm8k)
    print(f"Perplexity corpus: {len(ppl_texts)} texts ({len(GENERIC)} generic + gsm8k)")

    res = {
        "base": run_model("BASE", BASE, ppl_texts),
        "sft_step180": run_model("SFT_step180", SFT, ppl_texts),
        "n_ppl_texts": len(ppl_texts),
    }
    b, s = res["base"]["perplexity"], res["sft_step180"]["perplexity"]
    res["ppl_delta_pct"] = 100.0 * (s - b) / b
    print(f"\n==== FORGETTING: base ppl {b:.3f} -> sft ppl {s:.3f} "
          f"({res['ppl_delta_pct']:+.1f}%) ====")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
