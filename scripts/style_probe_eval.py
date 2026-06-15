"""Standalone vLLM eval for the 3-way provider task (Design B / C probes).

Conditions (no training): baseline prompt, +fingerprint cheatsheet, +few-shot
exemplars, and a thinking-ON variant. Mirrors the RL eval sampling (temp 0.7,
top_p 0.95, max 4096) and prompt (SYSTEM_PROMPT_3WAY), thinking OFF by default.
"""
import argparse, json, re, random
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3.5-9B"
DATA = "/home/ubuntu/blogger/prime-rl/data/blog_author_id_3way_v2"

SYSTEM_PROMPT_3WAY = """You are an expert forensic analyst of AI-generated text.

## TASK
You will be given a piece of text. It was written by exactly one of these three AI providers. Identify which one:
- CLAUDE — Anthropic's Claude
- CHATGPT — OpenAI's ChatGPT
- GEMINI — Google's Gemini

Examine the text from any angle you find useful. Judge only HOW the text is written, never WHAT it is about.
Rely on your own qualitative judgment of the writing's style and voice — weigh it holistically, not from any single superficial cue.

## OUTPUT FORMAT
Respond using exactly these two tags, nothing outside them:

<reason_why>
2-4 sentences. State only the most decisive evidence for your conclusion. Be specific and grounded in the text.
</reason_why>

<answer>
Exactly one of: CLAUDE / CHATGPT / GEMINI
Confidence: HIGH / MEDIUM / LOW
</answer>

## RULES
- Always classify. Never refuse or skip a tag.
- Choose exactly one provider.
- Your response MUST begin with the literal tag <reason_why> and contain NOTHING before it (no preamble, no restating the task).
- Keep <reason_why> to AT MOST 4 short sentences. Do NOT deliberate at length, enumerate many points, list every feature, second-guess, back-track, or repeat yourself. Commit to your single best judgment.
- You MUST close </reason_why> and then emit the full <answer> block. Never run out of room before answering.
- Do NOT output <think> or </think> tags, or any text before <reason_why>; put all of your reasoning only inside <reason_why>.
- If signals conflict, say so briefly and reflect that in your confidence."""

CHEATSHEET = """

## KNOWN STYLE TELLS (empirically derived; use as priors, still judge holistically)
- CHATGPT — hedging and enumerative: frequent "may", "may be", "can also", "for example", "such as", "not only ... but", "depends on"; qualifies claims; lists and numbered structure; even, balanced, slightly cautious register.
- CLAUDE — conversational and sincere: second-person "you", sincerity adverbs ("genuinely", "honestly", "precisely", "exactly"), "worth", "the honest truth"; essayistic, argument-driven, hedged but warm first-person voice.
- GEMINI — grandiose and formal: intensifiers ("highly", "massive", "profound", "fundamentally"), discourse markers ("however", "furthermore", "we must"), "paradigm", heavy use of mathematical/technical framing, confident declarative register, ASCII diagrams/notation."""

ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
LABELS = ["CLAUDE", "CHATGPT", "GEMINI"]

def parse_pred(text):
    m = ANS_RE.search(text)
    seg = m.group(1) if m else text
    seg_u = seg.upper()
    found = [l for l in LABELS if re.search(r"\b"+l+r"\b", seg_u)]
    if len(found) == 1:
        return found[0]
    # fall back: first label mention in whole text
    for l in LABELS:
        if re.search(r"\b"+l+r"\b", text.upper()):
            return l
    return "NONE"

def build_fewshot(train, tok, n_chars=1100, seed=0):
    rng = random.Random(seed)
    by = {l: [r for r in train if r["answer"] == l] for l in LABELS}
    msgs = []
    for l in LABELS:
        r = rng.choice(by[l])
        excerpt = r["question"][:n_chars]
        msgs.append({"role": "user", "content": f"TEXT (excerpt):\n{excerpt}"})
        msgs.append({"role": "assistant", "content": f"<reason_why>\nExemplar of {l} style.\n</reason_why>\n\n<answer>\n{l}\nConfidence: HIGH\n</answer>"})
    return msgs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "cheatsheet", "fewshot"], default="baseline")
    ap.add_argument("--split", default="val")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ds = load_from_disk(f"{DATA}/{args.split}")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    sys_prompt = SYSTEM_PROMPT_3WAY + (CHEATSHEET if args.mode == "cheatsheet" else "")
    fewshot = build_fewshot(load_from_disk(f"{DATA}/train"), tok) if args.mode == "fewshot" else []

    prompts = []
    for r in ds:
        msgs = [{"role": "system", "content": sys_prompt}] + fewshot + [
            {"role": "user", "content": r["question"]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=args.thinking)
        prompts.append(text)

    llm = LLM(model=MODEL, tensor_parallel_size=args.tp, max_model_len=16384,
              gpu_memory_utilization=0.9, trust_remote_code=True, enforce_eager=False)
    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=4096, seed=0)
    outs = llm.generate(prompts, sp)

    correct = 0; n = len(ds); none = 0
    per = {l: [0, 0] for l in LABELS}  # [correct, total] by gold
    pred_share = {l: 0 for l in LABELS}; pred_share["NONE"] = 0
    recs = []
    for r, o in zip(ds, outs):
        gen = o.outputs[0].text
        p = parse_pred(gen)
        g = r["answer"]
        per[g][1] += 1
        pred_share[p] = pred_share.get(p, 0) + 1
        ok = (p == g)
        correct += ok
        per[g][0] += ok
        if p == "NONE": none += 1
        recs.append({"gold": g, "pred": p, "ok": ok, "len": len(gen)})

    acc = correct / n
    macro = sum(per[l][0]/max(per[l][1],1) for l in LABELS) / 3
    print(f"\n=== MODE={args.mode} thinking={args.thinking} split={args.split} n={n} ===")
    print(f"ACC={acc:.3f}  MACRO-RECALL={macro:.3f}  NONE={none} ({none/n:.1%})")
    for l in LABELS:
        c, t = per[l]
        print(f"  {l}: recall {c}/{t}={c/max(t,1):.3f}  pred_share={pred_share.get(l,0)/n:.3f}")
    if args.out:
        Path(args.out).write_text("\n".join(json.dumps(x) for x in recs))
        print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
