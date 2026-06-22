"""E1 — STaR-style on-policy self-distillation data generation.

Contrast with gen_sft_teacher.py (which is GOLD-CONDITIONED: teacher sees the gold label
in an ANALYST NOTE and rationalizes it). Here the model NEVER sees the per-example gold
label while generating; gold is used ONLY as a verifier to gate which self-generated
trajectories become SFT targets.

Pipeline (single round):
  1. PLAIN pass: model + SYSTEM_PROMPT_3WAY (no cheatsheet), thinking-off, k=3 @ temp0.7.
     Gate: majority-vote label == gold AND >=2/3 correct  -> "plain" accepted.
  2. HINT pass (only on not-yet-accepted): model + SYS + CHEATSHEET (train-derived general
     rules, NOT the gold answer), k=2 @ temp0.7. Gate: >=1 of 2 correct -> "hint" accepted.
     Hinted rationales that explicitly cite the cheatsheet/hint/rules are rejected
     (must justify from observable text features only).
  3. Build SFT rows: PLAIN system prompt + blog -> self-generated <reason_why> + canonical
     <answer> gold. (Cheatsheet stripped from the SFT prompt => internalization target.)
  4. Class-balance: cap each class to the min accepted count (report counts before/after).

Zero-leakage framing (per rubber-duck): NO per-example gold label is shown during rationale
generation; gold is used only for verifier-gating on TRAIN. Cheatsheet is train-only derived.
"""
import json, re, random, importlib.util, argparse
from pathlib import Path
from collections import Counter, defaultdict
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3.5-9B"
PROVIDER = {"gpt-5.5": "CHATGPT", "claude-opus-4.8": "CLAUDE",
            "gemini-3.1-pro-preview": "GEMINI", "gemini-3.5-flash": "GEMINI"}
LABELS = ["CLAUDE", "CHATGPT", "GEMINI"]
RW = re.compile(r"<reason_why>(.*?)</reason_why>", re.S | re.I)
ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
# reject hinted rationales that leak the scaffolding rather than cite the text
LEAK_RE = re.compile(r"\b(cheat ?sheet|the hint|known style tells|prior(s)?\b|analyst|"
                     r"as (noted|instructed)|the rules?\b|tell sheet)\b", re.I)


def parse_pred(text):
    m = ANS_RE.search(text)
    seg = (m.group(1) if m else text).upper()
    found = [l for l in LABELS if re.search(r"\b" + l + r"\b", seg)]
    if len(found) == 1:
        return found[0]
    for l in LABELS:
        if re.search(r"\b" + l + r"\b", text.upper()):
            return l
    return "NONE"


def clean_reason(txt, forbid_leak=False):
    m = RW.search(txt)
    if not m:
        return None
    r = " ".join(m.group(1).split())
    w = len(r.split())
    if w < 12 or w > 110:
        return None
    toks = r.lower().split()
    if len(set(toks)) < 0.5 * len(toks):
        return None
    if forbid_leak and LEAK_RE.search(r):
        return None
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k_plain", type=int, default=3)
    ap.add_argument("--k_hint", type=int, default=2)
    ap.add_argument("--out", default="data/blog_sft_star_e1")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    random.seed(0)

    spec = importlib.util.spec_from_file_location("probe", "scripts/style_probe_eval.py")
    probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)
    SYS = probe.SYSTEM_PROMPT_3WAY; CHEAT = probe.CHEATSHEET

    d = load_from_disk("data/blog_author_id_3way_v2/train")
    if args.limit:
        d = d.select(range(min(args.limit, len(d))))
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    items = []  # (blog, gold)
    for ex in d:
        info = ex.get("info") or {}
        gold = PROVIDER.get(info.get("source_model")) or ex["answer"]
        if gold not in LABELS:
            continue
        items.append((ex["question"], gold))
    print(f"train items: {len(items)} | by gold: {Counter(g for _,g in items)}", flush=True)

    llm = LLM(model=MODEL, tensor_parallel_size=args.tp, max_model_len=16384,
              gpu_memory_utilization=0.9, dtype="bfloat16")

    def render(blog, with_cheat):
        msgs = [{"role": "system", "content": SYS + (CHEAT if with_cheat else "")},
                {"role": "user", "content": blog}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)

    # ---- Round 1: PLAIN, k_plain samples ----
    plain_prompts = [render(b, False) for b, _ in items]
    sp_plain = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=400, n=args.k_plain)
    print(f"PLAIN gen: {len(plain_prompts)} prompts x k={args.k_plain}", flush=True)
    plain_outs = llm.generate(plain_prompts, sp_plain)

    accepted = {}   # idx -> dict(blog, gold, reason, source)
    pending = []    # idx of not-yet-accepted
    for i, ((blog, gold), o) in enumerate(zip(items, plain_outs)):
        texts = [c.text for c in o.outputs]
        preds = [parse_pred(t) for t in texts]
        n_correct = sum(p == gold for p in preds)
        maj = Counter(preds).most_common(1)[0][0]
        if maj == gold and n_correct >= max(2, (args.k_plain + 1) // 2):
            reason = None
            for t, p in zip(texts, preds):
                if p == gold:
                    reason = clean_reason(t)
                    if reason:
                        break
            if reason:
                accepted[i] = dict(blog=blog, gold=gold, reason=reason, source="plain")
                continue
        pending.append(i)
    print(f"after PLAIN: accepted {len(accepted)} | pending {len(pending)}", flush=True)

    # ---- Round 2: HINT (cheatsheet) on pending, k_hint samples ----
    if pending:
        hint_prompts = [render(items[i][0], True) for i in pending]
        sp_hint = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=400, n=args.k_hint)
        print(f"HINT gen: {len(hint_prompts)} prompts x k={args.k_hint}", flush=True)
        hint_outs = llm.generate(hint_prompts, sp_hint)
        for i, o in zip(pending, hint_outs):
            blog, gold = items[i]
            for c in o.outputs:
                if parse_pred(c.text) == gold:
                    reason = clean_reason(c.text, forbid_leak=True)
                    if reason:
                        accepted[i] = dict(blog=blog, gold=gold, reason=reason, source="hint")
                        break
    print(f"after HINT: accepted {len(accepted)}", flush=True)

    # ---- report counts by gold x source ----
    by = defaultdict(int)
    for a in accepted.values():
        by[(a["gold"], a["source"])] += 1
    print("accepted by (gold,source):", flush=True)
    for g in LABELS:
        print(f"  {g}: plain={by[(g,'plain')]} hint={by[(g,'hint')]} "
              f"total={by[(g,'plain')]+by[(g,'hint')]}", flush=True)
    per_class = {g: by[(g, 'plain')] + by[(g, 'hint')] for g in LABELS}
    cap = min(per_class.values())
    print(f"class-balance cap = {cap} (min over classes)", flush=True)

    # ---- class-balanced SFT rows ----
    buckets = defaultdict(list)
    for a in accepted.values():
        buckets[a["gold"]].append(a)
    rows = []; final_by = defaultdict(int)
    for g in LABELS:
        random.shuffle(buckets[g])
        for a in buckets[g][:cap]:
            asst = (f"<reason_why>\n{a['reason']}\n</reason_why>\n\n"
                    f"<answer>\n{a['gold']}\nConfidence: HIGH\n</answer>")
            rows.append({"messages": [{"role": "system", "content": SYS},
                                      {"role": "user", "content": a["blog"]},
                                      {"role": "assistant", "content": asst}],
                         "source": a["source"], "gold": a["gold"]})
            final_by[(a["gold"], a["source"])] += 1
    random.shuffle(rows)
    print("FINAL balanced rows:", len(rows), flush=True)
    for g in LABELS:
        print(f"  {g}: plain={final_by[(g,'plain')]} hint={final_by[(g,'hint')]}", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(out / "train.parquet"))
    stats = {"n_items": len(items), "n_accepted": len(accepted), "cap": cap,
             "by_gold_source": {f"{g}/{s}": by[(g, s)] for g in LABELS for s in ("plain", "hint")},
             "final_rows": len(rows)}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print("WROTE", out, "rows", len(rows), flush=True)


if __name__ == "__main__":
    main()
