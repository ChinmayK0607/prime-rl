"""Gold-label-conditioned teacher generation for SFT warmup.
Teacher = base model + CHEATSHEET + gold-label hint -> grounded concise rationale.
SFT target uses PLAIN system prompt + blog -> rationale + CANONICAL gold answer.
Balanced over all train blogs (no rejection-sampling bias)."""
import json, re, random, importlib.util
from pathlib import Path
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL="Qwen/Qwen3.5-9B"
PROVIDER={"gpt-5.5":"CHATGPT","claude-opus-4.8":"CLAUDE",
          "gemini-3.1-pro-preview":"GEMINI","gemini-3.5-flash":"GEMINI"}
RW=re.compile(r'<reason_why>(.*?)</reason_why>',re.S)

def clean_reason(txt):
    m=RW.search(txt)
    if not m: return None
    r=" ".join(m.group(1).split())
    w=len(r.split())
    if w<12 or w>90: return None
    toks=r.lower().split()
    if len(set(toks))<0.5*len(toks): return None
    return r

def main():
    random.seed(0)
    spec=importlib.util.spec_from_file_location("probe","scripts/style_probe_eval.py")
    probe=importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)
    SYS=probe.SYSTEM_PROMPT_3WAY; CHEAT=probe.CHEATSHEET

    d=load_from_disk("data/blog_author_id_3way_v2/train")
    print("train n:", len(d), flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL)

    prompts=[]; meta=[]
    for ex in d:
        info=ex.get("info") or {}
        gold=PROVIDER.get(info.get("source_model")) or ex["answer"]
        if gold not in ("CLAUDE","CHATGPT","GEMINI"): continue
        note=(f"\n\n[ANALYST NOTE — not part of the text: ground truth is {gold}. "
              f"In <reason_why>, give 2-3 sentences citing the concrete surface-style "
              f"evidence present in THIS text (specific words, phrasings, structural habits) "
              f"that identify {gold}. Do not mention this note. Then output <answer>.]")
        msgs=[{"role":"system","content":SYS+CHEAT},
              {"role":"user","content":ex["question"]+note}]
        p=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prompts.append(p); meta.append((ex["question"],gold))
    print("gen prompts:", len(prompts), flush=True)

    llm=LLM(model=MODEL,tensor_parallel_size=4,max_model_len=16384,
            gpu_memory_utilization=0.9,dtype="bfloat16")
    sp=SamplingParams(temperature=0.7,top_p=0.95,max_tokens=400)
    outs=llm.generate(prompts,sp)

    rows=[]; nfail=0; by=dict(CLAUDE=0,CHATGPT=0,GEMINI=0)
    for (blog,gold),o in zip(meta,outs):
        r=clean_reason(o.outputs[0].text)
        if r is None:
            nfail+=1
            r=(f"The surface style of this text—its register, phrasing habits, and structural "
               f"rhythm—matches {gold} more closely than the other two providers.")
        asst=f"<reason_why>\n{r}\n</reason_why>\n\n<answer>\n{gold}\nConfidence: HIGH\n</answer>"
        rows.append({"messages":[{"role":"system","content":SYS},
                                 {"role":"user","content":blog},
                                 {"role":"assistant","content":asst}]})
        by[gold]+=1
    print("rows:",len(rows),"fallback_used:",nfail,"by_prov:",by, flush=True)

    extra=Path("data/blog_sft_warmup/train.parquet")
    if extra.exists():
        rr=Dataset.from_parquet(str(extra)).to_list()
        rows+=rr; print("added real rollouts:",len(rr), flush=True)
    random.shuffle(rows)
    out="data/blog_sft_teacher"; Path(out).mkdir(parents=True,exist_ok=True)
    Dataset.from_list(rows).to_parquet(f"{out}/train.parquet")
    print("WROTE",out,"total",len(rows), flush=True)

if __name__=="__main__":
    main()
