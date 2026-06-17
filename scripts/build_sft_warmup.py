"""Harvest good single-task rollouts -> SFT warmup dataset (messages format).
Filters: correct, parsed_ok, reason_ok, not truncated, substantive reason.
Leakage-safe: drops any example whose blog body matches val/val_ood.
Provider-balanced.
"""
import json, glob, re, hashlib, collections, random
from datasets import load_from_disk, Dataset

random.seed(0)
PROVIDER={"gpt-5.5":"CHATGPT","claude-opus-4.8":"CLAUDE",
          "gemini-3.1-pro-preview":"GEMINI","gemini-3.5-flash":"GEMINI"}

def body_key(text):
    a=re.sub(r'[^a-z0-9]','',text.lower())
    return a[:400]

# 1) leakage keys from val + val_ood
leak=set()
for sp in ["val","val_ood"]:
    d=load_from_disk(f"data/blog_author_id_3way_v2/{sp}")
    for q in d["question"]:
        leak.add(body_key(q))
print("leak keys:", len(leak))

# 2) harvest
files=sorted(glob.glob("outputs/*/run_default/rollouts/step_*/train_rollouts.jsonl"))
by_prov=collections.defaultdict(list)
seen=set()
n_leak=0
for f in files:
    for l in open(f):
        try: r=json.loads(l)
        except: continue
        info=r.get("info") or {}
        if info.get("task_type")=="trio": continue
        if not (r.get("correct_answer")==1.0 and r.get("reason_ok")==1.0
                and r.get("parsed_ok")==1.0 and not r.get("is_truncated")): continue
        sm=info.get("source_model"); prov=PROVIDER.get(sm)
        if prov is None: continue
        user=r["prompt"][-1]["content"]
        asst=r["completion"][0]["content"]
        # reason substance: >=20 words inside reason_why
        m=re.search(r'<reason_why>(.*?)</reason_why>', asst, re.S)
        if not m or len(m.group(1).split())<20: continue
        k=body_key(user)
        if k in leak: n_leak+=1; continue
        if k in seen: continue
        seen.add(k)
        by_prov[prov].append({
            "messages":[
                r["prompt"][0],            # system
                {"role":"user","content":user},
                {"role":"assistant","content":asst},
            ]})
print("dropped leakage:", n_leak)
for p,v in by_prov.items(): print("unique", p, len(v))

# 3) balance: cap to min across providers
cap=min(len(v) for v in by_prov.values())
print("cap per provider:", cap)
rows=[]
for p,v in by_prov.items():
    random.shuffle(v); rows+=v[:cap]
random.shuffle(rows)
print("total SFT rows:", len(rows))
ds=Dataset.from_list(rows)
out="data/blog_sft_warmup"
ds.to_parquet(f"{out}/train.parquet")
print("wrote", out, "rows", len(ds))
