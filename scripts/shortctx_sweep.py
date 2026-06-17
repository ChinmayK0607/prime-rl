"""Short-context difficulty sweep for the SFT classifier.

Classify from only the first N chars of each blog (label preserved). Maps where
the model breaks -> the one axis with genuine reward variance left (harder eval /
RL scope). Caps max_tokens to avoid runaway generations on ambiguous tiny inputs;
also reports mean generation length as a confidence/uncertainty proxy.
"""
import json, sys
from datasets import load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, "scripts")
from style_probe_eval import SYSTEM_PROMPT_3WAY, LABELS, parse_pred

M = "outputs/sft_warmup/weights/step_180"


def main():
  ds = load_from_disk("data/blog_author_id_3way_v2/val_ood")
  tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
  llm = LLM(model=M, tensor_parallel_size=8, max_model_len=16384,
            gpu_memory_utilization=0.9, trust_remote_code=True)
  sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512, seed=0)

  out = {}
  for c in [120, 200, 300, 500, 800, 1500]:
    prompts = []
    for r in ds:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT_3WAY},
                {"role": "user", "content": r["question"][:c]}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                       add_generation_prompt=True, enable_thinking=False))
    outs = llm.generate(prompts, sp)
    n = len(ds); correct = 0; none = 0; trunc = 0; glen = 0
    per = {l: [0, 0] for l in LABELS}
    for r, o in zip(ds, outs):
        t = o.outputs[0].text
        glen += len(o.outputs[0].token_ids)
        if o.outputs[0].finish_reason == "length":
            trunc += 1
        p = parse_pred(t); g = r["answer"]
        per[g][1] += 1
        ok = (p == g); correct += ok; per[g][0] += ok
        if p == "NONE": none += 1
    rec = {"acc": round(correct/n, 4),
           "macro": round(sum(per[l][0]/max(per[l][1],1) for l in LABELS)/3, 4),
           "none": none, "trunc_at_512": trunc, "mean_gen_tok": round(glen/n, 1),
           "per_class": {l: round(per[l][0]/max(per[l][1],1), 3) for l in LABELS}}
    out[c] = rec
    print(f"first{c:5d}  acc={rec['acc']:.3f} macro={rec['macro']:.3f} "
          f"none={rec['none']:3d} trunc512={rec['trunc_at_512']:3d} "
          f"mean_gen={rec['mean_gen_tok']:.0f}tok per_class={rec['per_class']}",
          flush=True)

  open("probe_results/adv_shortctx_step180.json", "w").write(json.dumps(out, indent=2))
  print("JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
