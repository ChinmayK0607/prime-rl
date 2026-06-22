"""Push selected blog-provider-id checkpoints to the HF Hub (public) under CK0607.
Writes a concise model card into each dir, then upload_folder (excluding runtime markers).
Usage: python scripts/push_models_hub.py <key1> <key2> ...   (keys from MODELS below; default all)
"""
import os, sys
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
from huggingface_hub import HfApi

USER = "CK0607"
BASE = "Qwen/Qwen3.5-9B"
COMMON = f"""
- **Base model:** [{BASE}](https://huggingface.co/{BASE}) (thinking OFF)
- **Task:** 3-way AI-provider classification — given a blog/essay, identify whether it was written by **CLAUDE**, **CHATGPT**, or **GEMINI**. Output format: `<reason_why>...</reason_why><answer>LABEL\\nConfidence: ...</answer>`.
- **Eval:** `val` (in-distribution topics, n=414) and `val_ood` (held-out topics, n=471), zero eval leakage.
- **Provenance:** prime-rl; code at https://github.com/ChinmayK0607/prime-rl/tree/blog-author-id-experiments
"""

MODELS = {
    "sft": dict(
        path="outputs/sft_warmup/weights/step_180",
        repo=f"{USER}/qwen3.5-9b-blogprovider-sft-goldcond",
        title="Qwen3.5-9B — Blog-Provider-ID — Gold-Conditioned SFT",
        body="**Method:** SFT distilling gold-label-conditioned teacher rationales (base+cheatsheet, told the answer) onto the PLAIN prompt. Balanced 894/class + rollouts (2892 samples).\n\n**Result:** val **1.000** / val_ood **1.000**. Forgetting cost ~zero (general-text perplexity +3.2%, capability probes intact).",
    ),
    "answeronly": dict(
        path="outputs/sft_star_e1_ans/weights/step_80",
        repo=f"{USER}/qwen3.5-9b-blogprovider-selfgated-answeronly",
        title="Qwen3.5-9B — Blog-Provider-ID — Self-Gated Answer-Only SFT",
        body="**Method (E1 control):** answer-only SFT on 1071 verifier-gated rows (the model's own rollouts; gold only verifier-gates which are kept, never shown during generation; rationale STRIPPED, only the label supervised).\n\n**Result:** val **1.000** / val_ood **1.000**. Headline finding: answer-only BEATS the with-reasoning STaR run (0.953) on identical data and MATCHES the gold-conditioned SFT (1.000) with fewer rows and no rationale — on this task the discriminative signal is a dense supervised label-mapping, and supervising self-generated rationales is net-harmful.",
    ),
    "star": dict(
        path="outputs/sft_star_e1/weights/step_80",
        repo=f"{USER}/qwen3.5-9b-blogprovider-star-selfdistill",
        title="Qwen3.5-9B — Blog-Provider-ID — STaR Self-Distillation SFT",
        body="**Method (E1):** STaR on-policy self-distillation. Model generates its OWN `<reason_why>` rationales; gold label only verifier-gates kept trajectories (plain k=3 majority-gate + cheatsheet-hinted k=2 on still-wrong). 1071 class-balanced rows.\n\n**Result:** best step_80 val **0.932** / val_ood **0.953**. Residual gap concentrated in CHATGPT (recall 0.80–0.89). See the answer-only control for the contrast.",
    ),
    "rlcheat": dict(
        path="outputs/rl_3way_trio_cheat/weights/step_40",
        repo=f"{USER}/qwen3.5-9b-blogprovider-rl-cheatsheet",
        title="Qwen3.5-9B — Blog-Provider-ID — RL (GRPO, cheatsheet)",
        body="**Method:** GRPO RL with a train-derived style cheatsheet in context, trio curriculum. Representative best RL checkpoint (step_40).\n\n**Result:** cheatsheet-free accuracy plateaued ~0.40; RL polished but did not exceed the SFT ceiling. Part of the RL-vs-SFT analysis (reasoning-channel RL is the mismatched tool for this densely-separable task).",
    ),
    "rlentdecay": dict(
        path="outputs/rl_3way_trio_entdecay/weights/step_40",
        repo=f"{USER}/qwen3.5-9b-blogprovider-rl-entropydecay",
        title="Qwen3.5-9B — Blog-Provider-ID — RL (entropy-decay ablation)",
        body="**Method:** GRPO RL with an entropy-decay schedule (ablation). Final checkpoint (step_40).\n\n**Result:** negative ablation — no improvement over the cheatsheet RL baseline. Archived for completeness.",
    ),
    "pureacc": dict(
        path="outputs/rl_3way_pure_acc/run_default/broadcasts/step_16",
        repo=f"{USER}/qwen3.5-9b-blogprovider-rl-pureacc-peak",
        title="Qwen3.5-9B — Blog-Provider-ID — RL (pure-accuracy, peak)",
        body="**Method:** answer-only / no-reasoning GRPO (reward = exact label match). This is the PEAK checkpoint (step_16) before the run collapsed.\n\n**Result:** peaked val 0.650 / val_ood 0.662, then collapsed (CLAUDE absorbed into CHATGPT → 2-class ceiling) and hit the zero-trainable-batch guardrail. Directionally showed answer-only > reasoning-RL, but sparse single-token RL collapsed the fine boundary. (NB: raw weight-broadcast snapshot; config/tokenizer copied from the matching base arch.)",
    ),
}

IGNORE = ["STABLE", "NCCL_READY", "*.pid", "README.md.lock"]


def card(m):
    return f"""---
license: apache-2.0
base_model: {BASE}
pipeline_tag: text-generation
tags: [qwen3, classification, ai-text-detection, self-distillation, prime-rl]
---

# {m['title']}

{m['body']}
{COMMON}
"""


def main():
    keys = sys.argv[1:] or list(MODELS)
    api = HfApi()
    print("whoami:", api.whoami().get("name"), flush=True)
    for k in keys:
        m = MODELS[k]
        p = m["path"]
        assert os.path.isdir(p), f"missing {p}"
        with open(os.path.join(p, "README.md"), "w") as f:
            f.write(card(m))
        print(f"\n=== [{k}] create_repo {m['repo']} (public) ===", flush=True)
        api.create_repo(m["repo"], repo_type="model", private=False, exist_ok=True)
        print(f"=== uploading {p} -> {m['repo']} ===", flush=True)
        api.upload_folder(folder_path=p, repo_id=m["repo"], repo_type="model",
                          ignore_patterns=IGNORE,
                          commit_message=f"Upload {m['title']}")
        print(f"DONE https://huggingface.co/{m['repo']}", flush=True)


if __name__ == "__main__":
    main()
