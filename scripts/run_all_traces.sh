#!/usr/bin/env bash
# Sequentially run plain-prompt trace inference for every Hub model (one fresh process each).
set -u
cd /home/ubuntu/blogger/prime-rl
source .venv/bin/activate
export USE_HUB_KERNELS=NO TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0

MODELS=(
  "qwen3.5-9b-blogprovider-sft-goldcond"
  "qwen3.5-9b-blogprovider-selfgated-answeronly"
  "qwen3.5-9b-blogprovider-star-selfdistill"
  "qwen3.5-9b-blogprovider-rl-cheatsheet"
  "qwen3.5-9b-blogprovider-rl-entropydecay"
  "qwen3.5-9b-blogprovider-rl-pureacc-peak"
)

for m in "${MODELS[@]}"; do
  echo "================ $(date '+%H:%M:%S')  START $m ================"
  python scripts/infer_traces_allmodels.py --repo "CK0607/$m" --short "$m" --out_dir blog-eval/traces
  rc=$?
  echo "================ $(date '+%H:%M:%S')  END $m (rc=$rc) ================"
done
echo "ALL TRACE INFERENCE DONE"
