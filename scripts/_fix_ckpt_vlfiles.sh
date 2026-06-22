#!/bin/bash
# Copy VL preprocessor files (needed by vLLM for Qwen3.5-9B VL arch) into each SFT ckpt.
SNAP=/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
for d in "$@"; do
  for f in preprocessor_config.json video_preprocessor_config.json vocab.json merges.txt; do
    [ -f "$d/$f" ] || cp "$SNAP/$f" "$d/$f"
  done
  echo "fixed $d"
done
