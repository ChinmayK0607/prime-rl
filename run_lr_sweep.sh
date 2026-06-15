#!/usr/bin/env bash
# LR sweep on the expanded blog author-id RL task (180 train / 90 val).
#
# Motivation: lr=5e-6 collapsed (val pinned to 0.500, trainable groups -> ~1%).
# This brackets the stable->collapse boundary with 3 points below it. Each run is
# ~1 epoch (15 steps). wandb is offline (flaky network); inference compile cache
# is reused between runs.
set -u
cd /home/ubuntu/blogger/prime-rl

wait_for_gpus() {
  for _ in $(seq 1 120); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
    [ "${used:-0}" -lt 2000 ] && return 0
    sleep 5
  done
}

run() {
  name="$1"; out="$2"; lr="$3"
  echo "=== launching $name (lr=$lr) -> $out ==="
  wait_for_gpus
  mkdir -p "$out"
  WANDB_MODE=offline uv run rl @ examples/blog_author_id/rl.toml \
    --output-dir "$out" --wandb.name "$name" \
    --trainer.optim.lr "$lr" --max-steps 15 > "$out/run.log" 2>&1
  echo "=== $name exited ($?) ==="
  sleep 15
}

run "qwen3.5-9b-lr1e6" outputs/lr_sweep/lr1e6 1e-6
run "qwen3.5-9b-lr2e6" outputs/lr_sweep/lr2e6 2e-6
run "qwen3.5-9b-lr3e6" outputs/lr_sweep/lr3e6 3e-6
echo "=== LR SWEEP COMPLETE ==="
