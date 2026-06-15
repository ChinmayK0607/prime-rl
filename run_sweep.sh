#!/usr/bin/env bash
# Sequential LR + duration sweep on the blog author-id RL task.
# Each run reuses the inference compile cache; wandb is offline (flaky network).
set -u
cd /home/ubuntu/blogger/prime-rl

wait_for_gpus() {
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
    [ "${used:-0}" -lt 2000 ] && return 0
    sleep 5
  done
}

run() {
  name="$1"; out="$2"; shift 2
  echo "=== launching $name -> $out ==="
  wait_for_gpus
  mkdir -p "$out"
  WANDB_MODE=offline uv run rl @ examples/blog_author_id/rl.toml \
    --output-dir "$out" --wandb.name "$name" "$@" > "$out/run.log" 2>&1
  echo "=== $name exited ($?) ==="
  sleep 15
}

run "qwen3.5-9b-lr5e6"      outputs/sweep/r1_lr5e6      --trainer.optim.lr 5e-6 --max-steps 7
run "qwen3.5-9b-lr1e5"      outputs/sweep/r2_lr1e5      --trainer.optim.lr 1e-5 --max-steps 7
run "qwen3.5-9b-lr5e6-3ep"  outputs/sweep/r3_lr5e6_3ep  --trainer.optim.lr 5e-6 --max-steps 21
echo "=== SWEEP COMPLETE ==="
