#!/usr/bin/env bash
# Launch the 3-way curriculum GRPO run + a background wandb online-sync loop.
#
# wandb runs OFFLINE inside the training process (online mode crashes the run via
# a ConnectionResetError cascade on this flaky network). A separate background
# loop incrementally `wandb sync`s the offline run dir to wandb.ai online, so the
# run streams to the dashboard without risking the training process.
#
# Curriculum ordering is only honored because PRIME_RL_PRESERVE_DATA_ORDER=1 makes
# TrainSource skip its shuffles (data is pre-sorted easy->hard on disk).
set -u
cd /home/ubuntu/blogger/prime-rl

# Args: $1 = config TOML (relative to repo root), $2 = output dir.
CONFIG="${1:-examples/blog_author_id/rl_3way.toml}"
OUT="${2:-outputs/rl_3way_entropy}"
mkdir -p "$OUT"

echo "=== launching 3-way curriculum RL ($CONFIG) -> $OUT ==="
# HF_HUB_OFFLINE=1: model is already cached; skip the flaky HF Hub check that
# crashed a prior launch with httpx.RemoteProtocolError during pre-download.
PRIME_RL_PRESERVE_DATA_ORDER=1 HF_HUB_OFFLINE=1 WANDB_MODE=offline \
  nohup uv run rl @ "$CONFIG" --output-dir "$OUT" \
  > "$OUT/run.log" 2>&1 &
RL_PID=$!
echo "RL pid=$RL_PID"

# Background online-sync loop: every 60s push any offline wandb runs under $OUT
# to wandb.ai. Each attempt is bounded by `timeout` so a blocked/slow wandb.ai
# (api.wandb.ai is firewalled in some environments) can't hang the loop; it just
# retries and will succeed once connectivity is available. `wandb sync` is
# incremental and safe to re-run while the run is live.
(
  while kill -0 "$RL_PID" 2>/dev/null; do
    # Sync EVERY offline run under $OUT, including the orchestrator run which
    # lives in run_default/wandb/ (this holds reward/* and eval/blog-val/*),
    # not just the trainer run in $OUT/wandb/.
    while IFS= read -r d; do
      [ -d "$d" ] && timeout 150 uv run wandb sync "$d" >> "$OUT/wandb_sync.log" 2>&1
    done < <(find "$OUT" -type d \( -name 'offline-run-*' -o -name 'run-*' \) -path '*/wandb/*')
    sleep 60
  done
  # Final sync after the run exits.
  while IFS= read -r d; do
    [ -d "$d" ] && timeout 300 uv run wandb sync "$d" >> "$OUT/wandb_sync.log" 2>&1
  done < <(find "$OUT" -type d \( -name 'offline-run-*' -o -name 'run-*' \) -path '*/wandb/*')
) &
echo "sync loop pid=$!"

# Background best-val checkpoint janitor: keeps ONLY the best-val weights
# checkpoint (weights_only saves ~17 GB/step, no optimizer), prunes evaluated
# non-best checkpoints, and reclaims leaked weight-broadcast dirs. Without this
# the run overflows the disk (a prior run died with ENOSPC on a 102 GB optimizer
# checkpoint). It self-exits (after a final sweep) once the RL process is gone.
nohup bash scripts/ckpt_janitor.sh "$OUT" 30 "$RL_PID" >/dev/null 2>&1 &
echo "janitor pid=$!"

echo "$RL_PID" > "$OUT/rl.pid"
