#!/usr/bin/env bash
# Recurring online-sync for offline wandb runs that live OUTSIDE $OUT/wandb/,
# specifically the orchestrator run under $OUT/run_default/wandb/ which holds the
# reward/* and eval/blog-val/* curves. The main sync loop in run_3way_curriculum.sh
# only covers $OUT/wandb/ (the trainer run); this fills the gap for live runs.
#
# Usage: wandb_orc_sync.sh <output_dir> [poll_secs] [parent_pid]
set -u
OUT="${1:?output dir required}"
POLL="${2:-60}"
PARENT="${3:-}"

parent_alive() {
  [ -z "$PARENT" ] && return 0
  kill -0 "$PARENT" 2>/dev/null
}

sync_all() {
  local to="$1"
  while IFS= read -r d; do
    [ -d "$d" ] || continue
    timeout "$to" uv run wandb sync "$d" >> "$OUT/wandb_orc_sync.log" 2>&1
  done < <(find "$OUT" -type d \( -name 'offline-run-*' -o -name 'run-*' \) -path '*/run_default/wandb/*')
}

while parent_alive; do
  sync_all 150
  sleep "$POLL"
done
# Final flush after parent exits.
sync_all 300
