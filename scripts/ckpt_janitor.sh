#!/usr/bin/env bash
# Best-validation checkpoint janitor for prime-rl runs.
#
# Storage policy (see examples/blog_author_id/rl_3way.toml [ckpt]):
#   - The trainer saves WEIGHTS ONLY (weights_only=true): no optimizer/DCP state,
#     just the ~17 GB HF safetensors export per checkpoint step.
#   - ckpt interval == eval interval, so every weights/step_N also gets an eval
#     reward logged as: "Evaluated blog-val (Step N) | Policy vK | ... | Reward R".
#
# This janitor keeps ONLY the best-val weights checkpoint and prunes the rest.
# Safety rules (a checkpoint is deleted ONLY when all hold):
#   1. it has a completed eval result in run.log (so it is fully written), AND
#   2. it is not the current best-val step.
# A not-yet-evaluated (pending) or in-progress checkpoint is NEVER deleted, so we
# can't race a mid-write export or drop a checkpoint before its eval lands.
#
# It also reclaims leaked weight-broadcast dirs: prime-rl pins broadcast dirs that
# fall on the ckpt interval (they never get cleaned), so we delete stale STABLE
# broadcast dirs older than the most recent few (never touching the top 3, which
# prime-rl itself is still managing).
#
# Usage: ckpt_janitor.sh <output_dir> [poll_seconds] [parent_pid]
set -uo pipefail

OUTPUT_DIR="${1:?usage: ckpt_janitor.sh <output_dir> [poll_seconds] [parent_pid]}"
POLL="${2:-30}"
PARENT_PID="${3:-}"
LOG="$OUTPUT_DIR/run.log"
WEIGHTS_DIR="$OUTPUT_DIR/weights"
BCAST_DIR="$OUTPUT_DIR/run_default/broadcasts"
JLOG="$OUTPUT_DIR/ckpt_janitor.log"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$JLOG"; }

log "=== janitor start: output_dir=$OUTPUT_DIR poll=${POLL}s ==="

# Parse eval results from run.log. Prints: "<best_step> <space-separated evaluated steps>".
# best_step = step with highest reward; ties broken toward the EARLIEST step
# (curriculum peak is expected early). Step 0 (base model, no export) is ignored.
parse_evals() {
  grep -oE "Evaluated blog-val \(Step [0-9]+\).*Reward [0-9]+\.[0-9]+" "$LOG" 2>/dev/null \
    | sed -E 's/.*\(Step ([0-9]+)\).*Reward ([0-9]+\.[0-9]+).*/\1 \2/' \
    | awk '
        $1 != 0 {
          step = $1; rew = $2 + 0;
          if (!(step in seen)) { seen[step] = 1; order[++m] = step; }
          r[step] = rew;  # last reward for a step wins (re-evals)
        }
        END {
          best = ""; bestr = -1; list = "";
          for (i = 1; i <= m; i++) {
            s = order[i]; list = list " " s;
            if (r[s] > bestr) { bestr = r[s]; best = s; }
          }
          print best list;
        }'
}

step_of() { echo "${1##*step_}"; }  # weights/step_12 -> 12

while true; do
  if [ -f "$LOG" ]; then
    parsed="$(parse_evals)"
    best_step="$(echo "$parsed" | awk '{print $1}')"
    evaluated=" $(echo "$parsed" | cut -d' ' -f2-) "  # padded for substring match

    # ---- weights retention: keep best + pending/in-progress, drop evaluated non-best ----
    if [ -d "$WEIGHTS_DIR" ] && [ -n "$best_step" ]; then
      # highest step dir present; used as a "fully written" guard below.
      maxw="$(ls -1d "$WEIGHTS_DIR"/step_* 2>/dev/null | sed -nE 's#.*/step_([0-9]+)$#\1#p' | sort -n | tail -1)"
      for d in "$WEIGHTS_DIR"/step_*; do
        [ -d "$d" ] || continue
        s="$(step_of "$d")"
        case "$evaluated" in
          *" $s "*) ;;             # evaluated -> candidate for deletion
          *) continue ;;            # not yet evaluated -> keep (pending/in-progress)
        esac
        [ "$s" = "$best_step" ] && continue   # keep the best
        # RACE GUARD 1: a step's eval result is logged BEFORE the trainer finishes
        # exporting weights/step_N (eval runs on broadcast weights, not the export),
        # so "evaluated" does NOT imply "fully written". Only prune step_N once a
        # strictly-higher step dir exists (proves N's export completed).
        [ -n "$maxw" ] && [ "$s" -lt "$maxw" ] || continue
        # RACE GUARD 2: never touch a dir modified in the last 3 minutes (mid-write).
        [ -n "$(find "$d" -maxdepth 0 -mmin -3 2>/dev/null)" ] && continue
        if rm -rf "$d"; then
          log "pruned weights/step_$s (evaluated, not best=$best_step, max=$maxw)"
        fi
      done
    fi
  fi

  # ---- broadcast leak cleanup: drop stale STABLE dirs below (max-2), keep top 3 ----
  if [ -d "$BCAST_DIR" ]; then
    maxb="$(ls -1 "$BCAST_DIR" 2>/dev/null | sed -nE 's/^step_([0-9]+)$/\1/p' | sort -n | tail -1)"
    if [ -n "$maxb" ] && [ "$maxb" -ge 3 ]; then
      cutoff=$((maxb - 2))
      for d in "$BCAST_DIR"/step_*; do
        [ -d "$d" ] || continue
        s="$(step_of "$d")"
        [ "$s" -lt "$cutoff" ] || continue
        [ -f "$d/STABLE" ] || continue
        # mtime guard: skip anything modified in the last 2 minutes
        [ -n "$(find "$d" -maxdepth 0 -mmin -2 2>/dev/null)" ] && continue
        if rm -rf "$d"; then
          log "pruned broadcasts/step_$s (stale, max=$maxb)"
        fi
      done
    fi
  fi

  sleep "$POLL"

  # Exit (after one last sweep on the next loop guard) once the training process
  # is gone, so the launcher's background janitor doesn't linger forever.
  if [ -n "$PARENT_PID" ] && ! kill -0 "$PARENT_PID" 2>/dev/null; then
    log "parent pid $PARENT_PID gone; doing final sweep then exiting"
    PARENT_PID=""        # clear so we don't loop again
    final_pass=1
    continue
  fi
  if [ "${final_pass:-0}" = 1 ]; then
    log "=== janitor exit ==="
    break
  fi
done
