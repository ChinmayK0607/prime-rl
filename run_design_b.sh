#!/bin/bash
source .venv/bin/activate
LOG=probe_results/design_b_progress.log
: > $LOG
for MODE in baseline cheatsheet fewshot; do
  echo "############ $(date +%H:%M:%S) MODE=$MODE ############" | tee -a $LOG
  python scripts/style_probe_eval.py --mode $MODE --split val --tp 4 \
    --out probe_results/B_${MODE}_val.jsonl 2>&1 | tee -a $LOG
done
echo "############ ALL DONE $(date +%H:%M:%S) ############" | tee -a $LOG
