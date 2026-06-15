#!/usr/bin/env python3
"""Collate val/train trajectories across the sweep runs from their run.log files."""
import re
import sys
from pathlib import Path

RUNS = {
    "r1_lr5e6": "lr=5e-6, 1 epoch (7 steps)",
    "r2_lr1e5": "lr=1e-5, 1 epoch (7 steps)",
    "r3_lr5e6_3ep": "lr=5e-6, 3 epochs (21 steps)",
}
BASE = Path("outputs/sweep")

eval_re = re.compile(r"Evaluated (\S+) \(Step (\d+)\).*?Reward ([0-9.]+)")
train_re = re.compile(r"SUCCESS Step (\d+) \|.*?Reward ([0-9.]+) \| Trainable")


def parse(log: Path):
    evals = {}  # env -> {step: reward}
    train = {}  # step -> reward
    for line in log.read_text(errors="ignore").splitlines():
        if (m := eval_re.search(line)):
            env, step, r = m.group(1), int(m.group(2)), float(m.group(3))
            evals.setdefault(env, {})[step] = r
        elif (m := train_re.search(line)):
            train[int(m.group(1))] = float(m.group(2))
    return evals, train


def fmt(d):
    return "  ".join(f"{s}:{d[s]:.3f}" for s in sorted(d))


for run, desc in RUNS.items():
    log = BASE / run / "run.log"
    print(f"\n=== {run}  ({desc}) ===")
    if not log.exists():
        print("  (not started)")
        continue
    evals, train = parse(log)
    for env in sorted(evals):
        d = evals[env]
        v0 = d.get(0, float("nan"))
        trained = [d[s] for s in d if s > 0]
        best = max(trained) if trained else float("nan")
        mean = sum(trained) / len(trained) if trained else float("nan")
        tag = "  <-- stable pass@1" if "greedy" in env else "  (temp-1 pass@4)"
        print(f"  {env}{tag}")
        print(f"    val@0={v0:.3f}  trained_mean={mean:.3f}  best={best:.3f}")
        print(f"    curve: {fmt(d)}")
    if train:
        print(f"  train_reward: {fmt(train)}")
