"""Build the 3-way + hard-pair auxiliary curriculum (per-row prompt column).

Why this exists
---------------
Under 3-way correctness-only GRPO the policy stopped improving net accuracy: it
raised CLAUDE/GEMINI recall while COLLAPSING CHATGPT into CLAUDE (a zero-sum recall
reshuffle). CHATGPT-vs-CLAUDE is the genuinely hard pair (both conversational prose).
A shaped/cost-matrix reward to punish that confusion backfires under group-normalized
GRPO: it makes all-wrong groups trainable and reinforces a "safe escape" to GEMINI.

The clean fix is an explicit CLAUDE-vs-CHATGPT 2-way auxiliary task interleaved into
the curriculum, scored with the SAME plain binary exact-match reward. Restricting the
label space forces the policy to find what distinguishes the hard pair (no GEMINI
escape; all-wrong groups stay zero-advantage-filtered as before).

Each row carries its own ``prompt`` column (a [system, user] message list) so a single
environment + single binary reward can serve both task types. info["task_type"] is "3way"
or "hardpair"; info["stripped"] flags the formatting-strip augmentation.

Construction (each consecutive block of BATCH_PROMPTS on-disk rows == one training
step under PRIME_RL_PRESERVE_DATA_ORDER=1):
  - Four streams: CLAUDE-3way, CHATGPT-3way, GEMINI-3way, hardpair-2way.
  - hardpair rows are drawn from CLAUDE/CHATGPT blogs, preferring the confusable
    middle difficulty (base 3-way n_correct in 1..3), gold balanced ~50/50.
  - Each stream is stratified across W windows (proportional n_correct mix,
    short-before-long drift). Window = round-robin C3/P3/G3/HP2 => each ~12-row
    window is provider-balanced 3/3/3 for the 3-way task + 3 hard-pair binary groups.
  - A seeded STRIP_FRAC of ALL rows is formatting-stripped (forces deeper style
    signal over surface cues, for both task types).

Val is 3-way only (eval parity with the offline baseline), also emitted with a
per-row 3-way prompt column so the env path is identical.

Output: data/blog_author_id_hardpair/{train,val}
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset, load_from_disk

# Single source of truth for the prompt text (must match the eval baseline).
from blog_author_id import SYSTEM_PROMPT_3WAY, SYSTEM_PROMPT_HARDPAIR

SRC = Path(__file__).resolve().parent / "blog_author_id_3way"
OUT = Path(__file__).resolve().parent / "blog_author_id_hardpair"
PREDS = Path("/home/ubuntu/blogger/blog-eval/logs/multimodel3way_predictions.jsonl")

LENGTH_RANK = {"short": 0, "long": 1}
BATCH_PROMPTS = 12
PER_WINDOW = {"CLAUDE": 3, "CHATGPT": 3, "GEMINI": 3, "HARDPAIR": 3}  # sums to 12
STREAM_ORDER = ["CLAUDE", "CHATGPT", "GEMINI", "HARDPAIR"]
STRIP_FRAC = 0.40
SEED = 0


def strip_formatting(t: str) -> str:
    """Remove markdown/formatting cues while preserving wording (matches the
    content-signal probe that scored ~100% val acc on stripped text)."""
    t = re.sub(r"```.*?```", " ", t, flags=re.S)          # code fences
    t = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", t)            # headers
    t = re.sub(r"(?m)^\s*[-*+]\s+", "", t)                 # bullets
    t = re.sub(r"(?m)^\s*\d+\.\s+", "", t)                 # numbered lists
    t = t.replace("**", "").replace("__", "")              # bold
    t = re.sub(r"(?<!\*)\*(?!\*)", "", t)                  # italics
    t = t.replace("\u2014", ", ").replace("\u2013", "-")   # em/en dash -> neutral
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)                         # collapse blank lines
    return t.strip()


def load_passrates() -> dict[tuple[str, str, str, str], int]:
    rates: dict[tuple[str, str, str, str], int] = {}
    for line in PREDS.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("split") != "train" or r.get("markdown") != "keep":
            continue
        rates[(r["category"], r["topic"], r["source_model"], r["length"])] = r["n_correct"]
    return rates


def nc_of(info: dict, rates: dict) -> int:
    return rates.get((info["category"], info["topic"], info["source_model"], info["length"]), 2)


def stratify(descs: list[dict], n_windows: int) -> list[list[dict]]:
    """Deal descriptors across windows: each window gets a proportional n_correct
    mix; within an n_correct bucket, short blogs are dealt to earlier windows
    (gentle short->long drift)."""
    by_nc: dict[int, list[tuple[int, dict]]] = defaultdict(list)  # nc -> [(lrank, desc)]
    for d in descs:
        by_nc[d["nc"]].append((d["lrank"], d))
    windows: list[list[dict]] = [[] for _ in range(n_windows)]
    for nc in sorted(by_nc):
        bucket = sorted(by_nc[nc], key=lambda x: x[0])  # short first
        size = len(bucket)
        for j, (_lrank, d) in enumerate(bucket):
            windows[(j * n_windows) // size].append(d)
    return windows


def build_descriptors(train: Dataset, rates: dict) -> tuple[dict[str, list[dict]], int]:
    """Return per-stream descriptor lists and the window count."""
    by_provider: dict[str, list[dict]] = defaultdict(list)
    for i, row in enumerate(train):
        info = row["info"]
        by_provider[row["answer"]].append(
            {"src": i, "gold": row["answer"], "nc": nc_of(info, rates),
             "lrank": LENGTH_RANK.get(info["length"], 0), "task": "3way"}
        )

    # Window count: size each 3-way provider stream to ~PER_WINDOW per window.
    n_windows = max(
        (len(by_provider[p]) + PER_WINDOW[p] - 1) // PER_WINDOW[p]
        for p in ("CLAUDE", "CHATGPT", "GEMINI")
    )

    # Hard-pair stream: prefer confusable middle-difficulty CLAUDE/CHATGPT blogs,
    # gold-balanced. Draw without reuse first; allow reuse only if short.
    need = PER_WINDOW["HARDPAIR"] * n_windows
    pool: dict[str, list[dict]] = {"CLAUDE": [], "CHATGPT": []}
    for p in ("CLAUDE", "CHATGPT"):
        ranked = sorted(
            by_provider[p],
            key=lambda d: (0 if 1 <= d["nc"] <= 3 else 1, d["lrank"], d["nc"]),
        )
        pool[p] = ranked
    hp: list[dict] = []
    half = need // 2
    for p, k in (("CLAUDE", half), ("CHATGPT", need - half)):
        src = pool[p]
        picked = [dict(src[j % len(src)], task="hardpair") for j in range(k)]
        hp.extend(picked)

    return {
        "CLAUDE": by_provider["CLAUDE"],
        "CHATGPT": by_provider["CHATGPT"],
        "GEMINI": by_provider["GEMINI"],
        "HARDPAIR": hp,
    }, n_windows


def assemble_order(streams: dict[str, list[dict]], n_windows: int) -> list[dict]:
    win = {s: stratify(streams[s], n_windows) for s in STREAM_ORDER}
    order: list[dict] = []
    for w in range(n_windows):
        cursors = {s: 0 for s in STREAM_ORDER}
        lists = {s: win[s][w] for s in STREAM_ORDER}
        remaining = sum(len(v) for v in lists.values())
        while remaining:
            for s in STREAM_ORDER:
                if cursors[s] < len(lists[s]):
                    order.append(lists[s][cursors[s]])
                    cursors[s] += 1
                    remaining -= 1
    return order


def make_prompt(text: str, task: str) -> list[dict]:
    sysmsg = SYSTEM_PROMPT_HARDPAIR if task == "hardpair" else SYSTEM_PROMPT_3WAY
    return [{"role": "system", "content": sysmsg},
            {"role": "user", "content": text}]


def materialize(order: list[dict], train: Dataset, strip_set: set[int]) -> Dataset:
    rows = {"prompt": [], "answer": [], "info": []}
    for pos, d in enumerate(order):
        base = train[d["src"]]
        text = base["question"]
        stripped = pos in strip_set
        if stripped:
            text = strip_formatting(text)
        info = dict(base["info"])
        info["task_type"] = d["task"]
        info["stripped"] = stripped
        rows["prompt"].append(make_prompt(text, d["task"]))
        rows["answer"].append(d["gold"])
        rows["info"].append(info)
    return Dataset.from_dict(rows)


def build_val(val: Dataset) -> Dataset:
    rows = {"prompt": [], "answer": [], "info": []}
    for row in val:
        info = dict(row["info"])
        info["task_type"] = "3way"
        info["stripped"] = False
        rows["prompt"].append(make_prompt(row["question"], "3way"))
        rows["answer"].append(row["answer"])
        rows["info"].append(info)
    return Dataset.from_dict(rows)


def main() -> None:
    rates = load_passrates()
    print(f"[hardpair] loaded {len(rates)} measured train pass-rates")

    train = load_from_disk(str(SRC / "train"))
    val = load_from_disk(str(SRC / "val"))

    streams, n_windows = build_descriptors(train, rates)
    order = assemble_order(streams, n_windows)
    print(f"[hardpair] {n_windows} windows; {len(order)} train rows "
          f"({sum(d['task']=='3way' for d in order)} 3way / "
          f"{sum(d['task']=='hardpair' for d in order)} hardpair)")

    rng = random.Random(SEED)
    strip_set = set(rng.sample(range(len(order)), int(round(len(order) * STRIP_FRAC))))

    train_out = materialize(order, train, strip_set)
    val_out = build_val(val)

    # Per-window composition report + degeneracy guard.
    print("\n[hardpair] per-window composition (first 12):")
    print("  win | C3/P3/G3/HP | gold C/P/G | nc{0..4} | #middle | #strip")
    bad = []
    for w in range(n_windows):
        block = [order[i] for i in range(w * 12, min((w + 1) * 12, len(order)))]
        if not block:
            continue
        task = Counter(d["task"] for d in block)
        c3 = sum(1 for d in block if d["task"] == "3way" and d["gold"] == "CLAUDE")
        p3 = sum(1 for d in block if d["task"] == "3way" and d["gold"] == "CHATGPT")
        g3 = sum(1 for d in block if d["task"] == "3way" and d["gold"] == "GEMINI")
        gold = Counter(d["gold"] for d in block)
        ncs = Counter(d["nc"] for d in block)
        middle = sum(ncs.get(k, 0) for k in (1, 2, 3))
        nstrip = sum(1 for i in range(w * 12, min((w + 1) * 12, len(order))) if i in strip_set)
        if w < 12:
            ncstr = "/".join(str(ncs.get(k, 0)) for k in range(5))
            print(f"  {w:3d} | {c3}/{p3}/{g3}/{task.get('hardpair',0)} | "
                  f"{gold.get('CLAUDE',0)}/{gold.get('CHATGPT',0)}/{gold.get('GEMINI',0)} | "
                  f"{ncstr} | {middle:2d} | {nstrip}")
        if len(block) == 12:
            # 3-way groups must stay balanced; hard-pair present; signal-rich.
            if max(c3, p3, g3) >= 6 or min(c3, p3, g3) == 0:
                bad.append((w, "3way-skew", (c3, p3, g3)))
            if task.get("hardpair", 0) == 0:
                bad.append((w, "no-hardpair", None))
            if middle < 5:
                bad.append((w, "low-signal", middle))

    print("\n[hardpair] global:")
    print("  task:", Counter(d["task"] for d in order))
    print("  gold:", Counter(d["gold"] for d in order))
    print("  nc:", dict(sorted(Counter(d["nc"] for d in order).items())))
    print("  stripped:", sum(1 for r in train_out if r["info"]["stripped"]),
          f"({sum(1 for r in train_out if r['info']['stripped'])/len(train_out):.0%})")
    short_pos = [i for i, d in enumerate(order) if d["lrank"] == 0]
    if short_pos:
        print(f"  short rows mean position: {sum(short_pos)/len(short_pos):.1f} of {len(order)} "
              f"(<{len(order)/2:.0f} => short->long drift)")

    if bad:
        print(f"\n[hardpair] WARNING: {len(bad)} degenerate windows: {bad[:10]}")
    else:
        print("\n[hardpair] OK: every full window 3-way-balanced (<6,>0), has hard-pair, >=5 middle.")

    OUT.mkdir(parents=True, exist_ok=True)
    train_out.save_to_disk(str(OUT / "train"))
    val_out.save_to_disk(str(OUT / "val"))
    print(f"\n[hardpair] wrote {len(train_out)} train / {len(val_out)} val to {OUT}")


if __name__ == "__main__":
    main()
