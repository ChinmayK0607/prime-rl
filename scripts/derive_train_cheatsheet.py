"""Derive per-provider style tells from the v2 TRAIN split ONLY (no val/val_ood).

Content-free style features (function words, punctuation, formatting markers) — the
same content-free axis that the separability analysis showed transfers cross-register.
Reports, per provider, the features with the highest in-class rate x lift over the
other two. Output is a train-only cheatsheet => no eval leakage for the RL run.
"""
import re, json
from collections import Counter
from datasets import load_from_disk

TRAIN = "data/blog_author_id_3way_v2/train"
LABELS = ["CLAUDE", "CHATGPT", "GEMINI"]

# Curated content-free style probes: function words / hedges / intensifiers /
# discourse markers / formatting+punctuation markers. Each maps name -> regex.
PROBES = {
    # hedging / enumerative (ChatGPT-ish a priori, but we measure)
    "may /may be": r"\bmay\b",
    "can also": r"\bcan also\b",
    "for example": r"\bfor example\b",
    "such as": r"\bsuch as\b",
    "not only..but": r"\bnot only\b",
    "depends on": r"\bdepends on\b",
    "numbered list": r"(?m)^\s*\d+[.)]\s",
    "bullet list": r"(?m)^\s*[-*+]\s",
    # sincerity / second person (Claude-ish a priori)
    "second-person you": r"\byou\b",
    "genuinely": r"\bgenuinely\b",
    "honestly": r"\bhonestly\b",
    "precisely": r"\bprecisely\b",
    "the honest truth": r"\bhonest\b",
    "worth": r"\bworth\b",
    "I (first person)": r"\bI\b",
    # grandiose / formal (Gemini-ish a priori)
    "profound/fundamental": r"\b(profound|fundamental|fundamentally)\b",
    "however": r"\bhowever\b",
    "furthermore": r"\bfurthermore\b",
    "we must": r"\bwe must\b",
    "paradigm": r"\bparadigm\b",
    "intensifier highly/massive": r"\b(highly|massive|massively)\b",
    # formatting / notation
    "LaTeX $": r"\$[^$\n]+\$|\\\(|\\\[|\$\$",
    "markdown header #": r"(?m)^\s*#{1,6}\s",
    "ASCII diagram |/_": r"[│┌┐└┘├┤┬┴┼─]|\+[-]{3,}\+",
    "em-dash": r"—",
    "bold **": r"\*\*[^*]+\*\*",
    "blockquote >": r"(?m)^\s*>\s",
}


def main():
    ds = load_from_disk(TRAIN)
    texts = {l: [] for l in LABELS}
    for r in ds:
        texts[r["answer"]].append(r["question"])
    counts = {l: len(texts[l]) for l in LABELS}
    print("train docs per class:", counts)

    # per-class fraction of docs containing each probe
    rate = {l: {} for l in LABELS}
    for l in LABELS:
        for name, pat in PROBES.items():
            rx = re.compile(pat)
            hits = sum(1 for t in texts[l] if rx.search(t))
            rate[l][name] = hits / max(len(texts[l]), 1)

    # for each provider, rank probes by rate * lift over mean of other two
    print("\n=== TRAIN-ONLY distinctive style tells (rate | lift x) ===")
    cheat = {}
    for l in LABELS:
        scored = []
        for name in PROBES:
            others = [rate[o][name] for o in LABELS if o != l]
            base = sum(others) / 2 + 1e-6
            lift = rate[l][name] / base
            scored.append((name, rate[l][name], lift))
        # keep tells that are both frequent-ish and distinctive
        scored = [s for s in scored if s[1] >= 0.15 and s[2] >= 1.3]
        scored.sort(key=lambda s: s[1] * s[2], reverse=True)
        cheat[l] = scored[:8]
        print(f"\n{l}:")
        for name, rt, lift in scored[:8]:
            print(f"  {name:28s} rate={rt:.2f}  lift={lift:.2f}x")

    json.dump({l: [(n, round(r, 3), round(x, 2)) for n, r, x in cheat[l]] for l in LABELS},
              open("probe_results/train_tells.json", "w"), indent=2)
    print("\nwrote probe_results/train_tells.json")


if __name__ == "__main__":
    main()
