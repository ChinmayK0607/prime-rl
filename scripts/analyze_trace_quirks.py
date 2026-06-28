"""Characterise, per model, HOW each training method reasons about
CLAUDE / CHATGPT / GEMINI -- as whole prose, not just isolated tokens.

For every model that emits a non-trivial <reason_why> we report, per gold
provider (correct predictions only):
  * representative full rationales (verbatim, for the write-up),
  * which concrete surface-features the prose actually points at
    (em-dash, headers, tables, lists, sentence length, transitions, emoji ...),
  * a coarse "concrete vs vibes" score = fraction of rationales that name at
    least one measurable surface feature,
  * prose stats (length, hedging, confidence markers).

Outputs blog-eval/analysis/trace_quirks.json.
"""
import json, glob, os, re, collections, statistics, random

TRACE_DIR = "blog-eval/traces"
OUT = "blog-eval/analysis/trace_quirks.json"
PROVIDERS = ["CLAUDE", "CHATGPT", "GEMINI"]
random.seed(0)

# Concrete, measurable surface-features a rationale can cite.
SURFACE = {
    "em_dash":      r"em[- ]?dash|—|\bdash(es)?\b",
    "headers":      r"header|heading|##|markdown head|section title",
    "tables":       r"\btable(s)?\b|tabular|column",
    "ascii_diagram":r"ascii|diagram|\bart\b|box[- ]draw|figure",
    "lists_bullets":r"bullet|\blist(s|ed)?\b|enumerat|numbered",
    "sentence_len": r"sentence length|long sentence|short sentence|terse|clipped|run[- ]on|sprawling",
    "transitions":  r"however|furthermore|moreover|therefore|transition word|connective",
    "emoji":        r"emoji|emoticon",
    "hedging":      r"hedg|qualifi|caveat|nuance|measured|cautious",
    "bold_caps":    r"\bbold\b|\*\*|all[- ]caps|capitaliz",
    "questions":    r"rhetorical question|asks the reader|question mark",
    "density":      r"dense|encyclopedic|jargon|technical abstraction|academic",
}
SURFACE_RX = {k: re.compile(v) for k, v in SURFACE.items()}

CONFIDENT = re.compile(r"clearly|unmistakab|definit|certainly|hallmark|signature|telltale|obvious")
HEDGED = re.compile(r"perhaps|possibly|might|could be|somewhat|seems|appears|may ")


def reason_of(comp):
    m = re.search(r"<reason_why>(.*?)</reason_why>", comp, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"<reason_why>(.*)", comp, re.S)
    return m.group(1).strip() if m else ""


def surface_hits(txt):
    low = txt.lower()
    return [k for k, rx in SURFACE_RX.items() if rx.search(low)]


def pick_examples(texts, k=3, lo=25, hi=110):
    scored = []
    for t in texts:
        wc = len(t.split())
        concrete = len(surface_hits(t))
        band = lo <= wc <= hi
        scored.append((band, concrete, -abs(wc - 60), t))
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    out, seen = [], set()
    for _, _, _, t in scored:
        key = t[:50]
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= k:
            break
    return out


def analyze_model(path):
    recs = [json.loads(l) for l in open(path)]
    by_prov = collections.defaultdict(list)
    rlens, concrete_flags, conf_flags, hedge_flags = [], [], [], []
    surf_counter = collections.Counter()
    n_reason = 0
    for r in recs:
        rz = reason_of(r["completion"])
        if len(rz.split()) >= 5:
            n_reason += 1
            rlens.append(len(rz.split()))
            hits = surface_hits(rz)
            surf_counter.update(hits)
            concrete_flags.append(1 if hits else 0)
            conf_flags.append(1 if CONFIDENT.search(rz.lower()) else 0)
            hedge_flags.append(1 if HEDGED.search(rz.lower()) else 0)
        if r["correct"] and len(rz.split()) >= 5:
            by_prov[r["gold"]].append(rz)
    if n_reason < 10:
        return None
    per_provider = {}
    for p in PROVIDERS:
        ts = by_prov.get(p, [])
        if not ts:
            continue
        sc = collections.Counter()
        for t in ts:
            sc.update(surface_hits(t))
        per_provider[p] = {
            "n_correct": len(ts),
            "surface_features_cited": sc.most_common(6),
            "examples": pick_examples(ts, k=3),
        }
    return {
        "n": len(recs),
        "n_with_reason": n_reason,
        "median_reason_words": statistics.median(rlens) if rlens else 0,
        "concrete_fraction": round(sum(concrete_flags) / len(concrete_flags), 3) if concrete_flags else 0,
        "confident_fraction": round(sum(conf_flags) / len(conf_flags), 3) if conf_flags else 0,
        "hedged_fraction": round(sum(hedge_flags) / len(hedge_flags), 3) if hedge_flags else 0,
        "surface_features_overall": surf_counter.most_common(10),
        "per_provider": per_provider,
    }


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    result = {}
    for d in sorted(glob.glob(os.path.join(TRACE_DIR, "*/"))):
        name = os.path.basename(d.rstrip("/"))
        f = os.path.join(d, "val.jsonl")
        if not os.path.exists(f):
            continue
        a = analyze_model(f)
        if a:
            result[name] = a
            print(f"[ok] {name}: reason={a['n_with_reason']}/{a['n']} "
                  f"med={a['median_reason_words']:.0f}w concrete={a['concrete_fraction']:.2f} "
                  f"conf={a['confident_fraction']:.2f} hedge={a['hedged_fraction']:.2f}")
        else:
            print(f"[skip] {name}")
    json.dump(result, open(OUT, "w"), indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
