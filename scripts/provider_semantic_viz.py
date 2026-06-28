"""Provider writing-style analysis + semantic-space visualiser.

What it does (all on CPU, no GPU needed):
  1. Loads the 3-way blog corpus (train/val/val_ood); text = `question`, label = `answer`.
  2. Builds TWO representations of every blog:
       - SEMANTIC (content): sentence-transformer embeddings (all-MiniLM-L6-v2).
       - STYLOMETRIC (style): ~40 hand-crafted structural/lexical features (sentence length,
         punctuation, markdown structure, hedging/function words, LaTeX, ASCII art, etc.).
  3. Projects both to 2D (UMAP) coloured by provider -> figures.
  4. Trains a logistic-regression classifier on each representation (fit on train, eval on
     val + val_ood) -> accuracy + confusion. Tests the project's core claim: providers
     OVERLAP on content (topic-balanced) but SEPARATE on style.
  5. Extracts the top discriminative stylometric features per provider (what each "looks like").

Outputs -> blog-eval/analysis/: provider_umap.png, provider_confusion.png,
           style_separation.png, analysis_summary.json + .md
"""
import os, re, json, math
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

DATA = "data/blog_author_id_3way_v2"
OUT = "blog-eval/analysis"
PROVIDERS = ["CLAUDE", "CHATGPT", "GEMINI"]
COLORS = {"CLAUDE": "#d97706", "CHATGPT": "#10a37f", "GEMINI": "#4285f4"}
os.makedirs(OUT, exist_ok=True)


def load_split(split):
    d = load_from_disk(f"{DATA}/{split}")
    texts = [r["question"] for r in d]
    labels = [r["answer"] for r in d]
    return texts, labels


# ----------------------------- stylometric features -----------------------------
HEDGE = ["however", "moreover", "furthermore", "nuanced", "crucial", "essentially",
         "indeed", "arguably", "delve", "tapestry", "underscore", "intricate",
         "nevertheless", "consequently", "notably", "ultimately"]
FEATS = []


def style_features(t):
    f = {}
    n_char = max(len(t), 1)
    words = re.findall(r"[A-Za-z']+", t)
    nw = max(len(words), 1)
    sents = re.split(r"[.!?]+", t)
    sents = [s for s in sents if s.strip()]
    ns = max(len(sents), 1)
    f["avg_sent_len_words"] = nw / ns
    f["avg_word_len"] = np.mean([len(w) for w in words]) if words else 0
    f["type_token_ratio"] = len(set(w.lower() for w in words)) / nw
    # punctuation rates (per 1k chars)
    for name, ch in [("comma", ","), ("semicolon", ";"), ("colon", ":"),
                     ("emdash", "\u2014"), ("question", "?"), ("exclaim", "!"),
                     ("paren", "("), ("dquote", '"')]:
        f[f"punc_{name}"] = 1000.0 * t.count(ch) / n_char
    # markdown / structure (per 1k chars)
    f["md_h1"] = 1000.0 * len(re.findall(r"(?m)^# ", t)) / n_char
    f["md_h2_h3"] = 1000.0 * len(re.findall(r"(?m)^#{2,3} ", t)) / n_char
    f["md_bullet"] = 1000.0 * len(re.findall(r"(?m)^\s*[-*] ", t)) / n_char
    f["md_numbered"] = 1000.0 * len(re.findall(r"(?m)^\s*\d+\. ", t)) / n_char
    f["md_bold"] = 1000.0 * len(re.findall(r"\*\*", t)) / n_char
    f["md_code"] = 1000.0 * t.count("```") / n_char
    f["md_table"] = 1000.0 * t.count("|") / n_char
    f["md_blockquote"] = 1000.0 * len(re.findall(r"(?m)^> ", t)) / n_char
    f["latex"] = 1000.0 * (t.count("$") + len(re.findall(r"\\\(|\\\[|\\frac|\\begin", t))) / n_char
    # ascii-diagram / box-drawing characters (Gemini hallmark)
    f["ascii_art"] = 1000.0 * len(re.findall(r"[+|\\/_=^<>]{2,}|[\u2500-\u257f]", t)) / n_char
    # first person
    f["first_person_I"] = 1000.0 * len(re.findall(r"\bI\b", t)) / n_char
    f["first_person_we"] = 1000.0 * len(re.findall(r"\b[Ww]e\b", t)) / n_char
    # hedging / signature words (per 1k words)
    low = t.lower()
    for h in HEDGE:
        f[f"w_{h}"] = 1000.0 * len(re.findall(rf"\b{h}\b", low)) / nw
    f["digit_ratio"] = sum(c.isdigit() for c in t) / n_char
    f["upper_ratio"] = sum(c.isupper() for c in t) / n_char
    return f


def build_style_matrix(texts):
    global FEATS
    rows = [style_features(t) for t in texts]
    FEATS = list(rows[0].keys())
    return np.array([[r[k] for k in FEATS] for r in rows], dtype=float)


# ----------------------------- main -----------------------------
def main():
    print("loading splits...", flush=True)
    tr_t, tr_y = load_split("train")
    va_t, va_y = load_split("val")
    oo_t, oo_y = load_split("val_ood")
    print(f"  train={len(tr_t)} val={len(va_t)} val_ood={len(oo_t)}", flush=True)

    # ---- semantic embeddings ----
    print("embedding (sentence-transformers all-MiniLM-L6-v2, CPU)...", flush=True)
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    def emb(texts):
        return enc.encode([t[:4000] for t in texts], batch_size=64,
                          show_progress_bar=False, normalize_embeddings=True)
    Etr, Eva, Eoo = emb(tr_t), emb(va_t), emb(oo_t)

    # ---- stylometric features ----
    print("stylometric features...", flush=True)
    Str_ = build_style_matrix(tr_t); Sva = build_style_matrix(va_t); Soo = build_style_matrix(oo_t)
    sc = StandardScaler().fit(Str_)
    Str_s, Sva_s, Soo_s = sc.transform(Str_), sc.transform(Sva), sc.transform(Soo)

    summary = {"counts": {"train": len(tr_t), "val": len(va_t), "val_ood": len(oo_t)}}

    # ---- classifiers ----
    def evaluate(name, Xtr, Xva, Xoo):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, tr_y)
        res = {}
        for sp, X, y in [("val", Xva, va_y), ("val_ood", Xoo, oo_y)]:
            p = clf.predict(X)
            res[sp] = {"acc": round(accuracy_score(y, p), 4),
                       "macro_f1": round(f1_score(y, p, average="macro", labels=PROVIDERS), 4),
                       "confusion": confusion_matrix(y, p, labels=PROVIDERS).tolist()}
        print(f"  [{name}] val acc={res['val']['acc']} val_ood acc={res['val_ood']['acc']}", flush=True)
        return clf, res

    print("classifiers...", flush=True)
    clf_sem, res_sem = evaluate("semantic", Etr, Eva, Eoo)
    clf_sty, res_sty = evaluate("stylometric", Str_s, Sva_s, Soo_s)
    summary["classifier_semantic"] = res_sem
    summary["classifier_stylometric"] = res_sty

    # ---- top discriminative style features per provider ----
    coefs = clf_sty.coef_  # (3, n_feat) in clf.classes_ order
    classes = list(clf_sty.classes_)
    top = {}
    for prov in PROVIDERS:
        ci = classes.index(prov)
        order = np.argsort(coefs[ci])[::-1][:8]
        top[prov] = [(FEATS[j], round(float(coefs[ci][j]), 3)) for j in order]
    summary["top_style_features"] = top
    for prov in PROVIDERS:
        print(f"  {prov} top style: {', '.join(k for k,_ in top[prov][:6])}", flush=True)

    # ---- mean style values per provider (interpretable) ----
    prov_means = {}
    allS = np.vstack([Str_, Sva, Soo]); allY = np.array(tr_y + va_y + oo_y)
    for prov in PROVIDERS:
        m = allS[allY == prov].mean(axis=0)
        prov_means[prov] = {FEATS[j]: round(float(m[j]), 3) for j in range(len(FEATS))}
    summary["provider_mean_style"] = prov_means

    # ---- 2D projections (UMAP) ----
    import umap
    def project(X, seed=0):
        return umap.UMAP(n_neighbors=25, min_dist=0.25, metric="cosine",
                         random_state=seed).fit_transform(X)

    print("UMAP projections...", flush=True)
    allE = np.vstack([Eva, Eoo]); allEy = np.array(va_y + oo_y)
    allSt = np.vstack([Sva_s, Soo_s])
    P_sem = project(allE)
    P_sty = umap.UMAP(n_neighbors=25, min_dist=0.25, metric="euclidean",
                      random_state=0).fit_transform(allSt)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    for ax, P, title in [(axes[0], P_sem, "SEMANTIC embedding space (content)\nMiniLM sentence embeddings"),
                         (axes[1], P_sty, "STYLOMETRIC feature space (style)\n~40 structural/lexical features")]:
        for prov in PROVIDERS:
            m = allEy == prov
            ax.scatter(P[m, 0], P[m, 1], s=10, alpha=0.55, c=COLORS[prov], label=prov, edgecolors="none")
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="best", frameon=True, markerscale=2)
    fig.suptitle("How the three providers occupy semantic vs. stylistic space (val + val_ood, n=%d)" % len(allEy),
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/provider_umap.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- separation summary figure: classifier accuracy bars ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["semantic\n(content)", "stylometric\n(style)"]
    valv = [res_sem["val"]["acc"], res_sty["val"]["acc"]]
    oodv = [res_sem["val_ood"]["acc"], res_sty["val_ood"]["acc"]]
    x = np.arange(len(labels)); w = 0.35
    ax.bar(x - w/2, valv, w, label="val", color="#6366f1")
    ax.bar(x + w/2, oodv, w, label="val_ood", color="#f59e0b")
    ax.axhline(1/3, ls="--", c="gray", lw=1, label="chance (0.33)")
    for i, (a, b) in enumerate(zip(valv, oodv)):
        ax.text(i - w/2, a + .01, f"{a:.2f}", ha="center", fontsize=9)
        ax.text(i + w/2, b + .01, f"{b:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.05)
    ax.set_ylabel("3-way classification accuracy")
    ax.set_title("Content barely separates providers; style separates them cleanly")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/style_separation.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- confusion matrices (stylometric, val_ood) ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (nm, res) in zip(axes, [("semantic", res_sem), ("stylometric", res_sty)]):
        cm = np.array(res["val_ood"]["confusion"], dtype=float)
        cmn = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3)); ax.set_xticklabels(PROVIDERS, rotation=30, fontsize=8)
        ax.set_yticks(range(3)); ax.set_yticklabels(PROVIDERS, fontsize=8)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                        color="white" if cmn[i, j] > .5 else "black", fontsize=9)
        ax.set_title(f"{nm}  (val_ood, acc={res['val_ood']['acc']:.2f})", fontsize=11)
        ax.set_ylabel("true"); ax.set_xlabel("predicted")
    fig.suptitle("Confusion matrices — linear probe on each representation (val_ood)", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/provider_confusion.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    with open(f"{OUT}/analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved figures + summary to {OUT}/", flush=True)
    print(json.dumps({k: summary[k] for k in ("classifier_semantic", "classifier_stylometric")}, indent=2))


if __name__ == "__main__":
    main()
