
## Part XVI — What each training method actually learned (trace-level prose analysis)

Parts XIV–XV established *what separates the providers* (stylometry) and *how each run scored*. This
part reads the **reasoning prose itself** — the `<reason_why>` text in every model's val traces — to
answer a different question: **what did each training method teach the model to *notice and say*?**

**Method.** `scripts/analyze_trace_quirks.py` parses the `<reason_why>` of all 414 val traces per
model. For each model it (a) measures prose stats (median length, fraction of rationales that name at
least one *concrete* measurable surface feature vs. pure vibes, fraction that are over-confident vs.
hedged), (b) tallies which surface features the prose points at (em-dash, headers, tables, ASCII
diagrams, lists, intensifiers, transitions, …), and (c) extracts representative verbatim rationales
per provider on **correct** predictions. Output: `blog-eval/analysis/trace_quirks.json`.

### XVI.1 — The shared "tell vocabulary" every healthy model converged on

Independently of the training recipe, every model that reasons coherently (SFT-goldcond, STaR,
cold-start RL, cold-start SFT, pure-accuracy peak) describes the **same three style signatures** — and
they line up exactly with the stylometric discriminators from Part XIV. Verbatim, from
`sft-goldcond` (val, correct):

> **CLAUDE →** *"…distinctively Claude-style markers, including the use of sincerity adverbs like
> 'genuinely,' 'honestly,' and 'precisely,' alongside essayistic phrasing… The voice is warm and
> argument-driven, employing second-person address ('you')… over the hedging or enumerative style
> typical of ChatGPT or the grandiose formalism of Gemini."*

> **CHATGPT →** *"…the distinctively hedging, enumerative style of ChatGPT, characterized by frequent
> phrases like 'may be,' 'can also,' and 'for example'… The structure relies heavily on explicit
> lists, numbered steps, and formal definitions (e.g., $E_i = P_i \times B_i$)…"*

> **GEMINI →** *"…Gemini's distinct formal register through heavy use of intensifiers like 'profound,'
> 'fundamentally,' and 'massive,' alongside a confident, declarative tone that avoids hedging. Its
> structure relies on complex mathematical formalization and ASCII diagrams…"*

So the model learned the **real** discriminators — Claude's em-dash/sincerity/second-person essay
voice, ChatGPT's hedged enumerated headers, Gemini's intensifier-laden ASCII/LaTeX density — the very
features the linear probe in Part XIV separates perfectly. The reasoning is a faithful natural-language
read-out of the stylometric signal, **not** topic/content (which Part XIV showed is intermixed).

### XVI.2 — Training recipe shapes the *character* of the reasoning

The same tell vocabulary is delivered very differently depending on how the channel was trained:

| Model (training) | val acc | median reason | concrete¹ | confident² | hedged² | pred distribution (collapse?) |
|---|---|---|---|---|---|---|
| `sft-goldcond` (gold-conditioned SFT) | 1.000 | 73 w | 0.98 | 0.35 | 0.14 | balanced 138/138/138 |
| `star-selfdistill` (STaR self-distill) | 0.952 | 82 w | 0.98 | 0.57 | 0.07 | balanced |
| `coldstart-rl` (SFT-60 → control GRPO) | 0.911 | 81 w | 1.00 | 0.36 | 0.06 | balanced |
| `coldstart-sft` (60-step SFT only) | 0.696 | 85 w | 0.99 | 0.46 | 0.07 | balanced |
| `rl-pureacc-peak` (pure-accuracy GRPO) | 0.384 | 91 w | 0.98 | **0.74** | 0.07 | **CHATGPT 336/414** |
| `rl-cheatsheet` (cheatsheet GRPO) | 0.372 | **15 w** | 0.33 | 0.07 | 0.01 | **CLAUDE 342/414** |
| `rl-entropydecay` (entropy-decay GRPO) | 0.290 | 23 w | 0.50 | 0.04 | 0.01 | **GEMINI 346/414** |
| `rlsd-e3` / `opcd-e2` (distillation, collapsed) | ~0.00 | **2000+ w**³ | 0.10 | 0.00 | 0.70 | 100% truncation |

¹ fraction of rationales naming ≥1 measurable surface feature. ² confident/hedged lexical markers.
³ un-terminated runaway text — the tag never closes.

**Reading the table:**

- **SFT (gold-conditioned & STaR) — the gold standard of *grounded* reasoning.** Compact (73–82 w),
  near-100% concrete, balanced across providers. STaR self-distillation makes the prose noticeably
  **more assertive** (confident 0.57 vs SFT's 0.35) while staying accurate — it learned to commit to a
  call. This is the prose that earns 0.95–1.00.

- **Cold-start RL — keeps SFT's grounding and stays balanced.** It inherits the concrete, balanced
  reasoning of its SFT-60 init (concrete 1.00, balanced predictions) and never collapses — consistent
  with its 0% truncation. RL here *polishes* the SFT reasoning rather than rewriting it. Example
  (GEMINI, correct): *"…relies heavily on mathematical formalism (using LaTeX equations like
  $\Phi: \mathcal{D}_{\text{Source}} \to \mathcal{D}_{\text{Target}}$) and ASCII diagrams… a distinct
  stylistic trait of Gemini."*

- **Pure-accuracy RL — concrete but over-confident and mode-collapsed.** Rationales stay long and
  feature-naming (concrete 0.98), but confidence jumps to **0.74** (the highest of any model) while
  accuracy *falls* to 0.38 because predictions **collapse onto CHATGPT (336/414)**. It learned to write
  a fluent, self-assured ChatGPT-justification for almost everything — reward hacking the majority class
  with persuasive but wrong prose.

- **Cheatsheet / entropy-decay RL — telegraphic word-salad that latches onto a single tell.** Median
  reasoning shrinks to **15–23 words** and concreteness halves. The single most-cited feature for both
  is the **em-dash** (Claude's #1 stylometric discriminator) — the channel fixates on one surface cue
  and discards the rest, then mode-collapses onto one provider each (cheatsheet→CLAUDE 342,
  entropy-decay→GEMINI 346). The prose degrades into incoherent fragments, e.g. cheatsheet (GEMINI):
  *"Verbose ontology taxonomy erupts + 'motion capture' typographic caps abandon hyphenation
  mid-transcript, emphatic glyph bombs in thesis echoes late-diffusion implosion, graphic-explanatory
  boxes perform oversized formatting excess."* — it still *gestures* at real tells (caps, diagrams) but
  has lost the ability to argue.

- **RLSD / OPCD (the collapsed distillation runs) — runaway, un-grounded reasoning.** Concreteness
  craters to 0.10 and hedging explodes to 0.70 as the model emits 2000+ tokens that **never close the
  `</reason_why>` tag**, frequently **echoing the input blog back verbatim**, e.g. rlsd-e3:
  *"- 'It is invoked by autocrats and anarchists… functioning as a seemingly unassifiable benchmark of
  legitimate governance…'"* (quoting the article, not analysing it). The reasoning channel has stopped
  doing classification and become an un-anchored text generator → 100% truncation, ~0 accuracy.

### XVI.3 — The throughline

The three providers' tells are **learnable and verbalizable**: every grounded model reads them off in
plain language, and what it names (em-dash & sincerity = Claude; hedged headers & lists = ChatGPT;
intensifiers & ASCII/LaTeX density = Gemini) matches the stylometric probe one-for-one. What differs is
**discipline of the channel**: SFT and cold-start RL keep the reasoning short, concrete and balanced;
unanchored reasoning-RL progressively trades grounded analysis for a confident single-class shortcut
(pure-accuracy), then for em-dash-fixated fragments (cheatsheet/entropy-decay), then for runaway echo
(RLSD/OPCD). This is the prose-level fingerprint of the same finding as Parts X–XIII: the tells were
never the problem — keeping the reasoning channel *anchored while it is rewarded* is.
