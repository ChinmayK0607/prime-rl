# Can an LLM Learn to Tell Other LLMs Apart? An RL Journey

*Teaching Qwen3.5-9B to discover — in its own words — what makes Claude, ChatGPT,
and Gemini write the way they do.*

---

## TL;DR

We took a 9B model and asked it a deceptively simple question: *given a blog post,
which AI wrote it — Claude, ChatGPT, or Gemini?* The interesting part isn't the
label. It's the **reasoning**. Our north-star goal was never to build a perfect
classifier; it was to use reinforcement learning to make the model **emergently
discover and articulate** what distinguishes each provider's writing voice — with
**no rubric**, so the discriminators are learned, not handed to it.

Eight GRPO runs later, here's the short version:

- The task is **trivially separable in the data** (a tiny formatting classifier hits
  97.5% accuracy) but the **base model sits near random** (~35% on the 3-way task).
- Plain correctness-only RL kept **reward-hacking the class prior** — it would ace
  two providers and quietly abandon the third. The *abandoned* class rotated from run
  to run (ChatGPT → Gemini → ChatGPT) because nothing in the loss anchors the model's
  marginal "what do I guess when shown one text" prior.
- **Symmetric contrastive pairs** (Run 7) fixed *discrimination* — the model can tell
  every pair of providers apart — but not the single-text marginal.
- An **entropy bonus** added directly to the GRPO loss (Run 8) produced our **best
  validation result yet (0.466 vs 0.226 base)** and kept all three classes alive, at
  the cost of some volatility.
- Along the way the model produced the real deliverable: **per-provider style
  fingerprints** in its own words (Claude = academic lecture voice; ChatGPT = "Worked
  Example" headers + tables + LaTeX; Gemini = grand metaphors like "the Rosetta Stone"
  and "the dark matter of intelligence").

---

## 1. The Question

Large language models have *voices*. Read enough output and you start to feel it —
ChatGPT's tidy taxonomies, Gemini's sweeping metaphors, Claude's measured lecture-hall
prose. But "you feel it" isn't science. We wanted a model to make those intuitions
**explicit and verifiable**.

The setup:

- **Task:** given one blog post, name the authoring provider — **CLAUDE / CHATGPT /
  GEMINI**.
- **The actual deliverable:** the `<reason_why>` text. We care *why* the model thinks
  what it thinks, because that articulated reasoning is the description of each
  provider's style.
- **Constraint:** the reasoning must be **learned via RL with no rubric**. If we hand
  the model a checklist ("look for em-dashes, count headers…"), we've biased the
  finding and learned nothing. The discriminators have to emerge.
- **Method:** GRPO (Group Relative Policy Optimization) in `prime-rl`, **labeled
  reward only** for now (an LLM-judge reasoning reward is a deferred ablation).
- **Model:** `Qwen/Qwen3.5-9B`, **thinking OFF**.
- **Hardware:** a single **8×H100** node.

Random baseline for the balanced 3-way task = **0.333**.

---

## 2. The Data

### Source
The `copilot-sdk-blogs` dataset (HuggingFace). Each row is a blog post written by a
named provider model, tagged with a topic category. The blogs are real, long-form
content (~3,000+ words on average) across categories like history, politics,
economics, science, and technology.

The dataset started as a **binary** problem (Claude vs ChatGPT, 126 rows balanced
63/63 across 9 categories) and we expanded it to the **3-way** provider task with
**137 / 137 / 137** examples (411 train rows), collapsing the two Gemini variants into
a single GEMINI label.

### The validation set is deliberately hard
Validation is **held out by topic category** — entire categories (history, politics,
economics-systems — the lowest base accuracy) never appear in training. This means a
good val score reflects **genuine generalization of style detection**, not topic
memorization.

### The key dataset insight (and the central tension)
We ran a diagnostic: how separable is this data, really?

- A **RandomForest on 8 formatting features** (em-dash frequency, header counts, bold,
  length, blank-line patterns) → **97.5% validation accuracy**. Em-dash usage alone
  carried 42% of the importance.
- **TF-IDF on the *stripped* text** (formatting removed) → **100% validation accuracy**.

So the signal is **massive and clean**. And yet the base model sat near random. The
model wasn't *using* the signal — it was doing vague prose analysis and ignoring the
stylometric fingerprints sitting in plain sight.

That created a fork: formatting is *legitimate* style signal (a model that notices
em-dash habits is learning something real) — but it's also a **shortcut** that could
let the model bank on surface cues instead of deeper voice. Our compromise:
**strip augmentation** — randomly strip markdown/formatting from ~40% of training rows
(em-dashes → commas, headers/bold/bullets removed, blank lines collapsed). We verified
the stripped content stays ~100% separable, so this forces the model to find deeper
signal without destroying the task.

---

## 3. The Baseline: pass@4

Before any training, we ran a **pass@4** evaluation with data-parallel inference
across all 8 GPUs (thinking off, generous 16k-token budget to stay close to training
conditions).

On the original **binary** task:

| Metric  | Score |
|---------|-------|
| pass@1  | 56.0% |
| pass@2  | 77.5% |
| pass@4  | **94.4%** |
| parse failures | 0% |
| truncation | 0% |

That's a beautiful RL target: a large **pass@1 → pass@4 headroom** (the model *can*
get it right, it just isn't reliable) with a clean, verifiable reward. A per-row
reward analysis confirmed **77.8% of rows were "mixed"** (1–3 of 4 correct → non-zero
GRPO advantage), with only 5.6% all-wrong — close to ideal.

On the harder **3-way** task the base model was much weaker: **pass@1 ≈ 0.35, pass@4 ≈
0.79** — and 0.79 is barely above the random pass@4 ceiling of ~0.80. In other words,
**near-zero discriminative signal** out of the box. Plenty of room for RL to work.

---

## 4. The RL Setup

We used **GRPO in `prime-rl`** with a custom `verifiers` environment.

- **GPU split:** 4 GPUs train (FSDP, HF implementation) + 4 GPUs inference (vLLM,
  tensor-parallel 2 → two replicas). We started 6/2 but profiling showed we were
  **inference-bound**, so rebalancing to 4/4 roughly doubled rollout throughput.
- **GRPO math:** `batch_size = 144` rollouts/step ÷ `group_size = 12` = **12
  prompts/step**. Advantage is computed *within* each group of 12 rollouts of the
  same prompt. A **zero-advantage filter** drops groups where every rollout got the
  same reward — so both all-correct and **all-wrong** groups contribute no gradient.
  (Remember that last part — it becomes the villain of this story.)
- **Sequence budget:** 16,384 tokens, 4,096 completion, temperature 1.0.
- **Reward:** exact-match on the parsed `<answer>` (weight 1.0). The `<reason_why>` is
  *not* parsed or scored — reasoning is free to roam. The prompt has **one** added
  line asking for holistic qualitative judgment, and **no rubric**.

### Two early infrastructure battles worth remembering

1. **The "empty response" mystery.** Qwen3.5's `reasoning_parser` was routing the
   model's output into a hidden "reasoning channel" and returning empty `content` →
   verifiers scored empty completions → eval errors of 18–71%. The fix was to disable
   the reasoning parser at the resolver level (setting it to `"None"` in the TOML
   doesn't propagate, due to `exclude_none` serialization). After the fix, eval error
   dropped to **0.0%**.

2. **The object-vs-dict trap.** `verifiers` passes message **objects** to reward
   functions at scoring time but serializes them to **dicts** in saved rollouts. A
   reward reader that filtered on `isinstance(m, dict)` silently dropped every message
   at runtime while *every offline test passed*. This cost us a full debugging cycle
   on Run 6's contrastive pairs.

---

## 5. The Journey: Eight Runs

Here's the heart of it. Each run was a hypothesis about why the previous one failed.

### Run 1 — Strict easy→hard curriculum → **COLLAPSED**
We sorted training prompts strictly by difficulty. This clustered same-difficulty
prompts together; the always-wrong tail produced **all-wrong (zero-advantage)
batches** → trainable fraction cratered to ~1% → the policy reward-hacked to a single
class (Gemini). Validation fell *below* baseline (0.419 → 0.366). **Lesson:**
difficulty clustering starves the gradient.

### Run 2 — Interleaved 4/4/4 curriculum
We rebuilt the data so every 12-prompt window is provider-balanced (4/4/4) with a
gentle short→long drift. This removed the dead-batch collapse (trainable fraction back
to 56–100%). **Lesson:** balanced windows remove the single-class-dump incentive
(dumping into one class only scores 1/3). This balance — not learning rate — was the
primary fix.

### Run 3 — Strip augmentation
Added the 40%-stripped rows to force deeper signal. Validation stayed **flat/declining
(~0.40, below baseline)**. The confusion matrix told the real story — a **zero-sum
recall reshuffle**: Claude recall rose 25%→69%, but **ChatGPT collapsed 55%→19%**,
dumped into Claude. **Lesson:** ChatGPT-vs-Claude is the genuinely hard pair (both
write conversational prose), and correctness-only reward makes it locally optimal to
ace the easy classes and dump the ambiguous one.

### Run 4 — Hard-pair auxiliary task
We added an explicit **Claude-vs-ChatGPT 2-way** auxiliary task (label space
restricted to the two confusable providers), scored with the same plain reward. A
rubber-duck critique killed a tempting alternative — a cost-matrix reward (−0.5/0/1) —
by showing it would make all-wrong groups *trainable* and let GRPO reinforce a "safe
escape" class. The hard-pair task **fixed the ChatGPT collapse (62%→83%)** but
**flipped the bias** — now Claude and Gemini got dumped into ChatGPT. Macro recall
slightly *down*. **Lesson:** we kept trading one single-class bias for its opposite.

### Run 5 — Strict per-step-balanced curriculum
We made *every* step exactly 4/4/4 gold, drawn only from each class's
*middle-difficulty* pool (so groups stay "mixed" and survive the zero-advantage
filter). Train reward climbed 0.33→0.625 with no hard collapse — **but the gain was an
illusion**: the model was acing Claude+Gemini and driving **ChatGPT recall from 52%
(base!) to 0%**. Because each step is balanced, Claude✓ + Gemini✓ + ChatGPT✗ = 8/12 =
0.67 reward — *rising reward masked a class abandonment*.

This run crystallized the **mechanism**: once ChatGPT recall → ~0, every ChatGPT
prompt becomes an all-wrong group → the **zero-advantage filter drops it** → **no
corrective gradient ever flows back** → an absorbing collapse. Strict balance fixes the
*gold* distribution but cannot stop this. Strikingly, **the base model was
better-balanced than the RL'd model** — RL was actively *destroying* ChatGPT recall.

### Run 6 — Contrastive pairs (Claude vs ChatGPT)
The structural fix: interleave a **contrastive pair** task — show the model TWO texts
(one Claude, one ChatGPT) and ask it to assign each. The magic: since the authors
*differ*, any constant-label policy is ≥50% wrong → the group **stays mixed** → it
survives the zero-advantage filter → **a restoring gradient always flows** on the
hard boundary.

It worked — for that boundary. ChatGPT recall stayed healthy (0.86–0.94). Validation
improved 0.245→**0.324** and **truncation self-healed from 60.8% → 19.1%** (the model
learned to emit its answer instead of over-reasoning). **But the collapse ROTATED:**
**Gemini** — which had no contrastive protection — collapsed to ~0. **Lesson:** a
contrastive pair only protects the *one* boundary it covers.

### Run 7 — Symmetric pairwise coverage
The obvious extension: cover **all three boundaries**. Each step = 6 single 3-way rows
+ 6 contrastive pairs (2× each of Claude-vs-ChatGPT, Claude-vs-Gemini,
ChatGPT-vs-Gemini), ~50% contrastive.

Results:
- ✅ **Gemini collapse fixed** — Gemini became the *most stable* class (recall
  0.44–0.79).
- ✅ **All-boundary discrimination preserved** — late pair accuracy 0.72–1.00 across
  boundaries. The model can demonstrably tell *every* pair apart.
- ⚠️ **But the oscillation rotated AGAIN** — ChatGPT became under-predicted on
  *single* texts (its predict-share pinned at 0.05–0.16), though not a hard collapse
  (the pairs keep its gradient alive).
- ➖ Validation roughly flat, ending 0.314.

This is where the **root cause finally crystallized**:

> Contrastive pairs guarantee per-boundary **discrimination** and prevent absorbing
> collapse, but they do **not** constrain the **single-text marginal prediction
> prior**. With **no KL-to-reference and no entropy term in prime-rl's GRPO loss**,
> nothing anchors "what does the model guess when shown ONE text," so the favored
> single-text class drifts freely (and the abandoned class rotates run to run). Pairs
> fix *"can it tell them apart"*; they don't fix *"what it predicts when shown one
> text."*

### Run 8 — Entropy bonus in the loss *(current)*
If the problem is an unconstrained marginal, attack the marginal directly — in the
loss. prime-rl's GRPO has no entropy term, so we added one.

The design (validated against a rubber-duck critique that *rejected* a naive
`+β·logprob` entropy term as a degenerate, anti-distillation estimator): a
**detached-surprisal entropy policy-gradient bonus**. We fold each token's surprisal
under the rollout policy, `s_t = −inference_logprob` (detached, clamped ≥ 0), into the
advantage:

```
shaped_advantage = adv_tau · advantage + entropy_coef · s_t
pg_loss          = keep_mask · shaped_advantage · importance_ratio
```

Minimizing `−pg_loss` *raises* the trainer's probability of high-surprisal (rare)
tokens → raises entropy **and** up-weights the rare answer-label token of an
under-predicted class, keeping that class sampled and receiving corrective gradient.
It's applied only on the DPPO trust-region `keep_mask`, it's compatible with the fused
LM head (uses only sampled-token logprobs — no full-distribution entropy backward, no
frozen teacher), and we verified numerically that `entropy_coef = 0` reproduces the
default loss exactly.

**Results (best of any run):**

| Step | 0 | 4 | 8 | 12 | 16 | **20** | 24 | 28 |
|------|---|---|---|----|----|--------|----|----|
| Val reward | 0.226 | 0.201 | 0.255 | 0.201 | 0.230 | **0.466** | 0.387 | 0.270 |
| Truncation | 61% | 64% | 63% | 78% | 44% | **0.5%** | 11% | 36% |

- ✅ **Best validation of any run: 0.466** (vs 0.226 base; prior bests 0.324 / 0.314).
  The peak coincided with **truncation collapsing to 0.5%** — the model briefly learned
  to answer concisely instead of over-reasoning into truncation.
- ✅ **The primary goal was met:** ChatGPT's single-text predict-share stayed **0.17–
  0.58 the whole run** (vs Run 7's 0.05–0.16 pin). The entropy bonus kept the
  previously-abandoned class alive — exactly as designed. The marginal was more
  balanced than any prior run through ~step 23.
- ⚠️ **But volatile.** Validation is non-monotonic (0.466 → 0.270), and the prior
  drifted Claude-heavy at the very end (step 27: Claude 0.64 / ChatGPT 0.17 / Gemini
  0.19). Entropy **raises the floor** on the abandoned class but doesn't fully **pin**
  the marginal — matching the prediction that token-entropy is a *weak* proxy for the
  cross-prompt marginal. The best checkpoint (step 20) is preserved by a best-val
  janitor.

---

## 6. The Deliverable: Learned Style Fingerprints

Throughout, the point was never the label — it was the reasoning. Harvesting the
`<reason_why>` text from correct rollouts gives us the model's own, **un-rubricked**
description of each provider's voice:

- **CLAUDE** — an academic *"lecture-room voice"*: rhetorical framing ("the strength of
  X is…"), philosophical precision blended with accessible analogies, complex
  self-referential transitions, structured "worked example" pedagogy.

- **CHATGPT** — the most distinctive learned tell is **literal "Worked Example X"
  headers**; Markdown tables and taxonomies; LaTeX equation blocks; "Risk assessment"
  style lists; "One [noun] is X" sentence frames; "In practice" transitions; early
  formal notation.

- **GEMINI** — **grand, ornamental metaphors** as the signature ("the Rosetta Stone,"
  "the dark matter of intelligence," "a toddler exploring the world," "the holy
  grail"); analogy-driven intros and conclusions; numbered lists plus dense technical
  exposition; idiosyncratic section hooks ("Modern computing is built on a
  contradiction").

These are exactly the *"what makes them distinct"* descriptions we set out to elicit —
and the model arrived at them on its own, from a reward that only ever told it whether
the final label was right.

---

## 7. What We Learned

1. **A clean, separable signal in the data does not mean the model uses it.** The base
   model ignored fingerprints a tiny RandomForest nails at 97.5%.

2. **Correctness-only multi-class RL has an unconstrained degree of freedom: the
   marginal prior.** Each prompt-group only rewards its own gold; nothing keeps the
   model's *global* "what do I guess" distribution calibrated. So it drifts, and a
   class gets abandoned.

3. **The zero-advantage filter turns abandonment into an absorbing state.** Once a
   class hits ~0 recall, its groups go all-wrong → filtered → no gradient → it can
   never recover *on single-text data*.

4. **Contrastive pairs are a clean structural fix for discrimination and collapse** —
   different-author pairs are always partially wrong, so their gradient never dies —
   **but they don't fix the single-text marginal.**

5. **The marginal needs a loss-level anchor.** An entropy bonus folded into the GRPO
   advantage gave us the best result yet and kept all classes alive — but token-level
   entropy is a *weak* proxy for the cross-prompt marginal, so it raises the floor
   without fully stabilizing it.

6. **A rubber-duck critique loop caught at least three would-be dead ends** before we
   burned GPU hours on them (the cost-matrix reward, the naive `+β·logprob` entropy
   term, and the contrastive-triple zero-advantage hole).

---

## 8. What's Next

- **Stabilize Run 8's gains:** LR decay / early-stopping on best-val (the step-20
  checkpoint is already preserved), and possibly pairing the entropy bonus with a
  gentle batch-level marginal regularizer (carefully — a naive version invites a
  "balanced but wrong" hack).
- **The deferred ablation:** an **LLM-judge reasoning reward** — reward grounded,
  evidence-based reasoning so the label is *driven by* style evidence rather than a
  free-floating prior. (We've held this back deliberately: the whole point is for the
  reasoning to emerge unbiased first.)
- **Deeper fingerprint harvest:** systematically aggregate stripped-vs-formatted
  reasoning to separate surface tells from genuine voice.
- **The Gemini generalization gap:** even the base model only gets ~15% on held-out
  Gemini categories — a real data-diversity question, separate from optimization.

---

*Infrastructure: `prime-rl` GRPO, 8×H100 (4 train / 4 inference), Qwen3.5-9B,
thinking off, labeled-reward-only. Full run-by-run technical log in `RL_LOG.md`.*

---

# Part II — Hardening the evaluation and breaking the Gemini wall (Runs 9–12)

Part I ended on an honest worry: validation accuracy was promising but *volatile*,
and one class (Gemini) kept getting abandoned. Part II is the story of chasing that
down to its root — and the root turned out to be nowhere near where we thought.

## 9. The held-out set was lying to us (truncation + a leaky split)

Two evaluation artifacts were quietly distorting every number:

1. **Greedy decoding was looping.** Validation used temperature 0 (greedy) for a clean,
   deterministic curve. But on long blogs, greedy Qwen3.5-9B fell into a *repetition
   loop* — it would free-form deliberate (“The user wants me to identify…”) and loop
   (“…CLAUDE. Wait… CLAUDE…”) for the entire 4096-token budget, **never emitting an
   `<answer>`**. That's a truncation, which scores 0. It hit long **Gemini** blogs
   hardest (56–78% of them truncated), so a big chunk of "Gemini recall = 0" was really
   "Gemini answer never got emitted." Training (temperature 1.0) truncated 0%.
   **Fix:** sample validation at temperature 0.7 / top-p 0.95. Truncation fell from
   ~78% to ~5–10%. (prime-rl's eval config exposes temperature/top-p; that's all it took.)

2. **Category holdout risked leakage and confounded the question.** We rebuilt the split
   to hold out 15% of *topics within each category* (all three provider variants of a
   held-out topic go to validation together), with an explicit anti-leakage assertion,
   and parked three whole categories as a separate out-of-distribution set.

With a trustworthy in-distribution validation set, we re-ran the default recipe (Run 11).
Result: truncation was fixed, but the marginal **collapsed cleanly to CLAUDE** —
prediction share C 0.96 / P 0.04 / G 0.04 by step 20. Gemini still flatlined at ~0.04.
So it *wasn't* a truncation mirage and it *wasn't* category generalization. Something
deeper was wrong.

## 10. The decisive test: is Gemini even separable?

Before burning more RL, we asked the question we should have asked at the very start:
**can a trivial, label-only classifier separate these providers at all** on the
held-out set? We trained a plain TF-IDF + logistic-regression on the training blogs and
tested on the topic-holdout validation set — including on **formatting-stripped** text
(em-dashes, headers, bold, bullets all removed), so it couldn't cheat on surface markup:

```
2-way separability (topic-holdout val, chance = 0.50):
  CLAUDE  vs CHATGPT   content (stripped) 1.000
  CLAUDE  vs GEMINI    content (stripped) 1.000
  CHATGPT vs GEMINI    content (stripped) 1.000   ← the "impossible" boundary
3-way (stripped):  1.000 accuracy, recall 1.00 / 1.00 / 1.00, balanced predictions
```

**Gemini-vs-ChatGPT is perfectly separable from content alone.** This single result
reframed the entire project. The wall was **not** a representational ceiling, and **not**
a data problem (we'd even regenerated a larger, humanities-weighted corpus to be sure).
The discriminative signal is massive, clean, and linearly decodable. The model was simply
**failing to use it** — a pure RL *optimization* pathology: GRPO's zero-advantage filter
drops groups where every sample is wrong, so the moment the policy stops sampling Gemini
on Gemini blogs, that class receives **no gradient at all** and the unconstrained marginal
slides into a one-class attractor. (prime-rl's loss has no reference-KL and no entropy
term, so nothing holds the marginal in place.)

That diagnosis points at two levers we *can* pull under the label-only constraint:
keep the abandoned class **sampled** (so gradient keeps flowing), and pour extra
contrastive pressure onto the **specific** boundary that's failing.

## 11. Run 12 — the two-arm ablation that worked

Both arms build on the truncation fix + topic-holdout split + the larger dataset, and
both add the **entropy bonus** from Run 8 (now retested cleanly, since the greedy-loop
confound that sank the earlier entropy attempt is gone). The ablation is on the
**curriculum**:

| | contrastive mix / step | best val | Gemini recall | marginal at peak |
|---|---|---|---|---|
| Run 11 (baseline) | balanced, no entropy | 0.37 | **0.04** | collapsed to CLAUDE (0.96) |
| **12B** balanced + entropy | 2 CvP / 2 CvG / 2 PvG | 0.371 | ~0.18 | **balanced** C0.43 / P0.49 / G0.18 |
| **12A** PvG-weighted + entropy | 1 CvP / 2 CvG / **3 PvG** | **0.398** | **0.27** | C-heavy at peak, rebalances to C0.45 / P0.33 / G0.25 |

Both arms **keep all three classes alive** — the first time in the whole project that
Gemini doesn't get abandoned on a clean held-out set. Entropy keeps the rare class
sampled so its gradient survives; concentrating contrastive pairs on the hard
Chatgpt-vs-Gemini boundary (12A) lifts Gemini **faster and higher** (recall ~0.27 vs
~0.18, and the best validation reward of any run on this harder evaluation).

**The honest caveat:** the entropy bonus is *late-phase unstable*. Around step ~32 it
starts inflating verbose, looping generations; truncation explodes and the reward
crashes. We rely on a best-validation checkpoint janitor to keep the peak, and the next
iteration should decay the entropy coefficient or early-stop. This is a stability knob,
not a wall.

## 12. The deliverable: what RL taught the model to *say*

The whole point was never the label — it was the `<reason_why>`. Harvesting the
rationales from **correct** classifications (label-only reward, no rubric, no judge),
the model has learned to articulate genuine, provider-distinctive signatures:

- **Claude** — *essayistic, nuanced, philosophical, measured*; first-person framing
  (“I want to resist…”), long-form argument that resists overt signposting.
- **ChatGPT** — *worked examples, numbered lists, example headers, pedagogical, highly
  structured*; the recognizable “explain-with-a-worked-example” default.
- **Gemini** — *ASCII diagrams, visual scaffolding*; synthesizes disparate disciplines
  into grand, dramatic metaphors and abstract declarations.

These emerged from nothing but a binary correct/incorrect signal. That's the result we
were after: not a classifier, but a model that can **say what distinguishes each voice**.

## 13. Where it stands

- **The wall is optimization, not data.** Proven by perfect label-only separability.
  The remaining gap on Gemini is about keeping gradient flowing to a class GRPO wants to
  abandon — solvable with anti-collapse pressure, not more data.
- **Best configuration so far:** PvG-weighted contrastive curriculum + entropy bonus +
  non-greedy validation, best checkpoint at step 16 (val 0.398, Gemini recall 0.27, all
  three classes alive). Checkpoint preserved.
- **It generalizes out-of-distribution.** On three entirely held-out *categories*
  (history, politics, economics-systems) the best checkpoint scores 0.338 with
  per-class recall 0.61 / 0.25 / 0.15 — a modest ~0.06 drop from in-distribution, Gemini
  weaker (0.27 → 0.15) but **still alive, no collapse**, and the model stays concise
  (1.2% truncation) on prose it has never seen.
- **Next:** entropy decay / early-stop to remove the late-phase instability; the
  long-deferred **LLM-judge reasoning reward** ablation (now that unbiased reasoning has
  emerged first); and closing the OOD Gemini gap with broader register coverage.

*Full run-by-run technical log in `RL_LOG.md`; per-provider fingerprint harvest in
`files/style_fingerprints_run12.txt`.*

---

# Part III — Do the modern GRPO variants help? (Run 13)

A natural question: the 2024–26 literature is full of GRPO refinements — DAPO, CISPO,
DPPO, MaxRL, Dr. GRPO, ScaleRL. Several look tailor-made for our exact failure (a rare
class whose gradient keeps getting clipped away). Do they beat the entropy-bonus recipe?

We first checked what `prime-rl` already implements, and the answer reframes the question:
its **default loss is already DPPO** (a Divergence-PPO total-variation trust region on the
probability *difference*) and its **advantage is already Dr. GRPO** (`r − mean`, no
std-normalization). So "try DPPO/Dr. GRPO" was already our baseline. That left two genuinely
different levers to test, each as a clean single-variable change against the Run-12A entropy
incumbent (val 0.398):

- **MaxRL** (custom *advantage*): reweight so a success on a rarely-solved prompt is
  amplified — `A_i ∝ (r_i − r̂)/r̂`. Aligned with the original pass@k framing.
- **CISPO** (custom *loss*): clip the importance-sampling *weight* and stop-gradient it, but
  **never mask** — so gradient keeps flowing on the rare tokens DPPO would drop.

Both were implemented, unit-tested, and reviewed (the review caught a zero-mean bug in the
clipped MaxRL advantage, which we fixed by re-centering).

**Both underperformed — and the *reasons* are the real result.**

| Arm | Best val | What happened |
|---|---|---|
| Entropy (incumbent) | **0.398** | truncation 38% → 2.5%, all 3 classes alive |
| MaxRL | 0.311 | **truncation stuck ~37%**, val capped |
| CISPO | 0.291 → 0.241 | **collapsed to ChatGPT**, truncation rose to 43% |

- **MaxRL** does lift Gemini's share a little, but by down-weighting the easy,
  high-success prompts it starves the very signal the base model still needs to learn the
  basic *"be concise, emit `<answer>`"* behaviour. Truncation never falls, so a third of the
  validation set scores zero and the ceiling sits around 0.31.
- **CISPO** revealed something sharper. Every training step logged **`Max Off-Policy 0`** —
  the asynchronous loop is only one step off-policy, i.e. effectively **on-policy**, so the
  importance ratio is ≈ 1 and CISPO's clip/stop-gradient machinery **never activates**. With
  its one distinguishing mechanism inert, CISPO collapses back into the plain default loss —
  and reproduces the original marginal collapse (this time onto ChatGPT).

**The conclusion.** For this task the trust-region family (CISPO / DAPO / DPPO) are
near-no-ops, because they only differ from the default loss *off-policy*, and our setup is
essentially on-policy. Advantage reshaping (MaxRL) does change behaviour but along the wrong
axis. The lever that actually moves the needle is **anti-collapse pressure on the
unconstrained class marginal** — which, empirically, only the entropy bonus supplies (it was
also the only run that broke the repetition/truncation loop, 38% → 2.5%). The path to better
numbers is therefore not a fancier estimator but a *better-controlled entropy lever* —
entropy decay, or a DAPO-style overlong soft-penalty to tame the late-phase blow-up — on top
of the recipe we already have.

*This is a useful negative result: it rules out a large, popular branch of the design space
for on-policy, collapse-prone classification RL, and explains why.*

---

# Part IV — Protecting the deliverable and taming the late-phase collapse (Runs 14–15)

Part III ended with a prediction: the next gain would come not from a fancier estimator but
from a **better-controlled entropy lever**. Two problems stood between us and that gain, and
we attacked them one clean variable at a time.

## 14. The answer-only reward was quietly eating the deliverable

Our whole premise is that RL should make the model *articulate* the per-provider tells inside
`<reason_why>`. But the reward only ever checked the `<answer>`. Inspecting late-run rollouts
showed the predictable consequence of optimizing what you measure: by step 24 about **11% of
correct rollouts had an empty or stub `<reason_why>`** — e.g. a 29-token
`<reason_why></reason_why><answer>CHATGPT</answer>` still scoring a perfect 1.0. Over-training
was collapsing the deliverable into a bare label.

The naive fix (a separate weighted "format" reward) is a known trap here: under GRPO group
normalization a small additive bonus becomes *full-strength* inside all-wrong groups whose only
variance is format, reinforcing arbitrary wrong-but-parsed answers. So we used a **multiplicative
reason-gate** instead: a rollout scores 1.0 only if the answer is correct **and** the
`<reason_why>` carries ≥12 *distinct* alphabetic words (a length floor that simultaneously blocks
repetition-padding, no LLM judge needed). Because a wrong answer still scores 0 regardless of its
prose, all-wrong groups stay zero-advantage-filtered — the gate adds no hackable surface.

**Run 14** (reason-gate + a mild truncation-only advantage penalty) did exactly its job:

- The deliverable was **fully protected**: the live `reason_ok` diagnostic went to 1.00 and
  stayed there. The empty-reason hack was gone, *non-hackably*.
- Truncation stayed solved (35.8% → 1.5%).
- Peak reason-gated val **0.3607 @ step 28** — an honest *correct-and-reasoned* number.

But Run 14 also exposed two separable failures, and the rest of this part is about fixing each:

| | Symptom | Cause |
|---|---|---|
| **(A) Marginal drift** | peak capped ~0.36; CLAUDE recall 0.54 → 0.19 while ChatGPT 0.31 → 0.66 | the entropy bonus keeps all 3 classes *alive* but does not *balance* them |
| **(B) Late collapse** | val 0.36 → **0.017** and truncation 1.5% → **91.5%** over steps 28→40 | the *undecayed* entropy bonus eventually drives the model back into the repetition loop |

## 15A. Entropy decay kills the late collapse — and raises the peak

Failure (B) has an obvious shape: the entropy bonus is essential *early* (it's what breaks the
repetition/truncation loop in the first place), but once concise answering is established, a
constant 0.02 bonus just keeps pushing the model toward high-surprisal tokens until it falls
back into the loop. So we made the coefficient **decay**: hold 0.02 through step 16 (where
truncation is already solved), then anneal linearly to 0 by step 32. Everything else identical
to Run 14. (Mechanically this needed the live training `step`/`max_steps` plumbed into the loss
function — forwarded only to custom losses that declare them, so the default path is untouched.)

The result is the cleanest single-variable win of the series:

| Step | Run 14 (constant entropy) | Run 15A (decayed entropy) |
|---:|---|---|
| 16 | 0.3408 (trunc 7.7%) | 0.3557 (trunc 6.2%) |
| 24 | 0.3433 | **0.3756** (trunc 1.0%) ← peak |
| 28 | **0.3607** (peak) | 0.2687 |
| 36 | 0.1766 (trunc 34%) | 0.3731 (trunc 0.2%) |
| 40 | **0.0174 (trunc 91.5%)** | **0.3408 (trunc 2.7%)** |

Two things to read off this table. First, the **catastrophic collapse is gone**: where Run 14
ends at 0.017 with 91% of the validation set truncated into the repetition loop, Run 15A ends at
a healthy 0.34 with truncation still under 3% — the run is *usable to the last step*. Second, the
**peak went up** (0.3756 vs 0.3607). The late-phase still *wobbles* between ~0.27 and ~0.38 — but
that wobble is the **marginal drift of failure (A)**, not the loop (truncation never exceeds 3%).

That isolates the last problem precisely. With the loop tamed, what remains is the unconstrained
class marginal — and that is what the triowise task in Run 15B is built to constrain.

## 15B. Triowise contrastive — a structural constraint on the marginal *(in progress)*

The pairwise tasks from Part I cured *per-boundary* collapse, but a pair only ever exposes **two**
of the three labels, so the global marginal is still free to drift. A **trio** closes that gap:
three texts A/B/C, each authored by one of the three providers with **repeats allowed**, and the
model must assign all three. The composition is deliberately *mixed* (sampling each slot i.i.d.
uniform gives ~67% two-of-one-plus-one, ~21% all-distinct, ~12% all-same) so there is no
"always three different labels" shortcut and no constant-class escape — and the gold marginal
stays uniform, matching validation.

The subtle part is the reward, and a design critique caught the trap before we burned a run on
it. The *obvious* choice — 1.0 only if **all three** slots are correct — is **sparsest exactly on
the under-predicted class**: the gold triples that contain it are the least likely to ever produce
a fully-correct rollout, so they generate *no* gradient precisely where drift is worst, and the
drift would *self-reinforce*. The fix is to reward the **fraction of slots correct**. This keeps
the rollout group *mixed* (rollouts differ in how many slots they nail) so a restoring gradient
flows even before any rollout gets all three — and it stays non-hackable, because its only
within-group variance axis *is* partial correctness (for uniform-marginal trios, no constant or
"escape" strategy beats 1/3-per-slot in expectation). Raising slot-accuracy therefore *requires*
tracking the true class frequencies.

**Run 15B delivered the highest peak of the entire project.** Replacing the pair stream with
trios (everything else identical to Run 14, including the *undecayed* entropy bonus) pushed the
reason-gated validation reward to **0.4080 at step 16** — above Run 15A's 0.3756, above Run 14's
0.3607, and the best macro-recall we'd seen (0.413). The trio's marginal pressure is real.

But Run 15B *also* collapsed late (step 36 → 0.13 / 24% truncation, step 40 → 0.005 / 84%),
exactly like Run 14 — because it kept the *undecayed* entropy bonus. That is the clean,
satisfying part of the result: **the two problems are independent and so are their fixes.**

| Run | Change vs Run 14 | Peak val | End state (step 40) |
|---|---|---:|---|
| 14 | (baseline: reason-gate) | 0.3607 | **collapsed** 0.017 / trunc 91% |
| 15A | + entropy **decay** | 0.3756 | **stable** 0.341 / trunc 2.7% |
| 15B | + **trio** data | **0.4080** | collapsed 0.005 / trunc 84% |

Entropy-decay fixes the *late collapse* (and nudges the peak up); the trio task raises the
*peak* (via marginal pressure) but does nothing for the collapse. They touch different failure
modes — so the obvious move is to combine them.

## 15E. Trio + entropy decay — the combined recipe

The final run keeps the trio data and its per-slot-accuracy reward but swaps the constant
entropy bonus for the decayed one. The hypothesis was simply additive: the ~0.41 peak that the
trio buys, held stable the way the decay schedule held Run 15A.

It paid off on the two things we built it for. The reason-gated validation reward peaked at
**0.4005 at step 20** — tied, within run-to-run noise, with Run 15B's project-best 0.408 — and at
that peak the model was, for the first time, **both accurate and balanced**:

| Class | Recall @ step 20 | Predicted share |
|---|---:|---:|
| CLAUDE | 0.336 | 0.289 |
| CHATGPT | 0.410 | 0.348 |
| GEMINI | **0.455** | 0.323 |
| **macro** | **0.400** | (uniform ≈ 0.333) |

The predicted-label marginal is essentially uniform — no class is being dumped — and **GEMINI,
the laggard that sat near 0.26 recall for the entire project, is now the *strongest* class at
0.455.** That is the trio's marginal pressure doing exactly what it was designed to do, and it is
the cleanest per-class balance of any run. Truncation was 4% at the peak and stayed low (≤2%)
through step 28.

The late phase is the one honest wrinkle. Run 15E does **not** suffer the catastrophic collapse
that ended Runs 14 and 15B (step 40 at 0.005 / 84% truncation). But once the entropy floor anneals
to *zero* at step 32, a **milder** re-degradation sets in — truncation drifts 16% → 32% → 20% and
the reward slides to ~0.16 by step 40. Run 15A (pairs + decay) held flat at <2.7% under the same
schedule, so the trio's denser per-slot pressure appears to want a small *nonzero* entropy floor
late that the pairwise task did not. The best-val janitor captures the step-20 checkpoint
regardless, so the deliverable is unaffected — but the obvious next tweak is a decay schedule that
floors at a small ε instead of 0.

| Run | Recipe | Peak val | Marginal @ peak | End state (step 40) |
|---|---|---:|---|---|
| 14 | reason-gate | 0.3607 | CLAUDE-skew | collapsed 0.017 / trunc 91% |
| 15A | + entropy **decay** | 0.3756 | drifting | **stable** 0.341 / trunc 2.7% |
| 15B | + **trio** | 0.4080 | CLAUDE-skew | collapsed 0.005 / trunc 84% |
| 15E | + **trio + decay** | **0.4005** | **uniform** (G 0.455) | mild 0.164 / trunc 20% |

**The deliverable.** The best checkpoint — Run 15E, step 20 — is the model this whole arc was
aiming for: it classifies all three providers at balanced accuracy without collapsing the marginal,
keeps answers concise (4% truncation), and, because of the reason-gate, *only* earns reward when it
also writes a substantive `<reason_why>`. Harvesting those reason texts from its correct rollouts
gives the project's actual payload — the stylistic tells the model taught *itself*, now for the
first time read off a checkpoint where every class is alive:

- **CLAUDE** — essayistic and argumentative: *deliberate* essay structure, *philosophical* framing,
  *hedging*, a first-person *voice*, *rhetorical* synthesis.
- **CHATGPT** — worked-examples and scaffolded: *numbered* *sections*, *worked* math *examples*,
  heavy *LaTeX*/*formatting* and *delimiters*, pedagogical *scaffolding*.
- **GEMINI** — visual and confident: *ASCII* *diagrams*, math *notation*, a *confident*,
  *narrative* explanatory register.

These line up with the Run-12 harvest, but this is the first time they come from a checkpoint that
is simultaneously accurate, balanced, and non-collapsed — which is the result the experiment set
out to earn.

---

# Part V — The diagnostic that reframes everything: it was never a data problem

After Run 15E we had a balanced, non-collapsed 0.40 classifier and a tidy story. But 0.40 raw
accuracy on a 3-way task — barely better than the 0.33 you get by guessing — kept nagging. The
working assumption all along was a **data/signal ceiling**: the three providers' styles overlap so
much on these blogs that ~0.40 is simply the most any model could extract, so the fix would be more
or harder data. Before spending a generation budget on that premise, we tested it directly.

**The test.** Train a *classical* classifier — TF-IDF n-grams + logistic regression — on the same
`train` split the RL model learns from, and evaluate it on the same `val` split. Crucially, the
train and val sets share **zero** `(category, topic)` pairs (234 vs 36, no overlap), so this is
honest generalization to unseen topics, not memorization.

**The result was unambiguous, and it demolished the premise:**

| Classifier | What it sees | Val accuracy (unseen topics) |
|---|---|---:|
| LogReg, word 1–2grams | full text | **100%** |
| LogReg, char 3–5grams | full text | **100%** |
| LogReg, word 1–2grams | only the **first 1500 characters** | **100%** |
| LogReg | **function words + punctuation only** (zero topic content) | **94%** |
| **RL'd Qwen3.5-9B** | full text | base 28% → **best 40%** |

A linear bag-of-words model separates the three providers **perfectly** on topics it has never seen
— and it still hits 94% using *nothing but* the frequencies of function words (`the`, `of`, `may`,
`we`) and punctuation, with every content word removed. The signal isn't subtle, isn't
topic-dependent, and isn't hiding deep in the text; it's in the **first 1500 characters**, in the
*style*. The features the linear model leans on are exactly the tells the RL model had been
groping toward:

- **CHATGPT** — hedging and enumerative: `may`, `may be`, `for example`, `such as`, `not only`, `depends on`.
- **CLAUDE** — conversational and sincere: `you`, `genuinely`, `precisely`, `honest`, `worth`, `exactly`.
- **GEMINI** — grandiose and formal: `furthermore`, `we must`, `profound`, `mathematically`, `paradigm`, `fundamentally`.

**What this means.** The ~0.40 ceiling was never the data. The classes are trivially, perfectly,
generalizably separable; a logistic regression learns them in seconds. The RL'd 9B leaves *sixty
points* of perfectly-available signal on the table. The bottleneck is **exploitation, not
information** — scalar-reward RL from scratch, with thinking off, is simply a very weak teacher for
a discrimination that supervised learning finds effortless. And the reason-gate is *not* the
culprit: at the 15E peak `reason_ok` is 0.975, so raw accuracy and gated reward are the same number.

This reframes the whole project. The next phase is not "more RL tricks to claw past 0.40." The
right levers are the ones that actually inject the available signal into the model:

1. **An SFT / distillation warmup on the gold labels.** The features are so learnable that a small
   supervised pass should lift raw accuracy toward the linear model's range before RL ever runs.
2. **Few-shot / a fingerprint cheatsheet in the prompt** — give the model the tells it keeps
   rediscovering.
3. **Thinking *on*** — let the model do the explicit feature-checking the linear model does in its
   weights.

We are generating one final, larger, balanced corpus (every category doubled, 15→30 topics) not
because separability needs it, but to give an SFT warmup something rich to learn from and to give us
a large, many-topic held-out set for a trustworthy learning signal. The experiments that follow are
designed around *exploitation*, not around chasing a data ceiling that, it turns out, was never
there.

---

# Part VI — The final corpus, and separability that survives a register change

We generated one last, deliberately large corpus: every category doubled from 15 to 30 topics
(630 topics, all four providers, short and long), bringing the blog set from ~2,700 to **5,258**
clean documents, balanced across providers with zero coverage gaps. Not because separability needed
more data — Part V showed it does not — but to give a supervised warmup something rich to learn
from and to give us a *large, many-topic, multi-register* held-out set on which to stress-test the
finding.

Rebuilt into a fresh split — train 2,682 (894/class), an in-distribution val of 414 (72 unseen
topics across 18 categories), and an out-of-distribution `val_ood` of 471 where three whole
categories (history, politics, economics) are held out **entirely** — the linear probe was rerun:

| Probe | val (unseen topics) | val_ood (unseen **categories**) |
|---|---:|---:|
| TF-IDF word 1–2grams | **100%** | **100%** |
| Function words + punctuation only (no content) | **98.8%** | **94.3%** |

The result doesn't just replicate at 2× scale — it *strengthens*. A bag-of-words model is **perfect**
even when the evaluation categories were never seen in training, and a model that sees *nothing but*
function-word and punctuation frequencies still hits 94% across a register change. Provider voice is
not a topic artifact and not a register artifact; it is a stable, low-level stylistic signature.

This also closes an old thread. Back in the early runs we attributed the model's poor
cross-register validation to GEMINI's "voice not transferring" from technical to humanities writing.
That story is now falsified: at the linear level GEMINI's voice transfers across register at 100%.
The cross-register gap, like the in-distribution gap, was never about the signal being absent — it
was about the RL'd model failing to read a signal that is, in information terms, blatant.

So the data chapter of this project is closed. The corpus is large, balanced, clean, and — provably
— saturated with separable signal. Everything that follows is about **getting that signal into the
model**, which is where the next experiments are aimed.

---

# Part VII — Exploitation experiments: the model *can* read the tells, it just won't on its own

The separability work (Parts V–VI) proved the signal is there and trivially linearly readable.
This part attacks the real bottleneck: **getting a 9B model to actually use it.** Three probes, no
training yet, on the v2 split (val: 414, held-out topics; val_ood: 471, entirely held-out
categories).

## The three no-training probes

| Condition | val acc | val_ood acc |
|---|---|---|
| Baseline (plain prompt) | 0.454 | 0.410 |
| **+ Cheatsheet (style tells in system prompt)** | **0.674** | **0.592** |
| Few-shot (3 truncated exemplars) | 0.321 (collapsed) | — |

Per-class recall, val cheatsheet: CLAUDE 0.877, CHATGPT 0.333, GEMINI 0.812.

Three things to sit with:

1. **The base model already beats every RL run.** Plain Qwen3.5-9B scores 0.454 on val — higher
   than the best RL checkpoint's 0.40. The RL runs were not just plateauing; relative to the base
   model they were *net-negative*. Whatever GRPO was optimizing, it was eroding the model's prior
   ability to read style.

2. **Handing the model the tells is worth +0.22 (val) / +0.18 (val_ood), with zero training.**
   The cheatsheet is nothing but the empirically-derived per-provider style tells (the same features
   the logreg uses) written into the system prompt. The model clearly *can* apply them — it just
   does not surface them unprompted. This is the cleanest possible confirmation that the ceiling is
   **attention/elicitation, not capability and not data.** And it transfers cross-register
   (val_ood 0.41→0.59), exactly as the separability analysis predicted.

3. **CHATGPT is the universal hard class** (recall 0.19 base → 0.33 cheatsheet, lowest everywhere).
   The model systematically under-predicts CHATGPT (its hedging/enumerative register is the easiest
   for a human to name but the model's weakest prior). Any training fix has to target this class
   specifically.

4. **Few-shot backfired.** Three truncated exemplars collapsed the model onto GEMINI (pred_share
   0.89). Truncation + in-context exemplars is the wrong lever here; discarded.

## Where this points

The cheatsheet *is already a deployable 0.67 system.* The remaining questions are whether we can (a)
internalize the tells so no cheatsheet is needed at inference, (b) beat 0.67, and (c) specifically
rescue CHATGPT. That is the SFT-warmup experiment (Design A), now running: a **gold-label-conditioned
teacher** (base + cheatsheet, told the answer) writes a short, grounded rationale for every train
blog; we distill those onto the *plain* prompt with the canonical gold answer. Using the teacher to
*explain* (not to *decide*) keeps all three classes fully and equally covered — avoiding the
rejection-sampling bias that would have starved the already-weak CHATGPT class.

## VII.b — SFT warmup result: the tells *do* internalize, and the task collapses to solved

The Design A SFT warmup ran (Qwen3.5-9B, `impl="hf"`, thinking OFF, 270 steps, ckpt @
90/180/270, body truncated ≤3000 tok / seq_len 4096 to dodge the `fla` gated-delta-rule
CUDA grid-Z>65535 crash; loss 1.38→~0.35). We then evaluated **each checkpoint on the
*plain* prompt — no cheatsheet, no few-shot — on both val and the cross-register val_ood.**

| checkpoint | val acc | val_ood acc | CHATGPT recall | GEMINI recall |
|---|---|---|---|---|
| base (no train)      | 0.454 | 0.410 | 0.19 | 0.57 |
| base + cheatsheet    | 0.674 | 0.592 | 0.33 | 0.81 |
| **SFT step 90**      | 0.906 | 0.909 | 0.98 / 1.00 | 0.74 / 0.73 |
| **SFT step 180**     | **1.000** | **1.000** | 1.00 | 1.00 |
| **SFT step 270**     | **1.000** | **1.000** | 1.00 | 1.00 |

Per-class recall and prediction share are all exactly balanced (0.333 each) at convergence —
zero errors across all 414 val + 471 val_ood examples, **including the out-of-distribution
register split.**

Why we trust a perfect score (it is not a leak or a parsing artifact):
- **No leakage.** SFT teacher data was generated over v2-*train* blogs only; an explicit
  prefix (first 400 alnum chars) *and* 300-char substring check against val/val_ood found
  **zero** overlapping blogs.
- **Believable learning curve.** step_90 is at 0.91 with GEMINI as the lone laggard (recall
  0.73); GEMINI is fully learned only by step_180. A monotone, class-by-class curve is what
  real feature acquisition looks like, not the step-function you would see from memorization.
- **Stochastic robustness.** Eval sampling is temp 0.7 / top_p 0.95 — yet every single one of
  885 samples is correct. A spurious match would leak errors under sampling noise.
- **The reasoning is genuine and grounded.** The model now *articulates the tells on its own*:
  Claude → sincere essayistic adverbs ("genuinely," "honestly"), warm first person; ChatGPT →
  enumerative structure, hedging, equations-in-prose; Gemini → grandiose intensifiers
  ("profound," "fundamentally"), declarative register, ASCII/LaTeX. These are exactly the
  empirically-derived fingerprints from Parts V–VI — now produced from the *plain* prompt with
  no cheatsheet present.

**This closes the thesis.** Parts V–VI proved the data is ~100% separable and the 0.40 RL
ceiling was model feature-blindness, not data. A short SFT warmup that simply *shows the model
how to read its own features* lifts plain-prompt accuracy 0.454 → 1.000 and rescues the
historically hopeless CHATGPT class (0.19 → 1.00). RL was never the missing piece; elicitation
was. The cheatsheet got us a training-free 0.67; SFT internalizes the same knowledge to a
perfect, cheatsheet-free, OOD-robust classifier that explains itself.

**Implications for next steps.** With val and val_ood both saturated, RL-polish (the planned
Design A→RL arm) has no headroom to chase and is dropped. The only remaining open ablation is
Design C (thinking-ON) — purely to see whether an explicit reasoning budget changes the
*articulation*, not the accuracy. Best deployable artifact: `outputs/sft_warmup/weights/step_180`
(earliest perfect checkpoint).

---

## Part VIII — The final RL run (Design B): cheatsheet-elicited GRPO

One last plain-RL run (no LLM judge), built to test a sharp question: *if we lift the
feature-blind base policy out of its blind spot with a train-derived style cheatsheet, can
GRPO then improve on top of it?* The cheatsheet (CLAUDE sincerity adverbs, CHATGPT
hedging/enumeration, GEMINI ASCII-diagram + grandiosity) was derived from **training data
only** — no val/test leakage — and injected into both the train and eval system prompts.
Init from BASE, trio data, entropy-decay loss + truncation-penalty advantage, 40 steps.

**Why this design.** From the base policy, GRPO has zero gradient on saturated full-text
accuracy and collapses from the ~0.40 feature-blind floor. The cheatsheet raises the floor to
a non-degenerate ~0.44 *gated* reward with **100% trainable (mixed-reward) groups** — exactly
the variance GRPO needs.

**What happened.** Gated val reward climbed smoothly 0.443 → **0.662** (peak @ step 28), and
eval truncation collapsed 24% → ~0%. A clean, healthy RL curve with no class collapse.

| step | 0 | 12 | 16 | 20 | 24 | **28** | 32 | 40 |
|---|---|---|---|---|---|---|---|---|
| gated val reward | .443 | .517 | .614 | .632 | .634 | **.662** | .649 | .627 |
| eval truncation | 24% | 19% | 5.5% | 4.7% | 0.5% | 0.2% | 1.7% | 1.7% |

**But did it learn the features?** Evaluating the best checkpoint (step 28) two ways:

| step_28 regime | val | val_ood |
|---|---|---|
| WITH cheatsheet (trained regime) | 0.659 | 0.650 |
| WITHOUT cheatsheet (internalization check) | **0.348** | **0.378** |
| base+cheatsheet, no RL (prior probe) | 0.674 | 0.592 |

**The honest conclusion.** Remove the cheatsheet and the RL'd model falls straight back to
base-level feature-blindness (~0.36). So **RL did not internalize the discrimination ability** —
it remains entirely cheatsheet-contingent. What RL *did* accomplish is real but narrow: it
taught the policy to **use** the elicited features reliably — eliminating truncation, passing the
substantive-reason gate, and improving **OOD robustness** (val_ood 0.592 → 0.650). Raw val
accuracy stayed flat versus the training-free base+cheatsheet (0.674 → 0.659).

This is the cleanest possible statement of the whole project's thesis:
- **The cheatsheet ELICITS** features (training-free, +0.18–0.22).
- **RL POLISHES** their use (format discipline, gate-passing, OOD) but creates no new knowledge.
- **Only SFT INTERNALIZES** them — to a perfect, cheatsheet-free, OOD-robust 1.000 (Part VII.b).

Feature-blindness, not data and not RL optimization, was always the binding constraint; supervised
elicitation is the only lever that removes it. Best deployable artifact remains
`outputs/sft_warmup/weights/step_180`. RL artifacts kept for the colleague's four-way judge
comparison: `outputs/rl_3way_trio_cheat/weights/step_{28,40}`.

### Part VIII addendum — could the *old* 0.40 RL model just use the cheatsheet?

A natural question: was the cheatsheet a prompt trick we could have bolted onto the earlier
plain-RL run? We tested it directly — eval the prior 0.40 checkpoint
(`rl_3way_trio_entdecay/step_40`, trained WITHOUT a cheatsheet) WITH the cheatsheet at eval time:

| Model + cheatsheet @ eval | val | val_ood |
|---|---|---|
| base (untrained) + cheatsheet | 0.674 | 0.592 |
| new RL (trained WITH cheatsheet) + cheatsheet | 0.659 | 0.650 |
| prior 0.40 RL (trained WITHOUT cheatsheet) + cheatsheet | **0.324** | **0.304** |

It performs **worse than the untrained base** — and reveals the truth about the "0.40 plateau":
it was actually **collapse**. The old model predicts GEMINI ~87% of the time (CLAUDE/CHATGPT
recall ~0–2%) and largely ignores the injected cheatsheet (NONE rate 10–18%, instruction-
following damaged). The cheatsheet cannot rescue a collapsed policy post-hoc.

**Conclusion:** the cheatsheet is not an inference-time add-on; it must be present **during
training**. It is precisely what kept the new run from collapsing — giving GRPO a non-degenerate
reward floor with 100% trainable (mixed-reward) groups — whereas the cheatsheet-free run collapsed
to a single class. This is the strongest evidence that the historical ~0.40 RL ceiling was a
training-dynamics collapse under feature-blindness, not a data or capacity limit.

---

## Part IX — Why the RL didn't internalize: a signal/entropy analysis

Config grounding: GRPO group size G=12, Dr.GRPO mean-centered (un-normalized) advantage,
surprisal entropy bonus beta=0.02 held through step 16 then annealed to 0 by step 32, train
temperature 1.0, reason-gated reward (1 answer token + up to ~4096 reason tokens), lr 1e-6, 40 steps.

Model: one 3-way categorical decision, correct-prob q (under temp), reward r in {0,1} = gate-passed
correctness, effective success p=qg. Advantage A_i = r_i - rbar. Let v = q(1-q).

### M1 — sparse-signal SNR ceiling
Correct-logit score: d log pi/dz_c = 1{y=c} - q. Per-group gradient ghat = sum_i A_i (1{y_i=c}-q).
  Signal  E[ghat] = G*Cov(r,1{y=c}) = G*p(1-q) = G*g*q(1-q)        (proportional to v -- the parabola)
  Noise   sqrt(Var ghat) ~ sqrt(G*v)
  => SNR ~ sqrt(G*v) <= sqrt(12 * 0.25) ~ 1.73  even in the best case (q=0.5).
The signal is the quadratic v=q(1-q): max at 0.5, zero at the ends. Only 1 of ~4096 completion
tokens carries correctness signal. Groups stayed 100% mixed (v near max) so saturation was avoided
-- but the per-step SNR is barely above 1, hence slow movement.

### M2 — entropy x sparseness = noise (quadratic in beta)
The surprisal bonus beta*s_t (s_t=-log pi_t) is added to EVERY kept token, incl. ~L reason tokens
that have near-zero task gradient. Full-sequence power:
  SNR^2_seq = (G v)^2 / ( G v  +  beta^2 * L * sigma_s^2 )
Entropy noise is quadratic in beta, linear in token count L. Crossover where entropy noise overtakes
the sparse answer signal:
  beta* ~ G v / (L * sbar) ~ (12*0.25)/(700*1.5) ~ 0.003.
We ran beta=0.02 ~ 6*beta* for the whole hold phase (steps 0-16); the linear anneal only crosses
below beta* near step ~30.
DATA CONFIRMS: eval reward flat while beta high (0.443->0.517 over steps 0-12), accelerates as beta
decays (0.614->0.662, steps 16-28), PEAKS at step 28 -- exactly as beta enters the signal-dominant
regime. The anchor-the-marginal bonus spent the first 40% of training drowning the answer signal.

### M3 — the cheatsheet is a confounder: internalization gradient ~ 0
d E[r]/d theta_features  proportional to  I(r ; theta_features | cheatsheet) ~ 0,
because the cheatsheet already supplies the tells in-context: conditioned on it, encoding the same
features in weights has ~zero marginal reward value, so there is NO gradient to internalize. RL can
only learn the cheap in-context routing (few bits). Information budget: lr 1e-6 x SNR~1.7 x 40 steps
is enough for shallow routing, nowhere near enough to write a 3-way discriminator into 9B weights.
=> 0.66 WITH cheatsheet, 0.35 WITHOUT. No SNR would have fixed this; it is structural.

### Synthesis & prescriptions
M1 caps signal (sqrt(G v) ~ 1.7; 1 signal token in 4096). M2 adds off-task noise ~ beta^2 L
(we ran 6x over beta*; reward peaked as beta decayed). M3 zeroes the internalization gradient
(confounder). Together: healthy-looking training, slow/capped gains, zero internalization.
Fixes: (1) mask the entropy bonus to the answer token / keep beta <= beta* ~ 0.003 or decay from
step 0; (2) raise SNR via larger G (SNR ~ sqrt G) not more steps; (3) to internalize, WITHDRAW the
cheatsheet during training (curriculum) -- or use SFT, whose dense per-token teacher gives an
O(sqrt L) higher gradient SNR per example than a 1-bit reward on one token. The dense-vs-sparse
supervision gap is the whole story behind SFT=1.000 vs RL-capped-0.66.
(Heuristic: assumes near-independent token scores, single signal token, constant gate prob; the
skeleton is exact and beta* predicts the observed step-28 peak.)

## Part X — The pure-accuracy / no-reasoning RL test (M2, isolated)

Part IX prescription #1 was: remove the reason tokens so the entropy/noise term beta^2*L
collapses (L: ~700 -> ~9) and the sparse answer-token signal is no longer diluted. Part X runs
exactly that experiment as a clean diagnostic.

Setup (zero-leakage, tight): init from BASE (no cheatsheet), v2 uniform single-3way data
(balanced 894/class train; val 276/class; val_ood ~314/class; exact/prefix/substr leakage = 0),
answer_only prompt (completion = a single `<answer>LABEL</answer>`, decode_len mean 9.6 tokens),
require_reason=false, max_completion_tokens=32, G=12, batch 144, lr 1e-6, train temp 1.0 / eval
temp 0.7, surprisal_entropy_decay beta=0.02 (hold_frac 0.4, end_frac 0.8), truncation_penalty adv,
evals on val+val_ood every 4 steps. (Infra notes: eval goes through the chat-completions client, so
enable_thinking=false must be set via eval extra_body chat_template_kwargs or eval truncates 100%;
run must be HF-offline to avoid a hub file-list hang; filesystem weight-broadcast + unpruned ckpts
fill disk -- use keep_last and nccl/keep_last.)

### Result -- cheatsheet-free eval accuracy (the number prior reasoning+cheatsheet RL capped at ~0.40)
  step0  val 0.348 / ood 0.375   (= chance, 0.333)
  step4  val 0.383 / ood 0.404
  step8  val 0.384 / ood 0.414   (CLAUDE already nearly dropped: predicted 3/828)
  step12 val 0.580 / ood 0.572   (CLAUDE fully dropped: 0 predictions)
  step16 val 0.650 / ood 0.662   (PEAK)
  step~20 ABORT: RuntimeError "10 consecutive zero-trainable batches" (prime-rl guardrail).

### What actually happened -- the confusion matrix (step16, temp 0.7)
  blog-val            pred:CLAUDE CHATGPT GEMINI   recall
    gold CLAUDE            0      270       6       0.000
    gold CHATGPT           0      267       9       0.967
    gold GEMINI            0        5     271       0.982
  blog-val-ood
    gold CLAUDE            0      313       1       0.000
    gold CHATGPT           0      314       0       1.000
    gold GEMINI            0        4     310       0.987

The policy learned a NEAR-PERFECT CHATGPT-vs-GEMINI discriminator (recall 0.97-1.00 each) and
absorbed ALL of CLAUDE into CHATGPT (CLAUDE recall exactly 0.000). The 0.65-0.66 "peak" IS the
2-class ceiling (2/3 = 0.667). Once the policy is deterministic, every GRPO group is uniform-reward
(CLAUDE groups all-wrong -> A=0; CHATGPT/GEMINI groups all-right -> A=0) -> zero gradient -> abort.
Note the collapse happened DURING the entropy hold phase (beta at full 0.02): the single-token
surprisal bonus was too weak to keep CLAUDE sampled once the reward gradient sharpened the policy.

### Reading (calibrated -- what this does and does NOT show)
DOES show: removing reason-token dilution lets answer-only GRPO extract real discriminative signal
and clear the prior ~0.40 cheatsheet-free ceiling -- it learned a strong 2-way (CHATGPT/GEMINI)
classifier from a 1-bit reward. That is genuine learning, not a degenerate fluke (0.97+ recall on two
classes cannot come from class-dropping alone). Directionally consistent with M1/M2: concentrating the
sparse signal on the single answer token (zeroing beta^2*L) unblocked optimization that the long-L
reasoning runs could not achieve.
Does NOT show: that M1/M2 was THE cause of the prior 0.40 ceiling (setups differ: prior =
cheatsheet-trained reasoning policy measured cheatsheet-free; this = from-scratch answer-only). And
it did NOT reach a stable 3-way solution -- it collapsed to an absorbing 2-class optimum. The 0.66 is
a transient peak of a degenerate policy, not a reportable 3-way accuracy.
New finding: CLAUDE collapses into CHATGPT (not GEMINI) -- the CLAUDE/CHATGPT style boundary is the
fine one; GEMINI is the easy split. Under sparse single-token RL the policy carves the easy boundary
and abandons the hard one, because GRPO has no within-group reward variance to recover a class it has
stopped sampling. (Consistent with the earlier SFT observation that CHATGPT/CLAUDE were the close,
last-learned classes; SFT's dense per-token teacher resolves the fine boundary, sparse RL cannot.)

### Open controls (cheap, decisive -- next, if pursued)
  1. Label-rotation control: rerun with neutral/rotated labels (A/B/C). If the SAME provider (CLAUDE)
     collapses -> content/feature-driven; if the collapse follows the label token -> tokenization/prior
     bias. (Labels tokenize 3-each with distinct first tokens CL / CHAT / G -- no shared-prefix clash,
     so a pure-tokenization artifact is unlikely but untested.)
  2. Binary CLAUDE-vs-rest answer-only GRPO: if it learns -> the 3-way failure is exploration/collapse;
     if it cannot -> residual CLAUDE feature-blindness under sparse RL.
  3. Anti-collapse 3-way (only if 1-2 implicate exploration): larger G (more within-group variance) +
     higher train temp (keep CLAUDE sampled) + lower lr (5e-7, slower sharpening) + entropy floor +
     early-stop if CLAUDE pred-rate -> 0. NOT a blunt global-entropy bump (won't help post-collapse).

### Part X addendum — base-prior control refutes the label-token-bias hypothesis (free)
Concern (rubber-duck): maybe CLAUDE collapses because its label token has a low prior, not because
its blogs are hard. Test, from the saved rollouts, the per-step CLAUDE prediction rate on blog-val:
  step0(BASE) CLAUDE 522/828 = 0.63   step4 0.29   step8 0.00   step12 0.00   step16 0.00
The BASE model OVER-predicts CLAUDE (63% -- it is the model's default/favorite class), and RL
actively SUPPRESSES it to 0. This is the opposite of a low-prior/token-bias story. So CLAUDE is not
hard to EMIT; the CLAUDE-vs-CHATGPT decision BOUNDARY is the hard one. Sparse single-token GRPO
reassigns the base's lazy CLAUDE-default mass into CHATGPT (the stylistically adjacent class),
collapsing the fine boundary into an absorbing state, while cleanly carving out the easy GEMINI
split. Mechanism is content/boundary difficulty + exploration collapse, NOT label tokenization.
(This makes a label-rotation rerun lower priority; the prior is already the wrong sign for token bias.)

---

## Part XI — OPSD ladder: internalization without gold-conditioned teaching

Motivation: SFT solves the task (1.000) but the teacher is GOLD-CONDITIONED — it is *told* the
answer (ANALYST NOTE) and writes a rationale to justify it. The OPSD ladder asks a sharper
question: can the model internalize the provider tells from its OWN reasoning, where the gold
label is NEVER shown during generation and is used only as a binary verifier to gate which
self-generated trajectories become training data? (Framing per rubber-duck: this is "no
per-example gold shown during rationale generation", not "zero leakage" globally — gold still
selects the kept set, and the kept completions contain the correct label.)

### E0 — forgetting / capability baseline (the yardstick) — DONE
Question: how much did the SFT-to-1.000 damage general capability? A method that internalizes
the tells WITHOUT this cost is strictly better.
Eval (scripts/eval_e0_forgetting.py, BASE vs sft_warmup/step_180, HF teacher-forcing):
  - General-text perplexity (4 generic paragraphs + 150 gsm8k test items):
    BASE 2.779  ->  SFT 2.866   (+3.2%)  ==> negligible forgetting.
  - 6 capability probes (arithmetic, rate, factual x2, trick-reasoning, one-line code):
    all intact and correct on the SFT model (e.g. 47x23 worked, "all but 9"->9, sum-of-squares
    one-liner identical to base).
Verdict: the gold-conditioned SFT internalized the tells at essentially ZERO general-capability
cost. This is the bar E1-E4 must match or beat.

### E1 — STaR self-distillation (on-policy, verifier-gated) — IN PROGRESS
Pipeline (scripts/gen_star_e1.py, single round):
  1. PLAIN pass: BASE + plain prompt (no cheatsheet), thinking-off, k=3 @ temp0.7 on all 2682
     train. Gate: majority-vote==gold AND >=2/3 correct.
  2. HINT pass on the still-wrong: BASE + train-derived CHEATSHEET (general rules, NOT the gold
     answer), k=2. Gate: >=1/2 correct; reject rationales that explicitly cite the cheatsheet/
     hint/rules (must justify from observable text only).
  3. SFT target = PLAIN prompt + blog -> the model's OWN <reason_why>+<answer> (gold canonical).
Yield: 1913/2682 accepted (71%).  Accepted by (gold, source):
     CLAUDE  plain=162 hint=518 total=680
     CHATGPT plain=297 hint= 60 total=357   <- bottleneck (base's hard class; cheatsheet barely
     GEMINI  plain=363 hint=513 total=876       helps CHATGPT, +60 only)
Class-balanced to cap=357/class = 1071 SFT rows (avoids amplifying the base's class skew, per
rubber-duck). Note the asymmetry: the cheatsheet rescues CLAUDE massively (+518) and GEMINI
(+513) but barely moves CHATGPT (+60) — consistent with CHATGPT being the stylistically "average"
class that the surface-tell rules under-serve.
SFT config: examples/blog_author_id/sft_star_e1.toml (BASE init, seq4096, body<=3000tok, lr1e-5,
max 200 steps ~3 epochs, ckpt every 40). Eval: plain prompt val/val_ood per ckpt. [result pending]

#### E1 RESULT — self-distilled reasoning reaches 0.93/0.95 without gold-conditioned teaching
Eval = PLAIN prompt (no cheatsheet), val + val_ood, temp0.7 top_p0.95, thinking-off, NONE=0 everywhere.
| ckpt | val acc | val_ood acc | CLAUDE r | CHATGPT r | GEMINI r (val) |
|------|---------|-------------|----------|-----------|----------------|
| step40  | 0.577 | 0.533 | 0.587 | 0.681 | 0.464 |
| step80  | **0.932** | **0.953** | 1.000 | 0.797 | 1.000 |  <- BEST
| step120 | 0.891 | 0.879 | 0.964 | 0.877 | 0.833 |
| step160 | 0.928 | 0.909 | 1.000 | 0.819 | 0.964 |
| step200 | 0.896 | 0.883 | 1.000 | 0.797 | 0.891 |

Takeaways:
- STaR self-distillation (gold NEVER shown during generation; only verifier-gates) reaches
  val 0.932 / val_ood 0.953 — vs base 0.45, the RL ceiling ~0.40-0.66, and gold-conditioned SFT
  1.000. So ~93-95% of the task is learnable from the model's OWN verifier-gated rationales.
- val_ood >= val at the peak (0.953 > 0.932): the self-generated rationales generalize OOD, not
  memorize. Format perfect (NONE=0). Both extreme classes hit 1.000 recall.
- The entire residual gap to 1.000 is CHATGPT (recall 0.80-0.89, never 1.0). This traces directly
  to the data-yield asymmetry: in generation the cheatsheet rescued CLAUDE (+518) and GEMINI (+513)
  but barely CHATGPT (+60), so CHATGPT's 357 kept rows are almost all low-margin plain-correct
  rationales — the weakest training signal for the hardest class.
- INTERPRETATION (calibrated, per rubber-duck): reaching 0.93 does NOT prove the *reasoning text*
  caused it (a clean labeled subset alone can teach a classifier). What gold-conditioned teaching
  buys over self-distillation is concentrated exactly where self-generated reasoning is scarce
  (CHATGPT) — i.e. the teacher's value is rescuing the class the model cannot yet articulate on its
  own. RECOMMENDED cheap control (not yet run): SFT answer-only on the SAME 1071 gated examples; if
  it matches ~0.93, the reasoning tokens were not the active ingredient (labeled subset sufficed);
  if it underperforms, the self-generated rationale text carries real signal.
- Forgetting (E0 yardstick): not separately measured for E1, but identical SFT recipe/scale =>
  expected comparably negligible.
Best artifact: outputs/sft_star_e1/weights/step_80. Data: data/blog_sft_star_e1_trunc (1071 rows).

#### E1 CONTROL — answer-only on the SAME gated rows BEATS the reasoning run (decisive)
The rubber-duck's causal control: SFT answer-only (strip <reason_why>, keep only <answer>LABEL)
on the IDENTICAL 1071 verifier-gated examples. If it matches the reasoning run, the rationale
text was not the active ingredient. Result (plain-prompt val/val_ood):
| ckpt | val acc | val_ood acc | notes |
|------|---------|-------------|-------|
| ans-only step40  | 0.976 | 0.981 | CHATGPT recall already 1.000 |
| ans-only step80  | **1.000** | **1.000** | ALL classes 1.0, pred_share 0.333 each |
| ans-only step200 | 1.000 | 1.000 | stable perfect |
vs reasoning-run BEST (step80): val 0.932 / val_ood 0.953.

This is the sharpest result of the ladder. CALIBRATED CLAIM (per rubber-duck): the supported
statement is narrow — "ON THIS task/data/eval, supervising SELF-GENERATED rationales hurts final
classification vs answer-only supervision on the same selected examples." NOT "reasoning is
inherently harmful" (gold-conditioned reasoning also hit 1.000 — the reasoning FORMAT is fine; the
issue is self-generated rationale CONTENT as a supervision target). Statistical note: 1.000 = 0
errors OBSERVED (val n=414, ood n=471), 95% rule-of-3 CI ~[0.991, 1.0]; the reasoning run 0.932
(386/414) CI ~[0.908, 0.956] does NOT overlap => the gap is real, not seed noise. Findings:
1. The self-generated reasoning was not merely unnecessary — as a supervision target it was net
   harmful here. On identical data, answer-only 1.000 > with-reasoning 0.953. Imitating the model's
   own (sometimes spurious) rationales injected noise / diluted the label gradient.
2. The damage was concentrated in CHATGPT exactly as predicted: reasoning-run CHATGPT recall
   0.80-0.89 (its kept rationales were the weakest — cheatsheet added only +60), but answer-only
   CHATGPT recall = 1.000. Stripping the noisy rationale text un-capped the hard class.
3. Answer-only on just 1071 SELF-gated rows MATCHES the gold-conditioned teacher SFT (1.000 on
   2892 gold-rationale rows). => neither the gold label-conditioning NOR the rationale text was the
   active ingredient; a small, clean, class-balanced labeled subset is sufficient and OOD-robust
   (1.000 on val_ood = unseen topics), reconfirming Parts V-VI: providers are ~perfectly separable
   by surface style.
4. Re-frames the whole RL saga (Parts VIII-X): RL optimizes a SPARSE signal through the REASONING
   channel for a task whose discriminative signal is a DENSE supervised label-mapping. The matched
   tool is answer-only supervised learning (1.000); the mismatched tools are reasoning-RL (collapse,
   0.40-0.66) and even reasoning-SFT (0.93, noise-capped). The pure-accuracy RL run (Part X) was
   directionally right (answer-only > reasoning) but RL's sparsity made it collapse; answer-only
   SFT gets the dense version of the same signal and wins cleanly.
Artifacts: outputs/sft_star_e1_ans/weights/step_80 (1.000/1.000, 1071 self-gated rows, no reasoning).
Caveat: "verifier-gated" still uses gold to SELECT the train subset; this is supervised learning on
a gold-selected, label-balanced set — not unsupervised. The novelty is that the SELECTION + labels
(not any rationale) carry all the signal.

#### OPSD ladder — STOP decision (E2-E4 pre-empted)
Decision (validated by rubber-duck): do NOT run E2 (OPCD), E3 (RLSD), E4 (ECHO) on this eval.
Rationale: E2-E4 are all REASONING-CHANNEL distillation methods requiring heavy custom-loss
engineering, but (a) the eval ceiling is already 1.000/1.000 via plain answer-only SFT (no
headroom to beat), and (b) the E1 control shows the reasoning channel is not the bottleneck and
self-rationale supervision is net-harmful here. Spending the engineering on sophisticated
reasoning-distillation against a saturated, reasoning-adverse task is low-value.
The genuinely open question — does sophisticated reasoning distillation help when the task is NOT
trivially separable — belongs on the HARDER eval (collaborator is building it), where there is
real headroom. Recommended single cheap strengthener if revisited on this eval: a rationale-token
loss-MASKING run (keep <reason_why> in the sequence but mask its loss, supervise only the answer
span) to disambiguate "noisy-rationale-supervision" from "reasoning-format/length conditioning."
LADDER STATUS: E0 done (forgetting yardstick), E1 done (+ decisive answer-only control). E2-E4
intentionally not run on the saturated eval; deferred to the harder eval.

---

## Model artifacts on the Hub (public, CK0607)
All checkpoints pushed to https://huggingface.co/CK0607and removed locally (disk cleanup):
- qwen3.5-9b-blogprovider-sft-goldcond — gold-conditioned SFT, val/ood 1.000
- qwen3.5-9b-blogprovider-selfgated-answeronly — answer-only on self-gated rows, 1.000 (headline)
- qwen3.5-9b-blogprovider-star-selfdistill — STaR self-distillation, 0.932/0.953
- qwen3.5-9b-blogprovider-rl-cheatsheet — best GRPO RL (cheatsheet), ~0.40 cheatsheet-free
- qwen3.5-9b-blogprovider-rl-entropydecay — entropy-decay ablation (negative)
- qwen3.5-9b-blogprovider-rl-pureacc-peak — pure-accuracy RL peak (0.66) before collapse
