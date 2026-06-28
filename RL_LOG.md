# Blog-Author-ID RL Experiment Log

3-way provider-classification GRPO RL in `prime-rl` on an 8×H100 node.
Model: **Qwen3.5-9B**, thinking OFF. Task: given a blog, classify the authoring
provider — **CLAUDE / CHATGPT / GEMINI** (the two Gemini variants collapsed into one).

> **North-star goal (user):** RL should make the model *emergently discover and
> articulate* what distinguishes each provider's writing style. The `<reason_why>`
> is the deliverable, not a perfect classifier. Reasoning must be learned via RL
> with NO rubric (rubrics bias the finding). Labeled-reward-only first; LLM-judge
> reasoning-reward is a deferred ablation.

Random baseline for balanced 3-way = **0.333**.

---

## Environment / infra facts (stable across runs)
- venv: `/home/ubuntu/blogger/prime-rl/.venv` (Python 3.12.13); run via `uv run`.
  `datasets` only importable via `uv run python`.
- Mandatory env: `HF_HUB_OFFLINE=1`, `WANDB_MODE=offline` (wandb.ai firewalled;
  background sync loop retries). `PRIME_RL_PRESERVE_DATA_ORDER=1` so on-disk row
  order == training steps (consecutive 12-row windows = one step).
- GPU split: **4 train (FSDP, HF impl) + 4 inference (vLLM tp=2 ⇒ auto dp=2, two
  replicas)**. Rebalanced from 6/2 after profiling showed the run was inference-bound.
- GRPO: `batch_size=96` rollouts/step ÷ `group_size=8` = **12 prompts/step**.
  Advantage is computed WITHIN each group (8 rollouts of ONE prompt, same gold).
  `zero_advantage` post-batch filter (enforced) drops groups where all rewards are
  equal — so all-correct AND all-wrong groups contribute no gradient.
- seq_len 16384, completion 4096, temp 1.0 (train+eval). Longest rendered prompt
  ~11.7k tok ⇒ fits with 4096 completion; 8192 completion would overflow long prompts.
- prime-rl writes TWO offline wandb runs: trainer (loss/kl/entropy) in `$OUT/wandb/`;
  orchestrator (reward/*, eval/blog-val/*, Trainable) in `$OUT/run_default/wandb/`.
  Sync MUST `find` BOTH dirs (early bug: only the trainer dir was synced).
- On-policy weight sync: after each step the trainer broadcasts a checkpoint +
  `STABLE` marker; the orchestrator pushes it to both vLLM replicas via
  `/update_weights` before the next rollout batch. (Separate from the on-disk
  best-val weights-only checkpoints kept by `scripts/ckpt_janitor.sh`.)
- **GOTCHA:** dataset-row `info` dicts must NOT use the key `"task"` — it is a
  reserved verifiers routing payload (`types.py: task_payload_from_info`); a plain
  string value crashes every rollout. Use `info["task_type"]`.

### Answer parser / reward (`deps/research-environments/.../blog_author_id.py`)
- Extracts the `<answer>` body; a body naming >1 provider = MISS (parse fail).
  `<reason_why>` is NOT parsed (reasoning may mention multiple providers freely).
- Reward = exact-match `correct_answer` (weight 1.0); `parsed_ok` diagnostic (weight 0).
- `SYSTEM_PROMPT_3WAY` has one added line asking for holistic qualitative judgment
  (no rubric). `SYSTEM_PROMPT_HARDPAIR` (2-way CLAUDE/CHATGPT) added for the
  hard-pair auxiliary task. Binary (CLAUDE/CHATGPT) prompt untouched.

---

## Dataset facts
- Base `data/blog_author_id_3way/{train,val}`: provider-balanced **137/137/137**
  (train 411), val = **held-out categories** (disjoint topics ⇒ generalization is real).
- **Task is trivially separable, but the model ignores the signal.** RandomForest on
  8 formatting features (em-dash, headers, bold, length, blank-lines) = **97.5% val**;
  TF-IDF on **stripped** text = **100% val**. Yet the base model sits near random.
- Strip augmentation: 40% of train rows have markdown/formatting stripped (em-dash →
  ", ", headers/bullets/bold removed, blank lines collapsed) to force deeper style
  signal over surface cues. Verified SAFE (stripped content stays ~100% separable;
  in-run, stripped acc ≥ formatted acc).

---

## Data generation — corpus evolution (9→21 categories, 4 models, short/long)

The blogs are SYNTHETIC, generated locally in `blogrl/` (`run_all.sh` + `topics.py` +
`generate_blogs.py`). Layout: `blogs/<category>/<topic>/<model>__<length>.md`. The corpus grew
along THREE axes over the project — providers, categories, and topics-per-category — each
expansion driven by a specific failure of the run before it.

**Providers (the label space).** 4 generator models, collapsed to a **3-way** provider task:
`claude-opus-4.8` (CLAUDE), `gpt-5.5` (CHATGPT), and two Gemini variants
`gemini-3.1-pro-preview` + `gemini-3.5-flash` (merged → GEMINI). Lengths per (category, topic,
model): **short** (~2500w / ~4k tok — fits the 16k RL window AND contrastive pairs), **long**
(~5500w / ~9k tok — single-text prompts), **epic** (8–12k w — OVERFLOWS the 16k window, NOT used).

**Axis 0 — the ORIGINAL 9 categories (binary HF `copilot-sdk-blogs`, 126 rows, claude/gpt only):**
`cognitive-science-neuroscience`, `cybersecurity-infrastructure`, `deep-learning-ai`,
`economics-systems`, `history`, `literature-cultural-theory`, `philosophy`, `politics`, `stem`.
These are heavily technical/analytic in register. Val held out the **3 hardest** (lowest base
pass@1): **history / politics / economics-systems**.

**Why generate a local corpus at all.** The 126-row HF set was too small to RL-train on (P0.1).
We needed thousands of balanced, verifiable rows ⇒ generate our own across many (category × topic
× model × length) combos, resuming from disk so each expansion only fills the new combinations.

**Axis 1 — providers expansion → fix the 4-way collapse.** Adding Gemini (pro + flash) gave a
4-way eval that was **degenerate**: the base model predicted GEMINI-PRO ~91% regardless of input
(P0.3). Motivation for the fix: collapse the two Gemini variants into one GEMINI and train on
**provider-BALANCED** data (random 0.333; a constant-class strategy scores only 0.333), so the
label-prior hack stops paying off.

**Axis 2 — categories 9 → 21 → fix the held-out-category generalization gap.** RL val stayed
capped because GEMINI style learned in the **technical-register** training categories did NOT
transfer to the **argumentative/narrative** val categories (history / politics / economics-systems).
So `topics.py` grew from **9 → 21 categories**, adding **12 humanities / social-science**
categories deliberately **register-matched to the val set**, all landing in TRAIN (verbatim from
`run_all.sh`: *"topics.py grew from 9 -> 21 categories … to fix the held-out-category
generalization gap … the new categories are register-matched to the val set and all land in
TRAIN"*). The 12 added: `sociology-anthropology`, `law-and-jurisprudence`, `art-architecture-design`,
`religion-and-theology`, `linguistics-and-language`, `education-and-pedagogy`,
`psychology-and-behavior`, `media-journalism-rhetoric`, `urban-environment-geography`,
`business-strategy-management`, `medicine-public-health`, `music-and-performing-arts`.
This is also what later enabled the **cross-register OOD val** (Part VI: separability that survives
a register change).

**Axis 3 — topics-per-category expansion.** Each category got a SECOND wave of topics appended to
`topics.py` (each category now lists two topic blocks), bringing the total to **~630 topics
(~30 / category)** for more within-category diversity (anti-memorization).

**Generation timeline** (`blogrl/logs/run_all_*.log`): 2026-06-10 (first large local gen),
2026-06-13, 2026-06-14 (the 9→21 diversity expansion). **Current scale: 21 categories, ~630
topics, 4 models × {short, long} = 5258 `.md` files.**

**Datasets derived from this corpus** (in `prime-rl/data/`, built by `build_blog_*.py`):
`blog_author_id_v2` (180/90 binary, frontmatter-stripped) → `blog_author_id_3way` (411 = 137³ /
204) → curriculum/contrastive/hardpair/balanced/pairwise/trio variants (Runs 2–15) →
**`blog_author_id_3way_v2`** (train 2682 / val 414 / **val_ood 471** = the cross-register held-out
set used by all the later SFT / cheatsheet / OPSD experiments). **Critical gotcha across every
build:** local `.md` files begin with YAML frontmatter containing `model: <author>` — ALWAYS
`strip_frontmatter()` before using as a prompt (the leak that faked val@0 = 0.906 in P0.2).

---

## Phase 0 — pre-3-way history (pass@4 baseline → binary RL → LR sweep → 4-way pass@k)


> Backfilled 2026-06-25 from session checkpoints 001–003 and the SQL `runs`/`passk_runs`
> tables. This is the work that PRECEDES Run 1: the original binary Claude-vs-ChatGPT
> pass@4 analysis, the first RL runs, the LR sweep, and the 4-way degeneracy finding that
> motivated the pivot to the provider-balanced 3-way task this log otherwise documents.

### P0.1 — FIRST pass@4 baseline (binary CLAUDE vs CHATGPT, `copilot-sdk-blogs`)
- Data: HF `copilot-sdk-blogs`, **126 rows**, balanced 63/63 (claude-opus-4.8 / gpt-5.5),
  9 categories ×14, ~3128 words avg. Harness `blog-eval/src/eval.py` (8-GPU DP pass@4 scaffold;
  max_tokens 16384, max_model_len 24576).
- **Result: pass@1 = 56.0%, pass@2 = 77.5%, pass@4 = 94.4%. 0% parse-fail, 0% truncation.**
- Per-row reward dist: **77.8% "mixed"** (1–3/4 correct ⇒ non-zero GRPO advantage), 16.7% solved,
  5.6% all-wrong ⇒ near-ideal RL target (large pass@1→pass@4 headroom, clean verifiable reward).
  Slight CLAUDE over-prediction. Caveat: 126 rows too small to train on directly.
- Split: **val = 3 hardest categories held out entirely** (history / politics / economics-systems),
  **train 84 / val 42**, balanced. Saved to `data/blog_author_id/{train,val}`.

### P0.2 — FIRST RL run + binary LR sweep (CLAUDE vs CHATGPT)
- Env: verifiers SingleTurnEnv, binary exact-match reward, SYSTEM_PROMPT with `<reason_why>`+
  `<answer>` (no `<think>`). Trainer fixes that carried forward: `impl="hf"`,
  `ac.mode="full"`, `fused_lm_head_token_chunk_size="auto"`, `max_model_len 16384`.
- **First real RL run** (lr 1e-6, 7 steps = 1 epoch, batch 96 / group 8 = 12 prompts/step):
  val pass@4 **0.452 → 0.558 mean / 0.625 peak**; train reward ~0.55; Error 0%, Truncation 0%.
  Overfit sanity (4-sample) 0.28 → 0.72.
- Eval-signal tuning: greedy val@0 0.610 vs noisy temp-1 mean 0.512, but greedy took 8m39s/step
  ⇒ reverted to **temp-1 parity eval every step**.
- **DATA-LEAKAGE bug** (user: "pass@1 is 90 wtf?"): local blog `.md` files have YAML frontmatter
  with `model: <author>` → builder leaked the label → fake val@0 = 0.906. Fixed with
  `strip_frontmatter()`; rebuilt clean **`data/blog_author_id_v2` (180 train / 90 val, 0 leaky)**.
  Memory stored.
- **Clean binary LR sweep** (v2 data; SQL `runs`): lr1e6 flat ~0.56; lr2e6 peak **0.642 @ step4**
  → 0.489; lr3e6 peak **0.700 @ step4** (+0.15, best) → collapse ~0.52; lr5e6 immediate collapse.
  Pattern: every LR peaks ~step 4 (⅓ epoch) then overshoots into class-collapse; higher LR =
  higher/earlier peak + faster collapse. **Sweet spot lr ≈ 2–3e-6 + early-stop ~step 4.**

### P0.3 — Multi-model pass@k (4-way degeneracy) → pivot to 3-way
- 4-way pass@k (`blog-eval/src/eval_multimodel.py`, 8-GPU DP, keep/strip markdown):
  **degenerate — predicts GEMINI-PRO ~91% regardless of input** (4404/4828 samples).
  4-way pass@1 ≈ 0.25 (= random), collapsed-3-provider ≈ 0.49. Markdown barely matters (<0.01).
  Per-model pass@1: gemini-pro 0.91, claude 0.05, gemini-flash 0.03, gpt 0.02. 0% parse-fail/trunc.
- This label-bias collapse motivated the pivot to a **provider-BALANCED 3-way** task
  (CLAUDE/CHATGPT/GEMINI, the two Gemini variants merged; random 0.333, "always GEMINI" only 0.333).
  Built `data/blog_author_id_3way` (411 train 137/137/137, 204 val 68/68/68; short+long;
  hard categories held out; body tokens ≤ 11800).

### P0.4 — 3-way pass@4 baseline (the RL starting point; SQL `passk_runs`)
- keep markdown: **pass@1 0.351 / pass@4 0.771**; strip: 0.360 / 0.769 (markdown not a tell).
- Per-model pass@1: claude 0.18 (under-predicted), gpt 0.44, gemini-pro 0.41, gemini-flash 0.47.
  Pred dist CLAUDE 328 / CHATGPT 1025 / GEMINI 1107 — **NOT collapsed** (unlike 4-way). 0 parse-fail/trunc.
- This balanced 3-way split (random 0.333, base pass@1 ~0.35) is exactly where **Run 1 below picks up.**

---

## Run history

### Run 1 — strict easy→hard curriculum  →  **COLLAPSED**
- Strict difficulty sort clustered same-difficulty prompts; the `n_correct=0` tail
  produced all-wrong (zero-advantage) batches ⇒ `Trainable` cratered to ~1% ⇒ policy
  reward-hacked to a single class (GEMINI). Val fell BELOW baseline (0.419→0.366).
- Also found+fixed the wandb dual-run sync bug (orchestrator reward curves had 0 syncs).

### Run 2 — interleaved 4/4/4 curriculum (lr 2e-6→1.5e-6, 4train/4infer)
- `build_blog_curriculum_3way_interleaved.py`: per-provider difficulty stratification ⇒
  every 12-prompt window is 4/4/4 provider-balanced with ≥7 trainable-middle prompts,
  gentle short→long drift. This removed the dead-batch collapse (`Trainable` 56–100%).
- Anti-collapse mechanism: balanced 4/4/4 windows remove the single-class-dump incentive
  (dumping scores 1/3). This is the PRIMARY fix, not LR.

### Run 3 — strip-augmentation ("aug"), labeled-reward-only
- `data/blog_author_id_3way_aug` (interleaved + 40% stripped). lr 1.5e-6, max_steps 30.
- **Eval (blog-val) reward FLAT/declining:** s0 0.425 → s4 0.419 → s8 0.376 → s12 0.355
  → s16 0.380 (killed ~s20). Below baseline.
- **Confusion matrix (train) revealed the real story — a zero-sum recall reshuffle:**
  | step | CLAUDE | CHATGPT | GEMINI |
  |---|---|---|---|
  | 1 | 25% | 55% | 50% |
  | 10 | 59% | 8% | 38% |
  | 18 | 69% | **19%** | 56% |
  CLAUDE recall rose, but CHATGPT **collapsed into CLAUDE**. Net accuracy flat ~0.40.
- Strip aug confirmed working (stripped acc ≥ formatted). No single-class collapse.
- Conclusion: CHATGPT-vs-CLAUDE is the genuinely hard pair (both conversational prose);
  correctness-only reward makes it locally optimal to max the separable classes and dump
  ambiguous prose into CLAUDE.

### Run 4 — hard-pair auxiliary task ("hardpair"), labeled-reward-only  ← most recent (COMPLETE)
- **Design (user chose "class-balanced reward + hard-pair oversampling"; refined via
  rubber-duck):** rubber-duck KILLED a cost-matrix reward (-0.5/0/1) — it makes all-wrong
  groups trainable (no longer zero-adv-filtered) and GRPO reinforces a "safe escape" to
  GEMINI; also confirmed per-class reward scaling is a GRPO no-op (cancels in (r-mean)/std).
  Adopted its cleaner alternative: an explicit **CLAUDE-vs-CHATGPT 2-way auxiliary task**
  interleaved into the curriculum, scored with the UNCHANGED plain binary reward.
  Restricting the label space forces the hard distinction; no escape; all-wrong groups
  stay filtered.
- **Implementation:** per-row `prompt` column ⇒ one env + one reward serves both task
  types. `build_blog_curriculum_hardpair.py`: 4 streams (CLAUDE3/CHATGPT3/GEMINI3/
  hardpair2); every 12-window = 3/3/3 three-way + 3 hard-pair; 40% strip; short→long drift.
  549 train (411 3-way + 138 hard-pair), gold C206/P206/G137. max_steps 40.
  `data/blog_author_id_hardpair`. (Bug: `info["task"]` reserved-key crash → renamed `task_type`.)
- **Eval reward (blog-val):** s0 0.444, s4 0.401, s8 0.413, s12 0.426, s16 0.409, s20 0.383,
  s24 0.369, s28 0.417, s32 0.404, s36 0.435, s40 0.406. **Still flat, ends below baseline.**
- **Eval per-class recall (held-out val, EXCLUDING truncated/empty):**
  | class | step 0 | step 40 |
  |---|---|---|
  | CLAUDE | 66% | **28%** |
  | CHATGPT | 62% | **83%** |
  | GEMINI | 15% | 13% |
  | macro | 47.5% | 41.4% |
- **Result:** hard-pair FIXED the CHATGPT collapse (62→83%) but **flipped the bias** —
  CLAUDE and GEMINI now dumped into CHATGPT. Macro-recall slightly DOWN. We traded one
  single-class bias for the opposite.

### Run 5 — strict per-step-balanced curriculum ("balanced"), stability-first  ← CURRENT (LAUNCHED 2026-06-11)
- **Why:** Runs 3–4 both showed a class-prior OSCILLATION, not learning. Per-step TRAIN
  analysis (this session) found the smoking gun: **per-step gold was imbalanced** (steps
  swung 4/4/4 → 4/3/5 → 6/3/3 …) and the prediction prior **chased the recent-majority
  gold class** (e.g. aug s8 gold CLAUDE-heavy → s12 predicts C60/P7/G28). GRPO normalizes
  advantage per-group, but the per-step GRADIENT averages over the 12 groups, so an
  imbalanced step tilts the gradient toward the majority class; momentum chases it. This
  is a **hackable degree of freedom**: the policy raises reward by shifting its global
  prior toward whatever recently dominated, instead of learning to discriminate.
- **Second cause (rubber-duck critique):** *active*-gradient imbalance. Only mixed groups
  (0<n_correct<8) train (zero-adv filter). Base-model difficulty is wildly asymmetric —
  CLAUDE nc{0:59,1:53,2:20,3:5} (43% always-wrong → filtered) vs CHATGPT/GEMINI much
  easier. So naive 4/4/4-over-all-difficulties still under-trains CLAUDE (less active mass).
- **Design (validated by rubber-duck):**
  1. **Strict 4/4/4 gold every step.** Every consecutive 12 on-disk rows = 1 step =
     exactly 4 CLAUDE + 4 CHATGPT + 4 GEMINI. Makes prior-shifting reward-NEUTRAL every
     step → the only way to gain reward is genuine per-prompt discrimination
     (strong, stable, non-hackable). Builder hard-fails if any step isn't 4/4/4.
  2. **Trainable-middle pool only (nc∈1..3).** Each class's 4 drawn from its middle-
     difficulty pool (CLAUDE 78 / CHATGPT 113 / GEMINI 122) so every group is expected
     mixed → balanced ACTIVE-gradient mass. nc0 (always-wrong→filtered dead weight) and
     nc4 (trivial→filtered) excluded.
  3. **Difficulty / length / strip matched across classes per step** (de-correlates the
     per-step gradient; no class is systematically easier/longer/stripped).
  4. **Dropped hardpair** (it broke balance & flipped the bias).
  5. **No format reward.** rubber-duck showed a "small" parsed_ok reward becomes
     full-strength in all-wrong groups whose only variance is parse success → reinforces
     arbitrary wrong-but-parsed answers (same failure class as the rejected cost-matrix).
     Kept parsed_ok at weight 0 (diagnostic).
  6. **Live per-class prediction-prior metrics → wandb.** Added weight-0
     `pred_{claude,chatgpt,gemini}` reward funcs; their batch means = the live prediction
     distribution, so prior drift is visible on the dashboard in real time.
  7. **Greedy eval** (temp 0, group 1): clean deterministic val-ACCURACY curve, ~4x
     cheaper (204 vs 816 rollouts), and avoids the temp-1.0 noise + empty-completion
     artifact. (pass@4 eval deferred to an offline run on the best checkpoint.)
- **Build:** `data/build_blog_curriculum_balanced.py` → `data/blog_author_id_balanced`
  (336 train = 28 steps × 12, strict 4/4/4, nc∈1..3, 40% strip; CLAUDE pool reused ~1.8x;
  val 204 held-out 3-way). Verified: every step 4/4/4, all nc∈1..3.
- **Config:** `rl_3way.toml` max_steps 28, eval temp 0 group 1 interval 4, wandb name
  `qwen3.5-9b-grpo-3way-balanced`. lr 1.5e-6, batch 96/group 8, temp 1.0 train,
  seq 16384/completion 4096, 4train/4infer. OUT `outputs/rl_3way_balanced`.
- **Status:** COMPLETE (killed at step ~22/28 once diagnosis was conclusive).
- **RESULT — partial success then HACKED.** Train reward climbed 0.33→0.625 with NO hard
  collapse (the strict-balance + trainable-middle fix worked for gold/active-gradient
  balance). BUT per-step TRAIN rollout analysis showed the gain was achieved by **acing
  CLAUDE+GEMINI and ABANDONING the confusable CHATGPT**:
    - TRAIN prediction prior C/P/G: s0 C23/P38/G34 → s8 C49/P20/G27 → s16 C40/P11/G45 →
      **s19 C46/P3/G46** (CHATGPT predictions nearly vanish).
    - TRAIN per-class recall C/P/G: s0 41/**52**/47 (base) → s8 66/28/44 → **s19 82/0/88**.
      CHATGPT recall falls from 52% (BASE, pre-RL) to 0%. Each step is strict 3-class
      balanced, so CLAUDE(✓)+GEMINI(✓)+CHATGPT(✗) = 8/12 = 0.67 reward → "rising reward"
      MASKS a class abandonment.
  - **Mechanism (root cause of the residual instability):** once CHATGPT recall→~0, every
    CHATGPT prompt is an all-wrong rollout group → GRPO **zero-advantage filter drops it** →
    NO corrective gradient flows back to fix CHATGPT → **absorbing collapse**. CHATGPT is
    the hardest (most confusable with CLAUDE); the model defaults CHATGPT→CLAUDE. Strict
    4/4/4 balances the GOLD but cannot stop this. **The BASE model is better-balanced than
    the RL'd model — RL actively destroys CHATGPT recall.**
  - **VAL** (greedy): acc 0.314 → **0.373 (s4 peak)** → 0.289 → 0.221 → 0.132, with
    `EmptyModelResponseError` exploding **18.6%→71.6%** (the "thinking-leak", below). Val is
    depressed/corrupted by the leak so its "best" (s4/s8) ≠ truly-best model.
  - **Thinking-leak confirmed precisely** (eval-only; TRAIN Error 0%): despite
    `enable_thinking=false` (template pre-fills empty `<think></think>`), Qwen3.5 RE-OPENS
    `<think>` mid-generation on hard/long blogs (one OK rollout had 7976 chars of reasoning
    before `</think>`). The orchestrator renderer `parse_qwen35` splits at `</think>`; if
    `<think>` is truncated before `</think>` (prompt ~11.7k + long think > 16k window) it
    returns content="" → verifiers raises `EmptyModelResponseError` → reward 0. Can't fix
    by raising completion budget (long prompt+think already approaches max_model_len). No
    reference-KL / entropy bonus / logit_bias available in prime-rl to anchor or suppress.
- **Watched on wandb:** `pred_*` (prior), train reward, greedy val accuracy.

---

### Run 6 — paired-CONTRASTIVE C-vs-CHATGPT + 3-way ("contrastive")  ← CURRENT (LAUNCHED 2026-06-11 ~17:07)
- **Why:** Run 5 proved strict-balance fixes gold/active balance but NOT the
  CHATGPT-abandonment collapse (an absorbing state from zero-advantage-filtering all-wrong
  hard-class groups). prime-rl has no reference-policy KL (the only KL is an inference-vs-
  trainer mismatch term, `kl_tau=1e-3`) and no entropy bonus; `teacher_logprobs` are only
  plumbed for `training_mode="opd"`, so a frozen-base KL anchor is not cheaply available.
- **Fix (structural, non-hackable) — validated by rubber-duck:** interleave a paired-
  CONTRASTIVE auxiliary task. Each step = **9 three-way rows (strict 3/3/3, trainable-middle,
  difficulty/length/strip matched)** + **3 pair rows** (each shows TWO texts A & B — one
  CLAUDE + one CHATGPT, random order — and asks the model to assign each). A contrastive
  pair makes abandonment IMPOSSIBLE: the authors differ, so a constant label is ≥50% wrong →
  the group stays MIXED (non-zero variance → survives the filter → gradient keeps flowing),
  and reward (plain binary, 1.0 iff BOTH slots correct) REQUIRES distinguishing the
  CLAUDE/CHATGPT boundary that collapsed. Serves the deliverable (articulate C-vs-P style).
  - rubber-duck caveats adopted: (a) **judge by per-class / macro / min-class recall, NOT
    train reward** (train reward is misleading here); (b) group_size↑ (8→12) is an
    exploration aid that keeps hard groups mixed *longer*, not the main fix; (c) do NOT
    patch `parse_qwen35` to emit truncated-think as content (would reward `<answer>` hidden
    inside `<think>` = eval-hackable); (d) suppress `<think>` via prompt (aligned with the
    `<reason_why>` deliverable).
- **Other changes:** explicit no-`<think>` prompt line in SYSTEM_PROMPT_3WAY + new
  SYSTEM_PROMPT_PAIR (thinking-leak mitigation); **lr 1.5e-6→1e-6** (slow drift away from
  the better-balanced base); **group 8→12, batch 96→144** (12 prompts/step); greedy eval,
  completion 4096, seq 16384, 4train/4infer unchanged. Val is pure 3-way (no pairs).
- **Env changes** (`blog_author_id.py`): added `SYSTEM_PROMPT_PAIR`, pair extractors
  (`_parse_pair`, `_parse_pair_gold`, `_completion_text`); `correct_answer` routes
  `info["task_type"]=="pair"` → both-slots-correct binary. Smoke-tested: pair/3-way reward
  routing all PASS, `info` flows to reward funcs.
- **Build:** `data/build_blog_curriculum_contrastive.py` → `data/blog_author_id_contrastive`
  (336 train = 28 steps × 12 = 252 three-way + 84 pair; pair A-slot gold balanced 42/42;
  pairs use SHORT blogs truncated to 14k chars so two fit the 16k window; 48% strip overall;
  per-row prompt column so env uses per-row messages verbatim). Verified every step 3/3/3 +
  3 C-vs-P pairs.
- **Config/launch:** `rl_3way.toml` data_dir `blog_author_id_contrastive`, wandb name
  `qwen3.5-9b-grpo-3way-contrastive`, OUT `outputs/rl_3way_contrastive`. RL pid 617715.
  Healthy: 4 trainer GPU @100%/75GB, 4 infer GPU (vLLM tp=2), zero_adv enforced, eval@0
  running, dual-wandb sync + best-val janitor up.
- **SUCCESS CRITERIA:** CHATGPT (min-class) recall STOPS falling to 0 and ideally rises;
  pred prior stays 3-way balanced; pair-task reward climbs above 0.5 (>random); macro recall
  climbs. ABORT if CHATGPT train recall hits ~0 for 2 evals (collapse not prevented).

- **RUN 6 EXECUTION (2026-06-11 ~18:00-19:30): a 3-bug saga, all fixed, then a janitor crash.**
  The pair reward read 0.00 at every step in the first launches. Root-caused THREE
  compounding bugs (each masked the next; all offline tests passed because the SAVED
  rollout differs from the runtime scoring object):
  1. Routing relied on `info["task_type"]`; switched to detect a pair by ANSWER format
     (`_is_pair_answer`: "A=..;B=..") which always reaches reward funcs.
  2. **qwen3 reasoning_parser** (auto-selected for Qwen3.5) routed the model's output into a
     reasoning channel and returned EMPTY `content` -> verifiers scored empty completions and
     eval threw `EmptyModelResponseError` (this was the Run-5 "no-answer" artifact too!).
     Fixed by disabling it in `packages/prime-rl-configs/.../utils/parsers.py`
     (REASONING_PARSER_PATTERNS Qwen3.5 line commented out -> resolver returns None). NOTE:
     `reasoning_parser="None"` in the TOML does NOT work — `rl.py` dumps the child inference
     config with `exclude_none=True`, so the child re-defaults to auto->qwen3. Must edit the
     resolver. **Eval Error% went 18-71% -> 0.0% after this fix.**
  3. **THE killer:** verifiers passes message **`AssistantMessage` OBJECTS** (not dicts) to
     reward funcs at scoring time but serializes to dicts in saved rollouts. The pair reader
     filtered `isinstance(m, dict)` -> dropped every object -> empty text -> pair reward 0,
     while every OFFLINE test on the saved rollout passed. Fixed `_completion_text` to read
     via `_msg_field` (dict.get OR getattr) + `_content_to_text` (str/list-part). VERIFIED:
     stored pair reward == recomputed, zero mismatch.
  (2)+(3) stored as user memories.

- **RUN 6 RESULTS through step 16 (then crashed — see below). HEALTHY & the design works:**
  - **No class abandonment** (the Run-5 failure mode is CURED): per-step train recall stayed
    C 0.28-0.64, **P(CHATGPT) 0.22-0.67** (never ->0), G 0.12-0.64; prediction PRIOR stayed
    BALANCED ~0.24-0.47 each class (Run 5 collapsed to C46/P3/G46). The contrastive pair
    keeps the C-vs-CHATGPT gradient flowing exactly as designed.
  - **Pair task learnable:** pair acc climbed w/ variance (0.17->0.56->...->0.88/0.96->0.58),
    peaks near 1.0 show the contrast IS separable.
  - **Val curve (greedy, Error 0.0%):** 0.245(s0)->0.260(s4)->**0.319(s8)**->0.250(s12)->
    0.299(s16). Modest & capped by ~60% val TRUNCATION (greedy decoding loops/over-reasons
    on long val blogs, never emitting `</answer>` within the 4096 completion budget ->
    auto-wrong). This truncation is the #1 thing to fix in the next run.

- **CRASH at step 16:** trainer died with `SafetensorError: I/O error: No such file or
  directory` writing `weights/step_16/model-00003-of-00004.safetensors`. Cause: the best-val
  `ckpt_janitor.sh` assumed "evaluated => fully written", but the eval for step N is logged
  BEFORE the trainer finishes exporting weights/step_N (eval runs on broadcast weights). The
  janitor deleted step_16 mid-write. FIXED with two race guards (scripts/ckpt_janitor.sh):
  only prune step_N if (a) a strictly-higher step dir exists AND (b) the dir wasn't modified
  in the last 3 min. Unit-tested. Relaunched fresh (weights_only ckpt = no optimizer state =
  no true resume).

- **RUN 6 FINAL (relaunched w/ fixed janitor; ran ALL 28 steps, 2026-06-11 ~19:29-20:14):**
  Completed cleanly — passed step 16 (prior crash point) with no SafetensorError; janitor
  pruned correctly (broadcasts stale + weights evaluated-not-best). Best-val ckpt kept.
  - **Val curve (greedy, Error 0.0% the WHOLE run):** 0.245(s0) -> 0.245(s4) -> 0.284(s8) ->
   0.275(s12) -> 0.270(s16) -> **0.324(s20)** -> 0.289(s24) -> **0.324(s28)**. +0.079 abs
   (+32% rel) over baseline. **Truncation fell 60.8% -> 19.1%** — the model learned to emit
   `<answer>` far sooner (the prior "no-answer" measurement cap is largely self-healing under
   this run; this is a real, valuable behavioral improvement).
  - **PARTIAL SUCCESS — collapse ROTATED, not eliminated.** The C-vs-CHATGPT contrastive pair
   DID cure the Run-5 CHATGPT collapse (CHATGPT train recall healthy late: 0.86-0.94). BUT a
   NEW failure emerged: **GEMINI collapsed.** GEMINI recall 0.39(s0) -> 0.08-0.17(s11-18) ->
   0.00-0.06(s24-27); predG -> 0.01-0.05 late. The global 3-way prior oscillated CLAUDE-heavy
   (predC 0.58-0.71, s11-18) then CHATGPT-heavy (predP 0.57-0.81, s19-27), squeezing GEMINI
   to ~0. Pair acc stayed strong/variable (peaks 0.88-1.00).
  - **ROOT CAUSE / LESSON:** the contrastive auxiliary only protects the ONE pair it covers
   (C-vs-CHATGPT). The third class (GEMINI) has no contrastive protection, and the global
   3-way prior is STILL unconstrained, so the abandoned class just rotated CHATGPT(Run5) ->
   GEMINI(Run6). The mechanism is proven but must cover ALL THREE classes.
  - **=> RUN 7 DESIGN:** replace the 2-way pair with a **3-way contrastive TRIPLE** auxiliary
   (one CLAUDE + one CHATGPT + one GEMINI in a single prompt; assign each label exactly once).
   A constant label is 2/3 wrong -> group stays MIXED -> survives zero-adv filter -> restoring
   gradient for ALL three -> no class can be abandoned. Directly extends the proven Run-6
   mechanism to protect GEMINI too.
   - **NOTE (design pivot before launch):** a rubber-duck critique flagged that the TRIPLE has
     a fixed-WRONG-permutation zero-advantage hole (a deterministic permutation policy makes a
     group uniform -> filtered -> no gradient) plus 1/6-baseline noise. It recommended instead
     SYMMETRIC pairwise coverage of ALL boundaries at ~50% contrastive. ADOPTED for Run 7 (see
     below): a 2-text different-author pair is ALWAYS exactly 50% wrong for any constant/
     position prior, so its group ALWAYS stays mixed -> guaranteed restoring gradient on that
     boundary every step. Lower-risk than the triple and reuses the proven pair reward.

---

### Run 7 — SYMMETRIC pairwise-contrastive coverage ("pairwise")  ← COMPLETED 2026-06-11 ~20:30-21:18
- **Design:** every step = 6 single 3-way (2/2/2) + 6 contrastive pairs
  (2x CLAUDE-vs-CHATGPT + 2x CLAUDE-vs-GEMINI + 2x CHATGPT-vs-GEMINI), ~50% contrastive. Each
  pair restricts the label space to its two providers (no third-label escape) and is scored
  with the UNCHANGED plain-binary pair reward (1.0 iff BOTH slots correct). Covering all three
  boundaries gives every class a guaranteed restoring gradient every step.
- **Impl:** env `make_pair_system_prompt(p1,p2)` (parameterized; reward path label-agnostic,
  already matches GEMINI — smoke-tested all boundaries object+dict+flipped+dup-invalid);
  `data/build_blog_curriculum_pairwise.py` -> `data/blog_author_id_pairwise` (336 train,
  gold 56/56/56, pair exposure 112/112/112, A-slot balanced, 54% strip). Config data_dir
  ->pairwise, wandb `qwen3.5-9b-grpo-3way-pairwise`, OUT `outputs/rl_3way_pairwise`. Ran all
  28 steps cleanly (no janitor crash); best-val = step_28; Trainable stayed ~100% (all groups
  mixed, as designed). eval Error 0.0% throughout.
- **RESULT — primary objective ACHIEVED; deeper issue exposed:**
  - ✅ **GEMINI collapse FIXED.** GEMINI is now the most STABLE class: single-text recall
    0.44-0.79 late, predG steady 0.37-0.61 (vs Run 6 where GEMINI hit 0.00 / predG 0.01 by
    step 27). Symmetric coverage works.
  - ✅ **All-boundary discrimination preserved** (serves the "what's distinct" goal): late
    pair acc CvP 0.75-1.00, CvG 0.72-1.00, PvG variable 0.08-0.92. The model demonstrably
    distinguishes every provider pair.
  - ⚠️ **Oscillation ROTATED again -> CHATGPT now under-predicted on SINGLE-TEXT.** Single
    CHATGPT recall oscillates 0.00-0.33 late; predP pinned low 0.05-0.16. NOT a hard absorbing
    collapse (the CvP/PvG pairs keep CHATGPT's gradient alive, recall bounces off 0), but the
    single-text marginal prior is miscalibrated against CHATGPT.
  - ➖ **Val ~flat:** 0.299(s0) -> 0.265(s8) -> 0.299(s16) -> 0.250(s20) -> **0.314(s28)**.
    Truncation stayed ~53-61% (did NOT self-heal like Run 6's 60->19%).
- **ROOT-CAUSE CRYSTALLIZED:** the unconstrained quantity is the **single-text MARGINAL
  prediction prior**. Contrastive pairs guarantee per-boundary DISCRIMINATION (and prevent
  absorbing collapse) but do NOT constrain the single-text marginal -> with no KL/entropy in
  prime-rl's loss, the favored single-text class still drifts (rotated C5:CHATGPT ->
  R6:GEMINI -> R7:CHATGPT). Pairs fix *can-the-model-tell-them-apart*; they don't fix
  *what-it-predicts-when-shown-one-text*. The next lever must target the marginal directly
  (loss-level entropy/KL, OR an LLM-judge reasoning reward, OR shifting eval/inference to use
  the contrastive ability) — a strategic fork worth the user's call.

### Run 7 DELIVERABLE — learned per-provider STYLE FINGERPRINTS (<reason_why>, correct rollouts, steps>=14)
(Harvested via /tmp/harvest_reasons.py from contrastive + single correct completions.)
- **CLAUDE** (n=204 correct, ~91% HIGH-conf): academic "lecture-room voice"; rhetorical
  framing ("strength of X"); philosophical precision blended with accessible analogies;
  complex, self-referential transition sentences; structured "worked example" pedagogy.
- **CHATGPT** (n=52 single correct — low, mirrors the under-prediction; rich in pairs, CvP
  n=239): the most distinctive learned tell is literal **"Worked Example X" headers** (35/52);
  Markdown tables / taxonomies; LaTeX equation blocks; "Risk assessment"-style lists;
  "One [noun] is X" sentence frames; "In practice" transition markers; early formal notation.
- **GEMINI** (n=195 correct): grand/ornamental **metaphors** as the signature ("Rosetta
  Stone", "dark matter of intelligence", "toddler exploring the world", "holy grail");
  analogy-driven intros & conclusions; numbered lists + dense technical exposition;
  idiosyncratic section hooks ("Modern computing is built on a contradiction").
- Contrastive boundaries surfaced relative discriminators (e.g. CHATGPT=bold lists + early
  math + "In practice" vs CLAUDE=denser rhetorical prose; GEMINI=named subheadings + metaphor
  hooks vs the others). This is the emergent "what makes them distinct" output the user wants.

---

## Current diagnosis (the two distinct problems)

1. **Class-prior oscillation = the real TRAINING instability.** Correctness-only GRPO on
   this hard task leaves the global prediction prior unconstrained: each prompt-group
   independently rewards its own gold, so nothing keeps the prior balanced/calibrated. The
   favored class drifts with whatever pressure we add (aug → over-predict CLAUDE/under
   CHATGPT; hardpair → over-predict CHATGPT). The reasoning is not grounding the label.
   Demonstrated across TWO labeled-only runs.

2. **Eval no-answer = a MEASUREMENT/config artifact (not training instability).**
   - TRAIN completions are SHORT and stable: median 162→215 tokens over 40 steps, 0 missing
     `<answer>`. Training rollouts are clean.
   - EVAL step 40: 205/816 are 0-char completions ("returned reasoning but no content"),
     73 truncated, 270 total missing `<answer>`. Error% rose 10%→26% over the run. This is
     the `reasoning_parser="qwen3"` + harder/longer val blogs triggering a long reasoning
     channel that never emits content. It depresses the measured val reward and masks the
     true per-class trend (hence we analyze recall on COMPLETED rollouts only).

3. **GEMINI does not generalize to val.** Even the BASE model gets only ~15% recall on
   held-out GEMINI (mostly predicted CHATGPT). A genuine generalization gap (val =
   disjoint categories), separate from optimization.

---

## Decisions / preferences captured
- Keep the old 3-way prompt + one qualitative-judgment line; NO rubric (rubrics bias the
  learned reasoning, which is the deliverable).
- Labeled-reward-only runs first; LLM-judge reasoning reward is a deferred ablation.
- **(latest) Exhaust reward design for STABLE RL before moving to the LLM judge.**
- 4train/4infer maximizes throughput (inference-bound). Don't add a 2nd eval env (doubles
  eval); run stripped-val ablation offline at the end.

## Next steps
- [ ] **Reward design for stability (current focus):** target the class-prior oscillation
      (e.g. valid-`<answer>` format reward to stop no-answer drift — GRPO-safe; explore
      confidence-calibration carefully re: GRPO escape route flagged by rubber-duck).
- [ ] **Fix the eval no-answer artifact** (raise eval completion budget and/or revisit the
      qwen3 reasoning parser with thinking OFF) so val numbers are trustworthy.
- [ ] Investigate the GEMINI val-generalization gap (data diversity?).
- [ ] Harvest `<reason_why>` style fingerprints per provider (use info["task_type"],
      info["stripped"]).
- [ ] Deferred: LLM-judge reasoning-reward ablation.

## Key paths
- Config: `examples/blog_author_id/rl_3way.toml`
- Env/prompts/parser/reward: `deps/research-environments/environments/blog_author_id/blog_author_id.py`
- Builders: `data/build_blog_curriculum_3way_interleaved.py`,
  `data/build_blog_3way_aug.py`, `data/build_blog_curriculum_hardpair.py`
- Launch: `run_3way_curriculum.sh` (sets env, launches RL + dual-wandb sync + ckpt_janitor)
- Outputs: `outputs/rl_3way_interleaved/`, `outputs/rl_3way_aug/`, `outputs/rl_3way_hardpair/`
  (each `run_default/rollouts/step_*/` has train + eval rollout jsonls)

---
## RUN 8 — "entropy" (detached-surprisal entropy-bonus PG)  [LAUNCHED 2026-06-12 16:21]
Output: outputs/rl_3way_entropy. wandb: qwen3.5-9b-grpo-3way-entropy. SAME pairwise data as
Run 7 (isolate the entropy variable). max_steps 28, lr 1e-6, group 12 / batch 144, seq 16384.

WHY: Runs 5/6/7 root cause = the single-text class MARGINAL prediction prior is unconstrained
(prime-rl GRPO has no reference KL, no entropy term) so the favored class drifts and the
abandoned class rotates (C5:CHATGPT -> R6:GEMINI -> R7:CHATGPT-under-predicted). Contrastive
pairs fixed discrimination + hard collapse but NOT the marginal.

FIX (rubber-duck-validated): a custom loss fn `surprisal_entropy_loss_fn` in
src/prime_rl/trainer/rl/loss.py = default DPPO+KL PLUS an entropy-bonus PG term. Per-token
surprisal under the rollout policy s_t=-inference_logprob (detached, clamped >=0) is folded
into the advantage:  shaped_adv = adv_tau*adv + entropy_coef*s_t, applied on keep_mask (DPPO
trust region respected). Minimizing -pg_loss raises trainer prob of high-surprisal (rare)
tokens -> raises entropy AND up-weights the rare answer-label token of an under-predicted
class -> keeps it sampled & receiving corrective gradient. entropy_coef=0.02.
- Fused-LM-head compatible: uses ONLY sampled-token logprobs (no entropy backward, no teacher).
- NOT the degenerate +beta*logprob form (that has zero on-policy expected grad & pushes the
  sampled token DOWN). Verified numerically: coef=0 == default_loss_fn exactly; coef>0 lowers
  loss when trainer raises prob of a rare token. New wandb metrics: surprisal, entropy_bonus.
- TOML: [trainer.loss] type="custom" import_path=...surprisal_entropy_loss_fn kwargs={...}.

STARTUP VERIFIED healthy: eval@0 Reward 0.2255 (base parity), steps 0-4 Error 0.0%,
Trainable 83-100%, Max Off-Policy 0, no NaN; custom loss runs (trainer completes steps).
SUCCESS CRITERIA: single-text CHATGPT predP recovers off ~0.05 and all 3 class shares stay
balanced WITHOUT tanking GEMINI/CLAUDE; macro recall climbs; val > 0.314. ABORT if min-class
share keeps falling despite rising entropy, or instability (NaN/KL blowup).

### RUN 8 RESULT (COMPLETED 2026-06-12 ~17:05, all 28 steps, Error 0.0% throughout)
- **VAL curve (greedy):** 0.226(s0) -> 0.201(s4) -> 0.255(s8) -> 0.201(s12) -> 0.230(s16) ->
  **0.466(s20, BEST)** -> 0.387(s24) -> 0.270(s28). Best-val ckpt = step_20.
  **0.466 is the BEST val of ANY run** (Run 6 best 0.324, Run 7 best 0.314; base 0.226).
  The s20 spike coincided with TRUNCATION collapsing 43.6%->0.5% (model emitted <answer>
  concisely instead of over-reasoning into truncation) — same self-healing as Run 6 but
  stronger. Volatile though: truncation rebounded to 36% and val fell back to 0.270 by s28.
- ✅ **Primary design goal MET — CHATGPT single-text under-prediction FIXED.** predP (CHATGPT
  single-text predict share) stayed 0.17-0.58 the WHOLE run (vs Run 7 where it pinned at
  0.05-0.16). The surprisal/entropy bonus up-weighted the rare under-predicted-class label
  token exactly as designed, keeping all three classes sampled. The marginal was MORE balanced
  than any prior run through ~step 23.
- ⚠️ **Late drift + volatility (not a clean win).** At the very end the prior tilted CLAUDE-heavy
  (s27 predC 0.64 / predP 0.17 / predG 0.19; GEMINI recall dipped to 0.17). So entropy raises
  the floor on the abandoned class but does NOT fully pin the marginal — consistent with the
  rubber-duck caveat that token-entropy is a WEAK proxy for the cross-prompt marginal. Val is
  non-monotonic; would benefit from LR decay / early-stop on best-val (step_20 kept).
- ✅ Discrimination preserved: per-boundary pair acc late CvP/CvG/PvG mostly 0.3-0.7 (peaks 1.0).
- **VERDICT:** entropy bonus is the best lever so far on BOTH axes that mattered (best val +
  marginal balance), but volatile. Next: LR-decay/early-stop, or pair it with a gentle
  batch-level marginal regularizer; LLM-judge reasoning reward still the deferred ablation.

---
## Runs 9/10 — DIVERSE-DATA ABLATION (entropy on/off), single var = entropy
Motivation: Run 8 val ceiling (0.466) traced to held-out GEMINI recall ~4% — training was
dominated by technical categories; val (history/politics/economics) is a different register.
FIX: regenerated a larger, humanities-weighted dataset (21 categories, 2742 blogs) and rebuilt
both the 3-way split (train 1491 / val 204) and pairwise curriculum.

Pipeline changes:
- build_blog_curriculum_pairwise.py: deal_class now samples UNIFORMLY over the trainable-middle
  pool (was round-robin-by-nc, which under-sampled nc=2 where ALL new data defaults — starving
  the single-text stream of the new register). N_STEPS 28 -> 56. OUT -> data/blog_author_id_diverse.
  Built: 672 train (336 single 2/2/2 + 336 pairs), 18 categories in singles (humanities-dominant),
  val 204 unchanged. Per-step 2/2/2 + 2x each boundary verified.
- Configs: rl_3way_div_default.toml (default loss) and rl_3way_div_entropy.toml (entropy_coef=0.02),
  both max_steps=56, data=blog_author_id_diverse, batch 144/group 12.
- run_3way_curriculum.sh now takes (config, out) args.

Run order: DEFAULT FIRST (if it stays stable, that's a finding; if it collapses, entropy justified).
GOTCHA fixed: an initial regex strip of [trainer.loss] also ate the [orchestrator] block ->
batch fell back to 128/group_size=1 (GRPO disabled, Trainable 0.8%). Recreated config cleanly;
relaunch verified batch_size=144/group_size=12, Trainable 75-92%.

### RUN 10 (default loss, outputs/rl_3way_div_default, wandb ...-div-default) — LAUNCHED
eval@0 val 0.24 (base parity), Error 0.0%, Trainable 75-92%. 56 steps. WATCHING.
Comparison metric: completed-only accuracy + per-class recall (truncation-robust; val truncation
~56% from reasoning-first format) + predict-share. SUCCESS = GEMINI recall climbs off ~4%,
shares balanced, best val > 0.466.

---
## RUNS 9/10 RESULT + ROOT-CAUSE PIVOT (supersedes the category-generalization theory)
Both div-data arms FAILED to revive held-out GEMINI on the CATEGORY-holdout val:
- Run 10 (default loss, 56 steps): GEMINI val completed-recall ~0.00 throughout; train reward
  ->0.83 while val flat ~0.26 (marginal collapse, oscillates C<->P, never G).
- Run 9 (entropy_coef=0.02): KILLED early. GEMINI alive only in temp-1 TRAIN sampling, NOT in
  greedy val; truncation WORSENED 62->72%, compAcc 0.51->0.33. Entropy = wrong mechanism
  (buys verbose rare tokens, not greedy-argmax calibration). Confirmed rubber-duck failure-mode.

### DEEP DIAGNOSIS (the real blockers):
1. TRUNCATION was a GREEDY-DECODING REPETITION LOOP. Inspecting truncated GEMINI val completions:
   the model never emits <reason_why>; it free-form deliberates ("The user wants me to identify...")
   and loops ("...CLAUDE. Wait... CLAUDE...") for the full 4096 tokens, never emitting <answer>
   -> truncated, reward 0, parse fail. TRAIN (temp 1.0) truncates 0%; VAL (temp 0.0 greedy) 56-78%.
   => greedy repetition pathology, worst on long GEMINI blogs. FIX: val temp 0.0 -> 0.7 (top_p 0.95).
   Result: val truncation 78% -> ~33%.
2. CATEGORY-GENERALIZATION THEORY DISPROVEN. Built an IN-DISTRIBUTION topic-holdout val (hold out
   15% of TOPICS per category, all 3 provider variants -> val; 3 OOD categories kept as offline
   val_ood). GEMINI recall STILL ~0.04 in-distribution => NOT a register/category gap.
3. THE REAL BOTTLENECK = GEMINI<->CHATGPT STYLISTIC CONFUSION. Pair (2-way) accuracy by boundary:
   CvP 0.62->0.92 (easy, learned), but CvG ~0.5 and PvG 0.17->0.46 (Gemini-vs-ChatGPT barely
   learnable). Val GEMINI blogs (BOTH gemini-pro AND gemini-flash) are predicted CHATGPT, not G.
   The model perceives Gemini's prose as ChatGPT's. In 3-way singles GEMINI is rarely the argmax
   (temp1 train predG 0.31, but temp0.7 val predG 0.03 — high mass, low argmax).

### RUN 11 (topic-holdout val + non-greedy val 0.7 + brevity prompt + default loss) — RUNNING
outputs/rl_3way_topic_default, wandb ...-topic-default. Data rebuilt: train 1290 (430x3, 18 cats),
val 201 (67x3, topic-holdout), val_ood 204. Prompt hardened (must start <reason_why>, <=4 sentences,
no deliberation/looping, always emit <answer>). Early: val trunc ~33% (was 78%), but GEMINI val
recall still ~0.04 @ step8. WATCHING whether PvG pair acc climbing (0.17->0.46) lifts val GEMINI
over 56 steps. If not -> Run 12 with increased ChatGPT-vs-Gemini contrastive weight.

### RUN 11 RESULT (default loss, topic-holdout + non-greedy val) — COMPLETE
56 steps. Truncation FIX confirmed: val truncation 38% -> ~7-13% (no more greedy loop). BUT the
single-text marginal COLLAPSED TO CLAUDE: val predshare C 0.50->0.96 by step20, completed recall
C=1.00 / P~0.04 / G~0.04 throughout. compAcc peaked 0.432 @ step16 then sat 0.33-0.36. GEMINI
never lifted off ~0.04. Clean confirmation: default loss + symmetric pairs still collapses; the
pairs keep PAIRWISE gradient alive but do NOT fix the single-text MARGINAL (no KL/entropy anchor).

### *** DECISIVE DIAGNOSTIC: the hard boundary is PERFECTLY SEPARABLE (label-only) ***
Ran a supervised label-only probe (/tmp/sep_probe.py) on the SAME data + topic-holdout val to
settle "representational ceiling vs optimization failure". Train a trivial classifier on train,
eval on the held-out val:
  2-way (chance 0.50):  CvP / CvG / CvG-content / PvG ALL = 1.000 val acc (TF-IDF on STRIPPED text,
    formatting removed). Even a formatting-only RandomForest: CvP 1.00, CvG 0.985, PvG 0.970.
  3-way content (stripped, chance 0.33): 1.000 val acc, recall 1.00/1.00/1.00, predshare 0.33/0.33/0.33.
=> CHATGPT-vs-GEMINI is PERFECTLY linearly separable from content alone. NOT a representational
   ceiling, NOT a data problem. The discriminative signal is massive and clean. The model's failure
   is PURELY an RL-optimization pathology: the zero-advantage filter starves the hard/abandoned
   class of gradient (all-wrong groups get filtered) and the unconstrained marginal collapses.
   This RULES OUT "accept the ceiling / generate more data" and points squarely at an anti-collapse
   optimization lever (entropy) + concentrating contrastive pressure on the underlearned boundary.

### RUN 12 (two-arm ablation, both build on the truncation fix + topic-holdout + new 1290-row data)
Both arms add the entropy bonus (entropy_coef=0.02 — the Run-8 anti-collapse mechanism, retested
now that the greedy-val truncation confound is removed). Ablate the CURRICULUM:
  - Run 12B "balanced+entropy": balanced 2/2/2 pairs (data blog_author_id_diverse). Clean retest of
    the Run-8 winning mechanism. Config rl_3way_div_entropy.toml, wandb ...-balanced-entropy.
  - Run 12A "pvg+entropy": pairs reweighted 1 CvP / 2 CvG / 3 PvG (data blog_author_id_pvg) to pour
    contrastive gradient onto the clean-but-underlearned CHATGPT-vs-GEMINI boundary while keeping
    CLAUDE anchored (milder skew per rubber-duck, vs 1/1/4, to avoid collapse-rotation). Singles stay
    2/2/2. Config rl_3way_pvg_entropy.toml, wandb ...-pvg-entropy.
Predefined SUCCESS criteria (rubber-duck): val GEMINI recall materially > Run 11 (~0.04); PvG pair
acc > 0.55-0.60; no class predshare > 0.70; truncation stays low; best-val survives >1 eval point.

### RUN 12B RESULT (balanced 2/2/2 pairs + entropy_coef 0.02 + non-greedy val) — best val 0.371 @ step28
*** First run where GEMINI stays ALIVE on val and no class collapses. *** Trajectory (val, "all" recall
incl. truncated-as-wrong | predshare C/P/G):
  s0  C0.37/P0.19/G0.03 | 0.54/0.42/0.04   (base)
  s16 C0.42/P0.42/G0.12 | 0.40/0.43/0.18   trunc 12%
  s20 C0.34/P0.46/G0.16 | 0.32/0.51/0.18   trunc 4.5%  <- most balanced marginal of any run
  s28 C0.43/P0.49/G0.18 | 0.37/0.44/0.19   trunc 1.7%  val reward 0.371 (BEST)
Entropy kept all three classes sampled AND (with the truncation fix) GEMINI climbed 0.04->~0.18-0.20
(5x every prior run) while the marginal stayed balanced (no predshare >0.52). compAcc ~0.34-0.44.
  *** LATE-PHASE ENTROPY INSTABILITY (known Run-8 volatility): *** after ~step32 the entropy bonus
  progressively inflated verbose/looping generation -> truncation EXPLODED (s32 1.2% -> s40 47% ->
  s44 88%), val reward crashed 0.371->0.03, steps slowed to 6min (every rollout maxes 4096 tok).
  KILLED at step51 (blown up). Best-val janitor kept weights/step_28. LESSON: entropy_coef 0.02 is
  productive for ~30 steps then destabilizes; cap horizon / decay entropy / early-stop on best-val.
  => Run 12A capped at 40 steps (best-val captured; avoids the garbage tail).

### RUN 12A RESULT (PvG-weighted pairs 1/2/3 + entropy 0.02 + non-greedy val) — best val 0.398 @ step16 *** BEST ON TOPIC-HOLDOUT ***
Pouring contrastive gradient onto the hard CHATGPT-vs-GEMINI boundary (3 PvG + 2 CvG + 1 CvP/step)
lifted GEMINI FASTER and HIGHER than balanced 12B. Val ("all" recall C/P/G | predshare | reward):
  s8  0.43/0.29/0.06 | 0.51/0.42/0.07 | 0.311
  s12 0.58/0.19/0.21 | 0.55/0.23/0.23 | 0.331   GEMINI alive by step12 (vs 12B step16)
  s16 0.66/0.27/0.27 | 0.56/0.22/0.22 | 0.398   BEST val; GEMINI recall 0.27 (vs 12B's ~0.18)
  s24 0.45/0.33/0.25 | 0.43/0.34/0.23 | 0.353   marginal REBALANCED, GEMINI holding 0.25
Truncation 38%->~2-4%. CRASHED at step25 on the known wandb-service ConnectionResetError (infra, not
training) — but the peak + rebalancing trend were captured; best-val janitor kept weights/step_16 (+20).
Entropy would have destabilized after ~32 anyway (see 12B), so little lost. GPUs auto-freed on crash.

### RUN 12 VERDICT (2-arm ablation, both >> Run 11's CLAUDE-collapse baseline 0.37/G0.04):
  - 12B balanced+entropy: best val 0.371, BALANCED marginal (C0.43/P0.49/G0.18), GEMINI ~0.18.
  - 12A pvg+entropy:       best val 0.398 (BEST), GEMINI recall 0.27 (highest), C-heavy at peak then
                           rebalances to C0.45/P0.33/G0.25. PvG-weighting wins on GEMINI + peak val.
KEY RESULT: entropy (anti-collapse) + non-greedy val (truncation fix) + boundary-targeted contrastive
pressure together keep ALL THREE classes alive and lift the abandoned GEMINI class from ~0.04 to
~0.25-0.27 recall on a clean topic-holdout val — confirming (with the perfect-separability probe) that
the bottleneck was OPTIMIZATION (marginal collapse), not data or a representational ceiling.
SHARED CAVEAT: entropy_coef 0.02 is late-phase unstable (verbosity blowup after ~step32). Future:
entropy decay / early-stop on best-val / cap horizon ~30-36 steps.

### OOD STRESS EVAL (best ckpt = Run 12A step16) on the 3 held-out OOD categories (history/politics/economics-systems)
val_ood (408 rollouts, temp0.7): overall acc 0.338, recall C/P/G 0.61/0.25/0.15, predshare 0.56/0.23/0.19,
truncation 1.2%. vs in-distribution topic-holdout (0.398, recall 0.66/0.27/0.27): a ~0.06 acc gap and
GEMINI recall 0.27->0.15 OOD — but ALL THREE CLASSES STAY ALIVE (no collapse) and the model stays concise
on unseen categories. The RL'd 3-way discrimination generalizes; GEMINI is the weakest-but-alive class OOD.
(Loading the weights-only ckpt in vLLM needed preprocessor_config.json/merges.txt/vocab.json copied from
the base snapshot — weights_only save omits the multimodal preprocessor configs Qwen3.5 probes for.)

================================================================================
### RUN 13 — MODERN GRPO-VARIANT ABLATION (MaxRL vs CISPO) — both NEGATIVE, but decisive
================================================================================
Motivation: survey of 2024-26 RL-for-reasoning methods (REINFORCE/PPO/GRPO/RLOO/Dr.GRPO/DAPO/
CISPO/DPPO/MaxRL/ScaleRL, ref aweers.de/blog/2026/rl-for-llms). Two were the best-matched to our
pathology (hard-class GEMINI starvation + the original pass@k framing), so we ran a 2-arm ablation,
each a SINGLE-VARIABLE change vs the Run-12A entropy incumbent (same PvG curriculum, non-greedy val,
batch144/group12, LR1e-6, max_steps40). NB prime-rl's DEFAULT loss is already DPPO (Divergence-PPO:
TV trust region on prob-difference) and its advantage is already Dr.GRPO (r-mean, no std-norm).

Implementations (unit-tested, math verified):
  - maxrl_advantage (src/prime_rl/orchestrator/advantage.py): custom ADVANTAGE. A_i ∝ (r_i-r_hat)/r_hat,
    r_hat=K/N, K=#successes; zero when K=0; adv_clip=6 then RE-CENTERED to keep group zero-mean
    (rubber-duck caught that clipping alone biased rare-success groups). Default DPPO loss, no entropy.
  - cispo_loss_fn (src/prime_rl/trainer/rl/loss.py): custom LOSS. Clip IS weight (upper eps_high=0.28,
    lower inactive) + stop-grad, NO token masking -> gradient flows on ALL tokens (incl. the rare-class
    tokens DPPO drops). Kept Kimi KL (kl_tau=1e-3). Default GRPO advantage. Grad direction/sign verified.

ARM A — MaxRL  (outputs/rl_3way_pvg_maxrl): best val 0.311@s16 (vs incumbent 0.398).
  Per-class val (predshare C/P/G): s0 .44/.54/.02 -> s16 .38/.58/.04. GEMINI predshare nudged up but
  recall(all) stuck ~0.03. *** TRUNCATION STUCK 32-39% ALL STEPS *** (incumbent dropped 38%->2.5% by s16).
  MECHANISM: MaxRL DOWN-weights high-success (easy) prompts -> base model never consolidates concise
  <answer> format -> ~37% of val truncated -> reward ceiling ~0.31. The dense GRPO signal it discards is
  exactly what teaches brevity here. Stopped @s18 (ceiling confirmed).

ARM B — CISPO (outputs/rl_3way_pvg_cispo): best val 0.291@s4, declined to 0.241@s16.
  *** COLLAPSED to CHATGPT *** (predshare P .44->.76, CLAUDE recall .37->.10), truncation rose 38->43%.
  *** "Max Off-Policy 0" on ALL 21 steps *** => the async loop is NEAR-ON-POLICY => importance ratio ~1
  => CISPO's clip/stop-grad NEVER fires => CISPO degenerates to the plain default loss => marginal
  collapse, identical in kind to the Run-11 default baseline. GEMINI stayed dead (~0.04). Stopped @s18.

### RUN 13 VERDICT — the decisive, generalizable conclusion:
  1. TRUST-REGION variants (CISPO, DAPO clip-higher, DPPO) only differ from the default loss OFF-policy.
     prime-rl async is 1-step (Max Off-Policy 0), i.e. effectively on-policy, so they are NEAR-NO-OPS here.
     => Don't expect clip/mask tweaks to help this task; the data confirmed it (CISPO == default collapse).
  2. ADVANTAGE-RESHAPING (MaxRL) DOES change behavior, but de-emphasizing easy/high-success prompts
     starves the brevity/format learning the base model still needs -> truncation never falls -> capped.
  3. The EFFECTIVE lever for THIS task is anti-collapse pressure on the unconstrained class MARGINAL +
     breaking the repetition/truncation loop. Empirically ONLY the entropy-bonus run achieved the
     truncation collapse (38%->2.5%) AND kept all three classes alive -> it remains the best recipe (0.398).
  => Best path to higher numbers is NOT a different trust-region/advantage estimator, but pushing the
     entropy/marginal lever (entropy decay or DAPO overlong soft-penalty to stabilize the late blowup).

================================================================================
### RUN 14 — REASON-GATED REWARD + TRUNCATION-PENALTY ADVANTAGE (launched)
================================================================================
Trigger: user asked to "run the overlong thing." Investigation BEFORE launching found the plain
overlong/length penalty is the WRONG lever here:
  - Truncation is ALREADY solved by the entropy recipe (val trunc 38%@s0 -> 2.5%@s16). An overlong
    penalty optimizes a non-problem.
  - The REAL late-phase failure: the reward is ANSWER-ONLY, so over-training collapses the
    deliverable. By step24, ~11% of CORRECT val rollouts have an EMPTY/stub or missing
    <reason_why> (e.g. out_tok=29, "<reason_why></reason_why><answer>CHATGPT</answer>" still
    scoring 1.0). A length penalty would PUSH HARDER toward this empty-reason hack.
  - Rubber-duck also flagged: a recentered length penalty leaks POSITIVE advantage onto
    short-but-wrong rollouts (escape hatch) and de-filters all-wrong groups.
Decision (user picked): reason-gate the reward + mild TRUNCATION-ONLY penalty (not general length).

Changes vs entropy incumbent (rl_3way_pvg_entropy, val 0.398):
  1. REWARD GATE (deps/.../blog_author_id.py): correct_answer now returns 1.0 only if the answer is
     correct AND the completion has a substantive <reason_why> (closed tag, >=12 DISTINCT alphabetic
     words -> enforces a length floor AND blocks repetition-padding, no LLM judge). MULTIPLICATIVE
     gate: wrong answers still 0, so all-wrong groups stay zero-advantage-filtered (sidesteps the
     env's documented additive-format-reward hazard). Applied to BOTH train and val (best-val janitor
     now selects for the true objective = correct+reasoned). Added weight-0 diagnostic reason_ok
     (batch mean = live deliverable health). Validated on real rollouts: gate KEEPS 96.9% of step16
     legit correct answers, ZEROS the late-phase empty-reason hacks (step24 11.3%).
  2. ADVANTAGE (advantage.py:truncation_penalty_advantage): default Dr.GRPO (r-mean) PLUS an additive
     -0.5 on is_truncated rollouts ONLY, NON-recentered. Unit-tested: in all-wrong groups only the
     truncated go negative (others stay 0 -> no wrong-answer reinforcement); correct stay highest;
     truncated pushed below other wrong. NOT a general brevity penalty (that's owned by the reason gate).
  Loss UNCHANGED (surprisal_entropy_loss_fn, entropy_coef=0.02) — entropy bonus stays the anti-collapse lever.
Config: examples/blog_author_id/rl_3way_pvg_reasongate.toml (parses via real loader). max_steps=40.
Watch: val (reason-gated) vs 0.398; reason_ok/mean (should stay high, NOT decay late); GEMINI recall;
truncation rate. NB reason-gated val ~= raw_val*reason_ok so expect ~0.386 at an equivalent step16.

================================================================================
RUN 14 RESULT (reason-gate + truncation-penalty) — rl_3way_pvg_reasongate
================================================================================
Full reason-gated val trajectory (eval every 4 steps, max_steps 40):
  s0 0.2363(tr35.8%) s4 0.2289 s8 0.2090(tr37%) s12 0.2985 s16 0.3408(tr7.7%)
  s20 0.3532(tr2.7%) s24 0.3433 s28 0.3607(tr1.5%, PEAK) | s32 0.3035 s36 0.1766(tr34%) s40 0.0174(tr91.5%)
TWO WINS:
  (1) DELIVERABLE PROTECTED: reason_ok -> 1.00 and held. The multiplicative reason-gate closed the
      empty-<reason_why> answer-only hack non-hackably (all-wrong groups stay zero-advantage). The
      gated peak (0.3607) is a HONEST correct+reasoned number, not answer-only.
  (2) TRUNCATION SOLVED through the peak (35.8% -> 1.5% by s28).
TWO FAILURES (both motivate Run 15):
  (A) MARGINAL DRIFT capped the peak: across s16->s28 the global 3-way prediction prior drifted
      CHATGPT-ward (CLAUDE recall 0.54->0.19, CHATGPT 0.31->0.66). The entropy bonus kept all 3
      classes ALIVE but does NOT BALANCE them -> peak stuck ~0.36 (below the 0.398 entropy incumbent).
  (B) CATASTROPHIC LATE COLLAPSE: past the peak the UNDECAYED entropy bonus (coef 0.02 to the end)
      eventually drove the model BACK into the high-surprisal repetition loop: s28->s40 val
      0.3607->0.0174, truncation 1.5%->91.5%. Best-val janitor saved the peak ckpt, but the run is
      unusable past ~s30.
=> Two clean, separable levers identified: (A) needs a MARGINAL-balancing signal; (B) needs the
   entropy bonus to DECAY. Run 15 ablates each in isolation off this same Run-14 base.

================================================================================
RUN 15 — ABLATION SERIES (each single-variable off the Run-14 reason-gate base)
================================================================================
15A ENTROPY-DECAY (rl_3way_pvg_entdecay) — targets failure (B).
  Loss surprisal_entropy_loss_fn -> surprisal_entropy_decay_loss_fn: entropy_coef HOLDS at 0.02
  through hold_frac=0.4 (step16, = where truncation was already solved) then LINEARLY decays to
  floor 0.0 by end_frac=0.8 (step32). Step/max_steps plumbed train.py -> compute_loss -> rl loss fn
  (forwarded only to custom losses that declare step/max_steps via inspect; default/cispo untouched;
  unit-tested: == plain entropy at step0, == coef-0 at floor). Hypothesis: removes the s32-40
  repetition-loop collapse; the anti-collapse entropy is only needed early to escape the loop, not
  late once concise answering is established. LAUNCHED; early val 0.234->0.328 by s12 (tracks Run14).
15B TRIOWISE (rl_3way_trio) — targets failure (A), the marginal drift.
  New env task: 3 texts A/B/C, each authored by one of {C,P,G}, SAME provider may repeat. Reward =
  FRACTION of slots correct (per-slot accuracy), multiplicatively reason-gated. Per-slot (NOT
  all-or-nothing) per a rubber-duck critique: all-or-nothing exact-match is SPARSEST exactly on the
  under-predicted class (its gold triples are least likely to ever produce a fully-correct rollout),
  giving NO gradient where drift is worst -> drift self-reinforces. Per-slot keeps the group MIXED
  (rollouts vary in #correct) so a restoring gradient flows even before any rollout nails all three,
  and stays non-hackable (its only within-group variance axis IS partial correctness; for
  uniform-marginal trios no constant/escape strategy beats E=1/3 per slot). Composition i.i.d.
  uniform per slot (~67% two-of-one, ~21% 3-distinct, ~12% 3-same; gold marginal uniform) -> honors
  the "no 3-distinct bias, 2+1 must appear" requirement AND makes constant-class a losing strategy.
  Data per step: 6 single 3-way (2/2/2, val-matching + difficulty spread so no step starves) + 6
  trios. Env: make_trio_system_prompt, _parse_trio/_parse_trio_gold/_is_trio_answer (checked BEFORE
  pair since trio gold also contains A=/B=; C= disambiguates), slot-accuracy reward path. Builder
  data/build_blog_curriculum_trio.py -> data/blog_author_id_trio (672 rows, verified composition +
  uniform marginal). All unit-tested via the real env rubric. QUEUED after 15A (shares all 8 GPUs).
(15C adaptive-temp / 15D low-weight class-balance: deferred — orchestrator temp control is the async
 pipelined seam (invasive); 15A+15B directly target the two known failure modes and are non-invasive.)

================================================================================
RUN 15 RESULTS (ablation series; full eval trajectories, gated val, eval every 4)
================================================================================
15A ENTROPY-DECAY (rl_3way_pvg_entdecay):
  s0 0.234 s4 0.229 s8 0.251 s12 0.328 s16 0.356 s20 0.366 s24 0.376(PEAK) s28 0.269
  s32 0.271 s36 0.373 s40 0.341 | truncation NEVER exceeds 2.7% the whole run.
  => Late catastrophic collapse ELIMINATED (Run14 s40 0.017/tr91.5% -> 15A s40 0.341/tr2.7%).
     Peak also UP (0.376 vs Run14 0.361). Late phase wobbles 0.27-0.38 = the marginal drift
     (failure A), not the loop. CONFIRMS: the late collapse was the UNDECAYED entropy bonus.
15B TRIOWISE (rl_3way_trio, per-slot-accuracy reward, PLAIN entropy):
  s0 0.209 s4 0.221 s8 0.231 s12 0.313 s16 0.408(PEAK, BEST OF ALL RUNS) s20 0.363 s24 0.333
  s28 0.353 s32 0.311 s36 0.132(tr24%) s40 0.005(tr84%).
  Per-class @ peak s16: recall C0.619/P0.358/G0.261, macro 0.413 (best); predshare CLAUDE-skew
  (179/117/95) -> marginal raised+rebalanced but not perfectly uniform; GEMINI still hardest.
  Live-verified: trio slot-accuracy reward yields MIXED groups (rewards 0/.33/.67/1.0), 6+6
  task composition holds from step2, some all-3-correct (1.0) by step1.
  => Trio RAISES THE PEAK (marginal pressure works) but, with undecayed entropy, collapses late
     EXACTLY like Run14 (entropy ballooned to 10.5 @ s37). Trio does NOT fix the collapse.
DECISIVE: 15A (decay) and 15B (trio) fix ORTHOGONAL failures (late-stability vs peak). => combine.
15E TRIO + ENTROPY-DECAY (rl_3way_trio_entdecay): trio data + surprisal_entropy_decay_loss_fn,
  else identical to 15B. Hypothesis: 0.408-class peak held stable to the end like 15A. LAUNCHED
  (eval@0 0.204 parity). Best ckpt of this run = the project deliverable; harvest fingerprints from it.

15E RESULT (trio + entropy-decay):
  s0 0.204 s4 0.214 s8 0.211 s12 0.281 s16 0.294 s20 0.4005(PEAK) s24 0.328 s28 0.316
  s32 0.249(tr16%) s36 0.142(tr32%) s40 0.164(tr20%). Truncation: 40%->4% by peak, low thru s28.
  Per-class @ peak s20: recall C0.336/P0.410/G0.455, macro 0.400; predshare C0.289/P0.348/G0.323
  = NEAR-UNIFORM marginal (best balance of any run), GEMINI recall 0.455 (best ever; was the
  hardest class ~0.26). trunc 4%.
  => Combined recipe DELIVERS on the peak (0.4005 ~= 15B 0.408, within run-noise) AND the marginal
     (uniform predshare, all 3 classes alive, GEMINI no longer the laggard). Late phase: NOT the
     catastrophic collapse (15B s40 0.005/tr84%) but a MILDER re-degradation once the entropy floor
     hits 0 at s32 (tr creeps 16->32->20%, reward to ~0.16). Trio's per-slot pressure appears to
     need a small entropy floor late that pvg did not (15A pvg+decay stayed <2.7%). The best ckpt
     (step_20) is preserved and is the project DELIVERABLE.
  Fingerprints harvested -> files/style_fingerprints_run15e.txt (CLAUDE=essayistic/argumentative;
  CHATGPT=worked-examples/numbered/LaTeX; GEMINI=ASCII-diagrams/notation/confident-narrative);
  first harvest from a balanced, non-collapsed checkpoint.
FINAL RUN-15 VERDICT: best DELIVERABLE = 15E step_20 (peak 0.4005, macro recall 0.400, uniform
  marginal, GEMINI alive 0.455, trunc 4%). Trio data = the marginal fix; entropy-decay = the
  collapse fix; together they give the best balance at a peak tied with the best single-arm.
  Open thread: a small nonzero entropy floor (instead of ->0) would likely also flatten 15E's
  late tail the way it did for pvg.

================================================================================
SEPARABILITY DIAGNOSTIC (2026-06-14, pre-data-gen) — DECISIVE, OVERTURNS DATA-CEILING HYPOTHESIS
Question: is the ~0.40 RL ceiling a DATA/signal limit (=> generate more) or a MODEL/RL limit?
Method: train classical classifiers on blog_author_id_3way/train, test on /val. ZERO (cat,topic)
overlap train(234) vs val(36) => pure generalization to unseen topics, NO leakage.
RESULTS:
  - TF-IDF word 1-2gram + LogReg, train->val:           acc 1.000 (perfect, all 3 classes)
  - char 3-5gram:                                        acc 1.000
  - first 1500 chars only (well inside LLM window):      acc 1.000  (truncation NOT the cause)
  - STYLOMETRIC ONLY (func-word + punct freq, 0 content):acc 0.940  (it's STYLE, not topic)
  - RL'd Qwen3.5-9B raw classification @ best ckpt s20:  acc 0.400  (reason_ok 0.975 => gate is
                                                          NOT the limiter; base s0 was 0.284)
Top LogReg style features (= genuine tells, NOT artifacts):
  CHATGPT: may, may be, for example, such as, not only, also, depends on, especially (hedge/enum)
  CLAUDE:  you, genuinely, precisely, honest, genuine, worth, exactly, "and it" (2nd-person/sincere)
  GEMINI:  furthermore, we must, profound, mathematically, paradigm, fundamentally, highly (grandiose)
CONCLUSION: the 3 providers are ~perfectly, generalizably separable by TRIVIAL surface style. The
data is NOT the bottleneck. The ~0.40 RL ceiling is a MODEL/RL-exploitation gap (60+ pts left on
the table that a linear model captures). => MORE DATA OF THE SAME KIND WILL NOT RAISE THE NUMBER.
Real levers (all NON-data): (1) SFT/distillation warmup on the gold labels (features are so
learnable a tiny SFT should jump raw acc to ~0.8+), (2) few-shot/fingerprint-cheatsheet in prompt,
(3) thinking-ON to let the model do explicit feature-checking. Scalar-reward RL from scratch is a
far weaker teacher than the supervised logreg that nails 100%.
Legit DATA directions exist but they're about DIFFICULTY/SCOPE, not quantity: (a) HARDER/style-
controlled/adversarial data so the benchmark is meaningful (currently a logreg solves it), (b) more
VAL categories for robustness. Neither is "more of the same." -> asked user how to proceed.

ROLLOUT EXPLOITATION-GAP DEEP-DIVE (15E step20 vs logreg, per-val-example, aligned by example_id):
  logreg acc 1.000 vs LLM acc 0.443 (14 = NONE/unparsed). LOGREG-right-but-LLM-wrong = 112/201 (56%).
  STRATIFIED BY LOGREG CONFIDENCE (= example "easiness"):
    logreg hi-conf >0.9 (n=175, textbook-obvious style): LLM acc only 0.474
    logreg mid-conf .6-.9 (n=26):                        LLM acc 0.231
  => The LLM fails even where the style signal is TRIVIALLY strong (logreg ~certain). This is genuine
     FEATURE-BLINDNESS / non-exploitation, NOT edge-case hardness. RL nudged raw acc 0.28->0.44 but the
     model is essentially not reading the style tells a linear model nails. CONFIRMS the lever is
     supervised injection of the signal (SFT/distillation warmup), not more RL or more/harder data.

================================================================================
FINAL DATAGEN COMPLETE (2026-06-15) + AT-SCALE SEPARABILITY CONFIRMATION
Datagen: topics.py doubled 15->30/category (630 topics). run_all.sh short+long, CONCURRENCY=24,
~5h. Corpus now 5258 blogs (was 2742), balanced ~630/model/length, provider-gaps:0. Only 3 long
gens failed (transient CAPI timeouts in education-and-pedagogy; negligible). Removed 3 stale empty
__epic.md (epic bucket unused by RL). Quality clean: all short+long >=12.5k chars.
Rebuilt split -> data/blog_author_id_3way_v2 (fresh dir; did NOT clobber the 15E split):
  train 2682 (894/class), val 414 (138/class, 72 held-out topics / 18 cats),
  val_ood 471 (157/class; history/politics/economics held out ENTIRELY = cross-register).
AT-SCALE SEPARABILITY (LogReg train->held-out, bigger + more diverse than before):
  WORD 1-2gram:           val 1.000   val_ood 1.000  (PERFECT even cross-REGISTER / unseen categories)
  FUNC-WORDS+PUNCT only:  val 0.988   val_ood 0.943  (content-free style alone ~perfect, cross-register too)
=> Confirms & STRENGTHENS the diagnostic at 2x scale: providers are perfectly, generalizably
   separable by surface style across BOTH unseen topics AND unseen categories. Also retires the old
   "GEMINI style doesn't transfer across register" hypothesis from runs 12-13: at the linear level it
   transfers at 100%. The entire gap to the RL'd 9B's 0.40 is MODEL-SIDE EXPLOITATION. More data is
   confirmed NOT the lever; the data is now abundant + clean for an SFT-warmup approach.

=== SFT WARMUP (Design A) — SOLVED ===
Qwen3.5-9B, impl=hf, thinking OFF, body<=3000tok/seq4096, 270 steps, lr1e-5 cosine.
Plain-prompt eval (no cheatsheet), val + val_ood (cross-register):
  base 0.454/0.410 -> cheatsheet 0.674/0.592 -> SFT step90 0.906/0.909
  -> SFT step180 1.000/1.000 -> step270 1.000/1.000 (all classes recall 1.0).
GEMINI is last class learned (0.73 @ step90 -> 1.0 @ step180); CHATGPT 0.19->1.0.
No leakage (prefix+substring checked). Genuine grounded style reasoning emitted.
Best artifact: outputs/sft_warmup/weights/step_180. RL-polish dropped (no headroom).

=== FINAL RL RUN (Design B) — cheatsheet-elicited GRPO, no judge ===
qwen3.5-9b-grpo-3way-trio-cheatsheet | trio data | entropy-decay loss + trunc-penalty adv |
init BASE | train-ONLY style cheatsheet injected into train+eval system prompt | 40 steps, lr1e-6, seq16384.
Purpose: give feature-blind base a non-collapsed reward floor so GRPO has real variance.
Gated val reward curve (reason-gated + truncation-penalized):
  step0 0.443 -> 12 0.517 -> 16 0.614 -> 20 0.632 -> 24 0.634 -> 28 0.662(PEAK) -> 32 0.649 -> 40 0.627
  eval truncation collapsed 24.1% -> ~0% (model learned to be decisive, stop runaway gen).
  Trainable groups 100% throughout (healthy variance, no class collapse).
Best ckpt = step_28. Raw accuracy (temp0.7, plain reward off):
  WITH cheatsheet:    val 0.659 / val_ood 0.650  (balanced; GEMINI strongest, CHATGPT weakest)
  WITHOUT cheatsheet: val 0.348 / val_ood 0.378  => NOT internalized; collapses to ~base.
Honest read: RL did NOT create internalized discrimination. It polished USE of the elicited
  features: killed truncation, reliably passed the reason-gate, and improved OOD robustness
  (val_ood 0.592->0.650 vs base+cheatsheet). Raw val acc ~flat vs base+cheatsheet (0.674->0.659).
  => Confirms thesis: cheatsheet ELICITS, RL POLISHES format/robustness, only SFT INTERNALIZES (1.000).
Kept ckpts: outputs/rl_3way_trio_cheat/weights/step_{28,40}. wandb run 6bac14a7. STOP (one run, per user).

=== PROBE: prior 0.40 RL ckpt + cheatsheet @ eval (answers "could old run just use cheatsheet?") ===
outputs/rl_3way_trio_entdecay/step_40 (trained NO cheatsheet), eval WITH cheatsheet:
  val 0.324 / val_ood 0.304  -- WORSE than untrained base+cheatsheet (0.674/0.592).
  COLLAPSED: predicts GEMINI ~87% (CLAUDE/CHATGPT recall ~0-2%), NONE 10-18% (ignores cheatsheet).
=> The "0.40 plateau" was actually single-class COLLAPSE. Cheatsheet cannot rescue post-hoc;
   must be present DURING training (gives GRPO non-degenerate floor, 100% trainable groups).
   probe_results/prior040_with_cheat.json

=== PURE-ACCURACY / NO-REASONING RL (Part X) — answer-only GRPO, BASE init, no cheatsheet ===
Config: v2 uniform data, answer_only (completion = single <answer>LABEL</answer>, decode_len~9.6),
require_reason=false, max_completion_tokens=32, G=12, batch144, lr1e-6, temp1.0/eval0.7,
surprisal_entropy_decay beta0.02(hold0.4/end0.8), trunc_penalty adv, eval val+ood @interval4.
Zero leakage (exact/prefix/substr=0); balanced classes. config: examples/blog_author_id/rl_3way_v2_pure_acc.toml
Eval (cheatsheet-FREE acc):
  step0 val0.348/ood0.375  step4 0.383/0.404  step8 0.384/0.414
  step12 0.580/0.572  step16 0.650/0.662 (PEAK)  step~20 ABORT (10 consecutive zero-trainable batches)
Confusion (step16): CLAUDE recall 0.000 (270/276 -> CHATGPT), CHATGPT 0.967-1.000, GEMINI 0.982-0.987.
=> Learned a near-perfect CHATGPT-vs-GEMINI classifier; absorbed ALL CLAUDE into CHATGPT. 0.66 = the
   2-class ceiling (2/3). Deterministic policy -> uniform-reward groups -> zero advantage -> abort.
   Collapse occurred during the entropy HOLD phase (beta full) -> single-token bonus too weak.
READING: answer-only RL DOES clear the prior ~0.40 cheatsheet-free ceiling via real 2-way signal
   extraction (directionally supports M1/M2 reason-token dilution), but does NOT reach a stable 3-way
   solution -- collapses to an absorbing 2-class optimum (drops the hard CLAUDE/CHATGPT boundary).
   0.66 is a transient degenerate peak, not a reportable 3-way accuracy.
Artifact: outputs/rl_3way_pure_acc/run_default/broadcasts/step_16 (peak weights, collapsed). wandb 60d51475.
Next controls (cheap): label-rotation (A/B/C) to separate content-vs-token bias; binary CLAUDE-vs-rest;
   then anti-collapse 3-way (larger G + higher train temp + lower lr + entropy floor + early-stop) if warranted.

================================================================================
OPSD LADDER — E0 (forgetting baseline) + E1 (STaR self-distillation)
================================================================================
E0 (eval-only, scripts/eval_e0_forgetting.py): BASE vs sft_warmup/step_180.
  General-text perplexity 2.779 -> 2.866 (+3.2%); 6/6 capability probes intact.
  => gold-conditioned SFT-to-1.000 cost ~zero general capability. Yardstick for E1-E4.

E1 (STaR on-policy self-distillation; gold NEVER shown in generation, only verifier-gates):
  Gen (scripts/gen_star_e1.py): plain k=3 maj-gate + cheatsheet-hint k=2 gate on still-wrong,
  reject hinted rationales citing the cheatsheet. 1913/2682 accepted; class-balanced cap=357
  => 1071 SFT rows. Accepted-by-source asymmetry: CLAUDE plain162/hint518, GEMINI 363/513,
  CHATGPT 297/60 (cheatsheet barely helps CHATGPT).
  SFT (sft_star_e1.toml, BASE init, seq4096, lr1e-5, 200 steps, ckpt/40). PLAIN-prompt eval:
    step40 0.577/0.533 -> step80 0.932/0.953 (BEST) -> step120 0.891/0.879 -> step160 0.928/0.909
    -> step200 0.896/0.883.  CHATGPT recall caps 0.80-0.89 (residual gap source); CLAUDE+GEMINI ~1.0.
  => self-generated verifier-gated reasoning internalizes ~93-95% of the task without any gold-
  conditioned teacher. Gap to 1.000 is exactly the class self-reasoning can't articulate (CHATGPT).
  Recommended cheap control (pending): answer-only SFT on same 1071 gated rows (isolates whether
  reasoning text vs labeled subset drives the gain). Artifact: outputs/sft_star_e1/weights/step_80.

E1 CONTROL (answer-only SFT on the SAME 1071 verifier-gated rows; reasoning stripped):
  step40 0.976/0.981 -> step80 1.000/1.000 (all classes, balanced) -> step200 1.000/1.000.
  => BEATS the reasoning run (0.932/0.953) on identical data. Self-generated rationales were
  ACTIVELY HARMFUL (noise), worst on CHATGPT (reasoning recall 0.80-0.89 -> answer-only 1.000).
  Answer-only on 1071 self-gated rows == gold-conditioned SFT (1.000 on 2892 gold rows): the
  rationale text and the gold-conditioning were BOTH non-essential; a small clean class-balanced
  labeled subset suffices and is OOD-robust. Confirms: this task's signal is a DENSE supervised
  label-mapping; RL-through-reasoning (sparse) is the mismatched tool. Artifact step_80 (1.000/1.000).

================================================================================
E2/E3 — OPCD + RLSD on-policy (self-)distillation: CODE BUILT, AWAIT LAUNCH
================================================================================
Status (2026-06-25): faithful experiment code built + validated WITHOUT GPUs; NOT launched
(held for explicit go). Honest framing: E1/STaR above is verifier-gated hard-label SFT (the
loose cousin), NOT a teacher-KL on-policy objective — so true OPD/OPSD had never been run.
E2/E3 close that gap as NATIVE prime-rl training modes (not a standalone script).

E2 — OPCD (On-Policy Context Distillation): student rolls out PLAIN; teacher = SAME policy with
  the train-derived CHEATSHEET spliced into the system prompt, scores those rollouts. Loss =
  detached per-token teacher/student logprob GAP as PG signal; DPPO trust region keyed to the
  distillation direction (sign of teacher gap), not a verifier advantage. (opcd_loss_fn)
E3 — RLSD (verifier-anchored on-policy self-distillation): per-token weight (P_T/P_S)^sign(A),
  PPO/CISPO-clipped on the student ratio. Verifier gives DIRECTION (reinforce correct rollouts,
  suppress incorrect), cheatsheet teacher gives MAGNITUDE; A==0 contributes nothing. (rlsd_loss_fn)

Wiring: training_mode={"opcd","rlsd"} threaded through transport/config/dispatcher/orchestrator;
  teacher-fetch gate uses TEACHER_LOGPROB_MODES; orchestrator splices the cheatsheet system-prefix
  into the teacher prompt and realigns the returned logprobs to student sample length. Teacher =
  the LIVE student inference pool (same weights, weight-broadcast each step) — most faithful
  same-weights self-distillation and the only option with 4-train/4-infer on 8xH100. Port 8300.
Init from BASE Qwen3.5-9B (feature-blind ~0.40) so the cheatsheet teacher (~0.66) is genuinely
  better => non-degenerate signal (init from the 1.000 SFT would be uninformative by construction).
Zero eval leakage: student rollouts + eval stay PLAIN; cheatsheet reaches the model ONLY via
  teacher scoring.
CAVEAT (in configs too): sampled-token, reverse-KL-flavoured (logprob gap on sampled tokens),
  NOT full-vocabulary forward KL D_KL(p_T||p_S). Not claimed as forward-KL.

Validation (no GPUs): opcd/rlsd loss fns registered + backprop-checked; splice + logprob-realign
  unit tested (5/5 pass); splice identity cheat_prefix+plain_tail==cheat_prompt verified
  token-for-token on 15 probe blogs (plain 399 -> cheat 621 tok, +222 cheatsheet tokens);
  data/cheatsheet_splice_3way.json written; config validator now REQUIRES both teacher AND
  cheatsheet_splice_path for opcd/rlsd (else teacher==student, empty signal); ruff clean.
Pre-flight gate (run first, on go): scripts/diag_teacher_student.py — teacher-vs-student accuracy
  / gold-label prob / label-space KL on val to confirm headroom before any 8xH100 run.
Files: src/prime_rl/trainer/rl/loss.py (opcd_loss_fn, rlsd_loss_fn, setup_loss_fns);
  src/prime_rl/orchestrator/{utils.py,orchestrator.py}; src/prime_rl/transport/types.py;
  packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py; scripts/build_cheatsheet_splice.py;
  scripts/diag_teacher_student.py; examples/blog_author_id/rl_3way_{opcd,rlsd}.toml;
  tests/unit/orchestrator/test_teacher_logprobs.py.

================================================================================
E2 — OPCD RESULT (RUN COMPLETE, 2026-06-25): transient gain -> truncation collapse
================================================================================
Launch: rl @ examples/blog_author_id/rl_3way_opcd.toml --output-dir outputs/opcd_e2
  Qwen3.5-9B from BASE, 4 train (FSDP hf) + 4 infer (vLLM tp=2 dp=2), port 8300,
  teacher=student pool w/ cheatsheet spliced in, max_steps=40, lr=1e-6, eval PLAIN every 4.
Pre-flight gate (diag_teacher_student.py, n=60 val): teacher_acc 0.300 > student 0.250,
  teacher gold_p 0.377 vs 0.284, label-KL 0.096, frac teacher>student 0.767 => GO (non-degenerate).

VAL trajectory (PLAIN eval, group_size=2, 414 val):
  Step  0  reward 0.2222  trunc 38.3%
  Step  4  reward 0.2367  trunc 38.2%
  Step  8  reward 0.2597  trunc 35.7%
  Step 12  reward 0.2911  trunc 25.4%   <- PEAK (+0.069 abs / +31% rel over step 0)
  Step 16  reward 0.2428  trunc 29.6%
  Step 20  reward 0.1353  trunc 45.9%
  Step 24  reward 0.0713  trunc 69.2%
  Step 28  reward 0.0543  trunc 65.9%
  Step 32  reward 0.0314  trunc 72.3%
  Step 36  reward 0.0109  trunc 83.2%
  Step 40  reward 0.0012  trunc 95.7%   <- COLLAPSE
Train reward (teacher-gap PG) peaked ~0.375 (step 2) and got noisy/low later (trainable frac
  fell to ~17-42%); off-policy stayed <=1.

READING: OPCD briefly internalizes cheatsheet-conditioned behaviour (val +31% by step 12 as
  truncation DROPS 38->25%), then runs away: the sampled reverse-KL gap keeps pushing the plain
  policy toward longer cheatsheet-style reasoning it cannot terminate, truncation climbs
  38->96% and reward collapses to ~0. No stable internalization; the distilled signal degenerates
  into non-terminating reasoning. Consistent with the project-wide finding that the reasoning
  channel is net-harmful for this task (SFT answer-only solves it; thinking hurts). A reverse-KL
  sampled objective without a length/termination anchor is unstable here. Best (transient)
  checkpoint would be ~step 12; final weights are degenerate and NOT worth pushing.
  Honest caveat (as configured): sampled-token reverse-KL-flavoured, NOT full-vocab forward KL.

OPS NOTE (disk): first launch CRASHED ~step 14 with OSError "No space left on device" (NOT GPU).
  Cause: trainer maybe_clean() PRESERVES the ~18 GB weight-broadcast on every ckpt-interval step;
  [ckpt] interval=4 over 40 steps retained 180 GB+ of broadcasts and filled the 2 TB disk (RPT
  624 G + HF cache 362 G already resident). FIX: [ckpt] & [trainer.ckpt] interval=40 keep_last=1
  (only the final broadcast/ckpt preserved; eval interval stays 4 for the curve). Re-run held
  steady at ~36 G broadcasts. Same fix applied to rl_3way_rlsd.toml.

================================================================================
FULL-CORPUS pass@4 (BASE Qwen3.5-9B reference, 2026-06-25)
================================================================================
scripts/passk_fullcorpus.py — PLAIN 3-way prompt (thinking OFF, no cheatsheet), k=4,
  temp 0.7 top_p 0.95 max_tokens 4096, vLLM tp=8, env SYSTEM_PROMPT_3WAY + _make_extractor
  (zero leakage, same parser as RL eval). Model = BASE Qwen/Qwen3.5-9B (corpus learnability
  reference; NOT a trained checkpoint). Results: blog-eval/results/passk_fullcorpus.json.

  split     n     pass@1   pass@4 | CLAUDE p1/p4   CHATGPT p1/p4   GEMINI p1/p4
  train    2682   0.3619   0.7987 | 0.2833/0.7159  0.3688/0.8221   0.4334/0.8579
  val       414   0.3671   0.8116 | 0.2844/0.7319  0.3913/0.8188   0.4257/0.8841
  val_ood   471   0.3381   0.7495 | 0.2118/0.5732  0.4188/0.8471   0.3838/0.8280
  parse_fail ~0.0002, truncation ~0 across all splits (clean).
  pred_dist (val): CLAUDE 349 / CHATGPT 637 / GEMINI 669 — model UNDER-predicts CLAUDE.

INTERPRETATION:
  * 3-way random at k=4 => pass@4 = 1-(2/3)^4 = 0.802, so pass@4 ~0.75-0.81 is NEAR the random
    ceiling and NOT itself evidence of skill. pass@1 is the meaningful metric.
  * pass@1 ~0.34-0.37 vs 0.333 chance => BASE is only marginally above feature-blind (matches the
    prior "base ~0.40 forced-choice" probe). Weak but real signal: provider asymmetry is consistent
    (GEMINI easiest ~0.38-0.43, CLAUDE hardest ~0.21-0.28) and the prediction distribution is
    skewed (systematic CLAUDE under-prediction), i.e. not pure uniform guessing.
  * val_ood is hardest (held-out categories): pass@1 0.338, and CLAUDE pass@4 only 0.573 — the
    OOD CLAUDE rows are the single hardest cell.
  * Baseline contextualizes the trained runs: SFT answer-only reaches val 1.000 (the signal IS
    learnable from the text); BASE near-chance pass@1 shows it is NOT trivially present zero-shot;
    OPCD/RL on the reasoning channel do not stably beat this (OPCD peaked val 0.291 then collapsed).

================================================================================
E3 — RLSD RESULT (RUN COMPLETE, 2026-06-25): best transient peak, then hard collapse
================================================================================
Launch: rl @ examples/blog_author_id/rl_3way_rlsd.toml --output-dir outputs/rlsd_e3
  Qwen3.5-9B from BASE, 4 train + 4 infer (tp=2), port 8300, teacher=student pool w/ cheatsheet
  spliced, max_steps=40, lr=1e-6, batch 144, eval PLAIN every 4. training_mode="rlsd":
  per-token weight (P_T/P_S)^sign(A), PPO/CISPO-clipped on student ratio; verifier gives DIRECTION,
  cheatsheet teacher gives MAGNITUDE, A==0 contributes nothing.

VAL trajectory (PLAIN eval, 414 val):
  Step  0  reward 0.2222  trunc 37.8%
  Step  4  reward 0.2585  trunc 35.5%
  Step  8  reward 0.2874  trunc 28.9%
  Step 12  reward 0.4094  trunc  9.2%   <- PEAK (+0.187 abs / +84% rel over step 0; trunc 38->9%)
  Step 16  reward 0.2983  trunc  8.6%
  Step 20  reward 0.0024  trunc 98.6%   <- CLIFF
  Step 24  reward 0.0000  trunc 100.0%
  Step 28  reward 0.0000  trunc 100.0%
  Step 32  reward 0.0000  trunc 100.0%
  Step 36  reward 0.0000  trunc 100.0%
  Step 40  reward 0.0000  trunc 100.0%
Train reward went to 0.0 from step ~20 with trainable frac ~99-100% (every rollout truncated,
  verifier reward 0) — the policy is fully degenerate (non-terminating) but still "on-policy".

READING: RLSD is the STRONGEST reasoning-channel result we have at its peak — val 0.4094 at step 12
  vs OPCD's 0.2911 and the BASE/plain eval 0.2222 — AND it is the only method that IMPROVED
  termination (truncation 38%->9%) while improving accuracy. The verifier anchor (reinforce
  correct, terminating rollouts; suppress incorrect) plus the cheatsheet teacher magnitude gives a
  genuinely useful early signal: the distillation direction is RIGHT. But with no KL-to-base / no
  length or entropy regularizer, the (P_T/P_S)^sign(A) weighting amplifies a degenerate mode and
  the policy falls off a cliff between step 16 and 20 into 100% truncation (reward 0), even harder
  than OPCD's gradual decay. Net: real but UNSTABLE; needs a trust-region/anchor to hold the peak.

OPCD vs RLSD (both BASE-init, 40 steps, identical eval):
  metric              base/step0   OPCD peak(s12)   RLSD peak(s12)   both final(s40)
  val reward          0.2222       0.2911           0.4094           ~0.00
  truncation          37.8%        25.4%            9.2%             ~100%
  => RLSD peak >> OPCD peak; RLSD also fixes truncation (verifier direction). Both collapse w/o a
  regularizer. Best (transient) checkpoint = RLSD step 12; final weights of BOTH are degenerate.

WHY THE COLLAPSE (theory, both runs): the teacher/student logprob gap on SAMPLED tokens is a
  reverse-KL-flavoured pull; reverse KL is mode-seeking and, unconstrained, collapses onto a
  high-teacher-logprob mode. Here that mode is "keep emitting cheatsheet-style reasoning" which
  never terminates within max_tokens => truncation -> 100%, verifier reward -> 0, and (RLSD) the
  ratio weight keeps reinforcing it because A is computed before the gate. A forward-KL (full-vocab)
  objective and/or an explicit length/termination penalty + KL-to-base trust region is the missing
  ingredient. Consistent with the project finding: the reasoning channel is net-harmful here unless
  tightly anchored; answer-only SFT (val 1.000) remains the only stable solver.

================================================================================
E3b — RLSD + TRUST-REGION ANCHOR: CODE BUILT + VALIDATED, QUEUED (2026-06-26)
================================================================================
Motivation: E3 RLSD peaked val 0.409 @s12 (trunc 38->9%) then cliffed to 100% truncation by s20.
Test: does a TRUST REGION hold the peak? New native mode training_mode="rlsd_anchored"
(rlsd_anchored_loss_fn) = RLSD signal + explicit per-step trust region. CLEAN ATTRIBUTION: only the
loss-side anchor changes vs E3 (truncation penalty OFF, advantage default).

Anchor (vs unanchored rlsd_loss_fn):
  - clip_c 2.0 -> 1.0           (coef in [1/e, e] instead of [1/e^2, e^2])
  - eps    0.2 -> 0.1           (tighter CISPO ratio clip)
  - RATIO-based trust-region MASK keyed on update direction (sign of coef): drop tokens whose
    trainer/rollout ratio already moved past 1 +/- 0.1 in the push direction. Original RLSD applied
    coef to EVERY token, no guard. [rubber-duck fix: ratio band, NOT the absolute-prob threshold an
    earlier draft used, which is blind at very low/high token probs]
  - KL-to-rollout: proper k3 KL (mismatch_kl), beta 0.5, replacing the near-zero 1e-3*log_ratio^2.

HONEST SCOPE (rubber-duck-reviewed): anchors to the BEHAVIOUR (rollout) policy mu, NOT a frozen base
(prime-rl plumbs no reference model; LossInputs has only trainer/inference/teacher logprobs + adv +
mask). => LOCAL per-step trust region: blocks the single-step explosion that caused the E3 cliff,
but cannot by itself pin the policy to the s12 peak if mu drifts slowly. If it still collapses, THAT
is the finding (a local anchor is insufficient; a frozen/best-ckpt reference or a length/termination
penalty would be the next lever). PG sign verified correct (coef>0 pushes prob up, <0 down; A==0 ->
0); k3 KL gradient (importance_ratio - 1) is NOT detached, correctly opposes drift.

Plumbing: training_mode "rlsd_anchored" added to transport/types.py (TrainingMode + TEACHER_LOGPROB_
MODES), orchestrator/dispatcher.py, configs/orchestrator.py (Literal + the teacher/cheatsheet
validators). loss.py: rlsd_anchored_loss_fn + setup_loss_fns dispatch. Config
examples/blog_author_id/rl_3way_rlsd_anchored.toml (output-dir outputs/rlsd_anchored_e3b). Validated:
py_compile OK; OrchestratorConfig(**toml) builds with mode=rlsd_anchored; loss fwd/backward finite,
A==0 coef==0, high-ratio push-up token correctly masked.

STATUS: QUEUED — awaiting a free 8xH100 (box currently running the RPT continued-pretraining job;
will not interfere). Launch when free:
  uv run rl @ examples/blog_author_id/rl_3way_rlsd_anchored.toml --output-dir outputs/rlsd_anchored_e3b

--------------------------------------------------------------------------------
E3b — RLSD + TRUST-REGION ANCHOR: RAN (2026-06-26). FINDING: STABILISED, NO GAIN.
--------------------------------------------------------------------------------
BASE-init, 40 steps, eval interval 4, identical PLAIN val eval as OPCD/RLSD. Full trajectory
(val reward = accuracy / truncation %):
  step:   0      4      8      12     16     20     24     28     32     36     40
  val:   0.214  0.211  0.198  0.213  0.181  0.185  0.197  0.248* 0.234  0.225  0.228
  trunc%: 37.1   40.6   48.8   47.9   55.1   56.6   57.0   47.7   46.6   47.5   44.2
  (* best val 0.2476 @ s28; best-step train reward stayed flat 0.28-0.34 throughout, no collapse)

RESULT: the trust region DID ITS JOB on stability and ONLY that. Contrast with E3 (RLSD):
  metric            BASE/s0   RLSD(E3) peak   RLSD(E3) final   RLSD-ANCHORED(E3b)
  val peak          0.2222    0.4094 (s12)    0.000 (s20+)     0.2476 (s28)
  truncation        ~38%      9% (s12)        100% (s20+)      37->57->44% (bounded)
  collapse?         --        HARD cliff s16->20 to 100% trunc  NONE (never > 57%)
  => E3b NEVER collapses (truncation bounded 37-57%, recedes to 44% by end) BUT never gains:
     val hovers in a flat 0.18-0.25 band, indistinguishable from the BASE/step-0 baseline.

READING (confirms the rubber-duck hypothesis exactly): anchoring to the BEHAVIOUR policy mu damps
  out BOTH the E3 truncation runaway AND the transient s12=0.409 peak. The s12 spike in E3 was an
  unstable excursion off mu; a per-step trust region that pins the policy near mu necessarily
  prevents that excursion in either direction. Net: the local anchor removes the downside (collapse)
  and the upside (peak) together -> flat, no-learning. As predicted, a mu-anchor CANNOT pin the
  policy to a peak that mu itself never holds. To capture the RLSD early signal you need a NON-moving
  reference (frozen base / best-ckpt EMA) or an explicit length/termination penalty + best-ckpt
  selection -- NOT a local trust region. E3b is the clean negative control that establishes this.

DISK: post-run reclaimed weights+broadcasts+rollouts (~55G); kept launch.log/logs/wandb summary.

================================================================================
E-LEX — PLAIN GRPO + HINT-FREE LEXICAL (TF-IDF) PROMPT: RUNNING (2026-06-26)
================================================================================
Motivation (user): keep the RL VANILLA -- CONTROL settings DPPO + Dr.GRPO (default DPPO+KL loss,
NO override = dppo_mask 0.2/0.2; default Dr.GRPO advantage; no teacher / no cheatsheet / no shaping)
-- and change ONLY the system prompt to a lexical-correlation (TF-IDF-style) one that tells the model
to weigh the most DISCRIMINATIVE tokens (high term-freq for one provider, rare in the others). Tests
whether STEERING the reasoning toward concrete lexical fingerprints (vs the generic holistic "judge
HOW it's written") lets plain RL climb above the ~0.36 feature-blind pass@1 / 0.222 plain-eval floor.

KEY DESIGN (user constraint): the prompt does NOT reveal which tokens map to which provider and gives
NO example words -- it explicitly says "you are NOT told which tokens belong to which provider ...
you must infer those associations entirely on your own". RL must DISCOVER the word<->provider
correlations itself; the prompt only directs attention to surface form (word/phrase choice,
punctuation, formatting) over topic. Identical to the CONTROL run (rl_3way_div_default) except
prompt_variant="lexical". Reasoning ON (require_reason, min_reason_words=12); same PLAIN val eval
(temp 0.7) as OPCD/RLSD/E3b for direct comparability. Config examples/blog_author_id/rl_3way_lexical.toml
(output-dir outputs/lexical_elex). STATUS: launched, healthy (eval@0 in progress). Results below.

--------------------------------------------------------------------------------
E-LEX — RAN (2026-06-26). FINDING: FIRST CLEAN (no-teacher) RUN WITH A SUSTAINED CLIMB, NO COLLAPSE.
--------------------------------------------------------------------------------
BASE-init, 40 steps, eval interval 4. Eval uses the SAME lexical prompt (prompt_variant=lexical) at
PLAIN temp 0.7. Full trajectory (val reward = accuracy / truncation %):
  step:   0      4      8      12     16     20     24     28     32     36     40
  val:   0.137  0.124  0.109  0.109  0.132  0.163  0.203  0.239  0.239  0.251  0.262*
  trunc%: 44.8   47.0   49.3   50.7   43.0   33.0   31.0   27.4   33.5   30.0   26.2
  (* best = FINAL, s40 0.2621, and the slope is still POSITIVE -- not plateaued)
  train reward climbed in parallel (~0.31 early steps; trainable 100%, train-temp truncation 0%).

SHAPE: a U with a strong recovery. s0 starts BELOW the generic-prompt baseline (lexical BASE 0.137 vs
generic plain-eval 0.222) -- the more demanding lexical prompt CONFUSES the untrained BASE -- dips to
a trough 0.109 @ s8-12, then RL learns to EXPLOIT the lexical framing: a monotonic climb 0.109 -> 0.262
across s12->s40 while truncation FALLS 51% -> 26%. By s40 it is above the generic plain-eval baseline
(0.262 > 0.222) and still rising.

WHY THIS IS THE STANDOUT RESULT: across every reasoning-channel method tried (trio, entropy-decay,
PVG variants, OPCD, RLSD, RLSD-anchored) this is the FIRST run that is simultaneously (a) CLEAN -- no
teacher, no cheatsheet, no advantage shaping, no custom loss; pure CONTROL DPPO + Dr.GRPO -- and (b)
shows a SUSTAINED, non-collapsing productive trajectory (accuracy up AND truncation down through the
final step). OPCD/RLSD gained only transiently then cliffed; E3b was flat; the trio runs plateaued or
collapsed. The ONLY thing changed here vs the collapsing control is the PROMPT: steering the model to
hunt discriminative lexical tokens (and -- per user constraint -- WITHOUT being told which tokens map
to which provider; it must infer the correlations itself) gives the verifier reward a learnable
gradient that does not run away into truncation.

HONEST CAVEATS:
  - Absolute number is still MODEST: 0.262 single-sample val, far below answer-only SFT (val 1.000)
    and below generic pass@4 (0.367). The win is the TRAJECTORY/STABILITY, not a new SOTA accuracy.
  - The eval prompt differs from the OPCD/RLSD/E3b/pass@4 baselines (those use the GENERIC prompt),
    so cross-config s0 numbers are NOT apples-to-apples at the prompt level. The clean within-run
    statement is: under its OWN prompt, RL took BASE 0.137 -> 0.262 (~1.9x) with truncation halved.
    The clean cross-config statement is: lexical-RL s40 0.262 edges the generic plain-eval 0.222, and
    is the only clean run still trending UP at step 40.
  - 3-way chance = 0.333; 0.262 is below chance on raw accuracy, consistent with the model
    UNDER-using the answer slot early (truncation 26-50%); the signal is the slope + truncation drop.
  - Zero-shot lexical pass@4 baseline (BASE, lexical prompt, full corpus) launched alongside to
    isolate the prompt effect vs the 0.367 generic pass@4 -- results appended below when done.

NEXT LEVER (if pursued): E-LEX had not plateaued at s40 -> run LONGER (80-120 steps) and/or add
best-ckpt selection; the positive, non-collapsing slope is the first evidence that a PROMPT change
(not a loss/teacher change) is what unlocks productive reasoning-channel RL here.

ZERO-SHOT LEXICAL pass@4 (BASE, lexical prompt, full corpus, k=4, temp 0.7, max_tokens 4096, tp=8):
  split     pass1    pass4    | generic-prompt pass1/pass4 (prior)   delta pass1
  train     0.3499   0.8016   | 0.362 / 0.799                        -0.012
  val       0.3400   0.7826   | 0.367 / 0.812                        -0.027
  val_ood   0.3519   0.8025   | 0.338 / 0.749                        +0.014
  per-provider (val): CLAUDE 0.221/0.601  CHATGPT 0.444/0.899  GEMINI 0.355/0.848
  (truncation ~0% here -- standalone vLLM harness; same CLAUDE-hardest/CHATGPT-easiest ordering)

READING: the lexical prompt does NOT help the UNTRAINED BASE zero-shot -- pass@1 and pass@4 are
basically tied with the generic prompt (val even slightly LOWER: 0.340 vs 0.367 pass1, 0.783 vs 0.812
pass4). So E-LEX's climb is NOT the prompt handing the model free accuracy; it is RL LEARNING to use
the lexical framing. The prompt provides a better SURFACE TO LEARN ON (a structured "find
discriminative tokens" objective the verifier gradient can act on), even though it is initially
neutral-to-slightly-worse. This is the clean version of the result: prompt-alone ~ 0; prompt + RL =
the only sustained non-collapsing climb. HARNESS CAVEAT (important): standalone pass@k truncates ~0%
whereas the RL-orchestrator eval truncates ~38-45% at step 0 (a known, consistent harness gap that
also affects the generic baselines) -- so the RL-harness E-LEX s0 (0.137) and this standalone lexical
pass1 (0.340) are NOT the same measurement; compare lexical-vs-generic WITHIN each harness only.
Result file: blog-eval/results/passk_fullcorpus_lexical.json.

--------------------------------------------------------------------------------
E-LEX-LONG — RAN (2026-06-27). KEY FINDING: THE 40-STEP CLIMB DID NOT REPRODUCE (variance).
--------------------------------------------------------------------------------
Identical config to E-LEX (plain GRPO, control DPPO+Dr.GRPO, lexical hint-free prompt, BASE init)
EXCEPT max_steps 40 -> 100 (and ckpt interval 40 -> 100 for disk). Goal: does the E-LEX climb
(0.137 -> 0.262, still rising @s40) continue, plateau, or collapse at a longer horizon?

Full val trajectory (reward = accuracy / truncation %), eval every 4:
  s0  0.126/48   s4  0.110/50   s8  0.092/56   s12 0.089/55   s16 0.091/61   s20 0.099/60
  s24 0.074/62   s28 0.069/62   s32 0.063/66   s36 0.091/70   s40 0.073/68   s44 0.080/70
  s48 0.065/72   s52 0.073/72   s56 0.080/76   s60 0.094/77*  s64 0.117/72   s68 0.117/64
  s72 0.129/58   s76 0.146/55^  s80 0.132/54   s84 0.111/52   s88 0.101/45   s92 0.066/46
  s96 0.040/40   s100 0.041/39
  (* truncation peak ~77% @s56-60;  ^ best val 0.146 @s76 -- far below E-LEX's 0.262 @s40)

THE RESULT IS A NON-REPRODUCTION. Same config, reseeded (rollout sampling temp 1.0 + async
off-policy dispatch are stochastic even with PRIME_RL_PRESERVE_DATA_ORDER=1), gives the OPPOSITE
shape to the 40-step E-LEX:
  metric                E-LEX (40-step)        E-LEX-LONG (100-step)
  s0 val                0.137                  0.126           (~same start)
  s40 val               0.262 (climbing)       0.073           (drifted DOWN)
  best val              0.262 (@s40, the end)  0.146 (@s76, a transient wobble)
  s40 truncation        26% (falling)          68% (rising)
  final val             0.262                  0.041
  shape                 monotonic CLIMB        truncation DRIFT up to ~77%, low accuracy throughout,
                                               one transient partial recovery (s64-80), then decays

HONEST REVISION of the E-LEX writeup: the earlier "first clean run with a SUSTAINED climb / standout
result" claim DOES NOT HOLD UP. The 40-step climb was a FAVORABLE STOCHASTIC EXCURSION, not a stable
property of plain-RL + lexical-prompt. A reseeded identical run does not reproduce it; instead it
shows the SAME truncation-drift fragility as every other reasoning-channel method here (OPCD, RLSD,
trio, PVG). Run-to-run variance DOMINATES the effect of the lexical prompt at this scale/recipe. The
correct conclusion is the conservative one consistent with the whole project: the reasoning channel
is NOT a reliable lever for this task under plain GRPO; answer-only SFT (val 1.000) remains the only
stable solver. The lexical prompt neither reliably helps (zero-shot pass@4 ~ generic; see prior
entry) nor reliably trains (40-step up, 100-step down).

METHODOLOGICAL NOTE: a single 40-step RL curve is NOT sufficient evidence of a "trend" on this task
given the observed variance -- future reasoning-channel claims need >=2-3 seeds before any climb is
called real. E-LEX-LONG is the seed that refutes the single-seed optimism.

DISK: post-run reclaimed weights+broadcasts+rollouts; 93G free. Config
examples/blog_author_id/rl_3way_lexical_long.toml.

================================================================================
COLD-START ABLATION (SFT warm-up -> reasoning RL) — RAN 2026-06-27
================================================================================
MOTIVATION. Every reasoning-channel RL run in this project so far was initialised from the
BASE Qwen3.5-9B and started near chance (val ~0.13-0.21) with truncation that drifts to ~100%
(OPCD, RLSD, trio, PVG, E-LEX-LONG). Hypothesis: the collapse is a COLD-START pathology — the
base model cannot yet produce a well-formed bounded <reason_why>/<answer> trace, so RL optimises
into degenerate long/truncated generations before it can find signal. Fix: give RL a competent
starting policy via a short SFT warm-up on verified reasoning traces, THEN run the SAME control RL.

RECIPE (two phases, all 8 GPUs, one at a time):
  Phase 1 — sft_coldstart.toml: 60-step SFT from BASE Qwen3.5-9B on data/blog_sft_star_e1_trunc
    (1071 STaR self-distilled traces that PASSED the verifier; system prompt == env
    SYSTEM_PROMPT_3WAY exactly, assistant in <reason_why>...</reason_why><answer>...</answer>).
    lr 1e-5, warmup 10, max_steps 60. Loss converged ~0.74. Saved weights/step_60 (HF-loadable).
    NOTE: trainer weight-save omits the multimodal processor files (Qwen3.5-9B is a VLM); had to
    copy preprocessor_config.json / video_preprocessor_config.json / merges.txt / vocab.json from
    the base snapshot into step_60 so vLLM + hf trainer could load the local path.
  Phase 2 — rl_3way_coldstart.toml: generic reasoning GRPO, CONTROL settings (DPPO + Dr.GRPO,
    no teacher, no cheatsheet, hint-free generic SYSTEM_PROMPT_3WAY, require_reason min 12 words),
    init from outputs/coldstart_sft/weights/step_60, max_steps 40, eval every 4, port 8300.

FULL val trajectory (reward = accuracy / truncation %), eval every 4 steps:
  s0  0.670/0   s4  0.666/0   s8  0.704/0   s12 0.743/0   s16 0.771/0   s20 0.767/0
  s24 0.773/0   s28 0.826/0   s32 0.819/0   s36 0.882/0   s40 0.895/0
  TRUNCATION = 0.0% at EVERY eval and EVERY train step. Error 0.0%. Turns 1.0.

RESULT — THIS IS THE HEADLINE. Cold-start turns reasoning RL from "collapses every time" into a
clean, near-monotonic CLIMB that NEVER collapses:
  metric                BASE-init reasoning RL (OPCD/RLSD/E-LEX-LONG)   COLD-START (this run)
  s0 val                ~0.13-0.21                                      0.670
  s0 truncation         high & rising                                   0.0%
  trajectory            truncation drifts to 60-100%, val sinks         val climbs 0.670 -> 0.895
  truncation @ end      ~40-100%                                        0.0%
  collapse?             YES (every seed)                                NO
The 60-step SFT warm-up both (a) lifts the starting policy to 0.67 @ 0% truncation and (b) removes
the truncation-drift failure mode entirely — confirming the collapse was a cold-start pathology,
not an intrinsic property of the reasoning channel under plain GRPO. Best val 0.895 @ s40, still
rising at the horizon.

CAVEATS (honest, per the E-LEX-LONG lesson on single-seed optimism):
  - SINGLE SEED. The evidence here is far stronger than E-LEX's single excursion (11 consecutive
    evals, monotone-ish climb, 0% truncation throughout — not one lucky point), but a 2nd seed
    would confirm reproducibility before calling the +0.22 climb a stable slope.
  - STILL BELOW answer-only SFT (val 1.000). Cold-start RL reaches 0.895; it does NOT beat the
    answer-only solver. Its value is SCIENTIFIC: it isolates *why* reasoning-channel RL collapsed
    (cold start) and shows a warm start fixes the dynamics, not that reasoning > answer-only.
  - The Phase-2 gain rides on the SFT prior; how much is RL vs. residual SFT drift is not separated
    here (the SFT-60 itself starts Phase 2 at 0.67).

DISK: post-run reclaimed coldstart_rl weights+broadcasts+rollouts (68G free). Kept
outputs/coldstart_sft/weights/step_60 (the warm-start checkpoint). Configs:
examples/blog_author_id/sft_coldstart.toml + rl_3way_coldstart.toml.

================================================================================
ALL-MODELS TRACE INFERENCE (val + val_ood, plain prompt) — RAN 2026-06-27
================================================================================
Downloaded each of the 6 pushed Hub finetunes (CK0607/qwen3.5-9b-blogprovider-*), ran inference
on the FULL val (414) + val_ood (471) sets with the PLAIN generic SYSTEM_PROMPT_3WAY (comparable
across models), stored per-model JSONL traces + summary.json under blog-eval/traces/<model>/, then
deleted the weights. Hub finetunes lack a chat_template -> rendered with the base Qwen3.5-9B
template (identical vocab). One model per process, tp=8.

  model                          val acc   val_ood acc   notes
  sft-goldcond                   1.000     1.000         answer-only solver, 0% trunc
  selfgated-answeronly           1.000     1.000         answer-only solver, 0% trunc
  star-selfdistill               0.952     0.960         reasoning SFT, 0% trunc, ~balanced preds
  rl-cheatsheet                  0.372     0.374         collapsed to mostly-CLAUDE, ~1-2% trunc
  rl-entropydecay                0.290     0.304         collapsed to mostly-GEMINI, ~12% trunc
  rl-pureacc-peak                0.384     0.382         collapsed to mostly-CHATGPT, 0% trunc

Confirms the standing picture on held-out data: the two answer-only SFT models generalise perfectly
(1.000 on both val and the OOD split); STaR reasoning SFT is close (~0.95-0.96); and all three
reasoning-channel RL checkpoints sit near/below chance with degenerate single-class prediction
collapse. Traces archived at blog-eval/traces/. (This is the BASE-init RL collapse that the
cold-start ablation above fixes.)

================================================================================
PROVIDER WRITING-STYLE ANALYSIS (semantic vs stylometric) — 2026-06-27
================================================================================
Built scripts/provider_semantic_viz.py: embeds every val/val_ood blog two ways and runs a linear
probe (LogReg, fit on train) to test the project's core claim (style not topic; densely separable).

LINEAR-PROBE 3-way accuracy (fit on train, eval held-out):
  representation                 val      val_ood
  semantic embedding (MiniLM)    0.811    0.849     <- content leaks SOME style but confusable
  stylometric (~40 feats)        1.000    1.000     <- PERFECT separation, incl. OOD topics
=> the task IS a dense, linearly-separable STYLE map. Same 1.000 ceiling as answer-only SFT; no
   reasoning headroom -> explains why reasoning-channel RL matched the ceiling or collapsed.

PER-PROVIDER mean style (the giveaways):
  feature                CLAUDE   CHATGPT  GEMINI
  em-dash rate           1.89     0.10     0.59     (Claude ~19x ChatGPT -- top single feature)
  avg sentence len (w)   20.5     13.0     18.9
  bold ** rate           2.07     0.61     1.86
  ##/### header rate     0.52     0.94     0.68
  LaTeX rate             2.29     2.48     4.14
  ASCII-diagram rate     0.06     0.06     4.99     (Gemini ~80x -- top single feature)
  md-table rate          0.07     0.29     0.84
  however/furthermore    low      low      high
  CLAUDE  = essayistic (long sentences, em-dashes, semicolons, bold, first-person argument)
  CHATGPT = structured/pedagogical (short sentences, ## headers, worked examples, questions)
  GEMINI  = visual/explanatory (ASCII diagrams, tables, LaTeX, connective markers, "we" voice)

CORROBORATION: matches the model-derived fingerprints harvested from correct <reason_why> rollouts
(run12/run15e: Claude "essayistic/nuanced/em-dash" up to 10x lift; ChatGPT "worked example/numbered/
LaTeX"; Gemini "ascii 7.5x / diagrams 6.8x / confident narrative"). The models learned the REAL
stylometric signature, not a shortcut; "reasoning" is post-hoc rationalisation of a style-determined
decision -> why supervising rationales (STaR 0.95) < answer-only (1.000).

Artifacts: blog-eval/analysis/{provider_umap,style_separation,provider_confusion}.png +
analysis_summary.json. Pushed to dataset CK0607/qwen3.5-9b-blogprovider-traces.

================================================================================
FULL RUN LADDER (every run: change, result, wandb + HF links) — 2026-06-27
================================================================================
W&B entity ChinmayK0604 / project blog-author-id-rl (offline runs synced post-hoc). Checkpoints
public under hf.co/CK0607; traces in dataset CK0607/qwen3.5-9b-blogprovider-traces.

| Run | What changed | Key result (val / val_ood) | W&B | HF ckpt | HF traces |
|---|---|---|---|---|---|
| **SFT — gold-conditioned** | SFT distilling gold-label-conditioned teacher rationales onto plain prompt (2892 balanced) | val **1.000** / ood **1.000** | — | [ckpt](https://huggingface.co/CK0607/qwen3.5-9b-blogprovider-sft-goldcond) | [traces](https://huggingface.co/datasets/CK0607/qwen3.5-9b-blogprovider-traces/tree/main/qwen3.5-9b-blogprovider-sft-goldcond) |
| **SFT — answer-only (self-gated)** | Answer-only SFT on 1071 verifier-gated self-rollouts; rationale stripped | val **1.000** / ood **1.000** (beats STaR) | — | [ckpt](https://huggingface.co/CK0607/qwen3.5-9b-blogprovider-selfgated-answeronly) | [traces](https://huggingface.co/datasets/CK0607/qwen3.5-9b-blogprovider-traces/tree/main/qwen3.5-9b-blogprovider-selfgated-answeronly) |
| **SFT — STaR self-distill** | On-policy STaR; supervise model's OWN verifier-gated <reason_why> | val 0.952 / ood 0.960 | — | [ckpt](https://huggingface.co/CK0607/qwen3.5-9b-blogprovider-star-selfdistill) | [traces](https://huggingface.co/datasets/CK0607/qwen3.5-9b-blogprovider-traces/tree/main/qwen3.5-9b-blogprovider-star-selfdistill) |
| grpo-thinkoff (baseline) | First GRPO, thinking off, 3-way | early baseline; sub-SFT | [wandb](https://wandb.ai/ChinmayK0604/blog-author-id-rl/runs/83bc29b237884b86bb813438827e7503) | — | — |
| div-default (control) | Control DPPO + Dr.GRPO, generic prompt, topic-diverse data | plateaus ~0.40 (reasoning ceiling) | [wandb](https://wandb.ai/ChinmayK0604/blog-author-id-rl/runs/0tnz73pq) | — | — |
| div-entropy | + entropy bonus on the control | no gain over control | [wandb](https://wandb.ai/ChinmayK0604/blog-author-id-rl/runs/oyje2fq2) | — | — |
| curriculum / interleaved | Topic-curriculum & interleaved orderings | no durable gain | [wandb](https://wandb.ai/ChinmayK0604/blog-author-id-rl/runs/rguqxbgn) | — | — |
| contrastive / pairwise / hardpair | Pair-mining data variants (hard provider pairs) | no durable gain | [wandb](https://wandb.ai/ChinmayK0604/blog-author-id-rl/runs/0aq5vwk2) | — | — |
| pvg-entropy | Prover-verifier-game style + entropy | no durable gain | [wandb](https://wandb.ai/ChinmayK0604/blog-author-id-rl/runs/y3sb9g8m) | — | — |
| **RL — cheatsheet (trio)** | GRPO w/ train-style cheatsheet in context, trio curriculum | cheatsheet-free ~0.40; plain **0.372/0.374** | [wandb](https://wandb.ai/ChinmayK0604/blog-author-id-rl/runs/c8os8n8z) | [ckpt](https://huggingface.co/CK0607/qwen3.5-9b-blogprovider-rl-cheatsheet) | [traces](https://huggingface.co/datasets/CK0607/qwen3.5-9b-blogprovider-traces/tree/main/qwen3.5-9b-blogprovider-rl-cheatsheet) |
| **RL — entropy-decay** | Entropy-decay schedule ablation (final ckpt) | negative ablation; plain **0.290/0.304** | [wandb](https://wandb.ai/ChinmayK0604/blog-author-id-rl/runs/c8os8n8z) | [ckpt](https://huggingface.co/CK0607/qwen3.5-9b-blogprovider-rl-entropydecay) | [traces](https://huggingface.co/datasets/CK0607/qwen3.5-9b-blogprovider-traces/tree/main/qwen3.5-9b-blogprovider-rl-entropydecay) |
| **RL — pure-accuracy (peak)** | Answer-only / no-reasoning GRPO (reward=label match), peak step_16 | peaked 0.65 then collapsed; plain **0.384/0.382** | [wandb](https://wandb.ai/ChinmayK0604/blog-author-id-rl/runs/b1v5cru5) | [ckpt](https://huggingface.co/CK0607/qwen3.5-9b-blogprovider-rl-pureacc-peak) | [traces](https://huggingface.co/datasets/CK0607/qwen3.5-9b-blogprovider-traces/tree/main/qwen3.5-9b-blogprovider-rl-pureacc-peak) |
| **OPCD (E2)** | On-policy context distillation; cheatsheet teacher → plain student | re-run reproduced: peak ~0.27 (s8) → truncation-collapse; final ckpt val/ood **0.03/0.02** (96% trunc) | [wandb](https://wandb.ai/ChinmayK0604/blog-author-id-rl/runs/4dywekt0) | [ckpt](https://huggingface.co/CK0607/qwen3.5-9b-blogprovider-opcd-e2) | [traces](https://huggingface.co/datasets/CK0607/qwen3.5-9b-blogprovider-traces/tree/main/qwen3.5-9b-blogprovider-opcd-e2) |
| **RLSD (E3)** | Verifier-anchored self-distillation (sign×teacher-gap, PPO-clip) | re-run reproduced: peak ~0.35 (s12, trunc↓17%) → collapse; final ckpt val/ood **0.00/0.00** (100% trunc) | [wandb](https://wandb.ai/ChinmayK0604/blog-author-id-rl/runs/h485ltix) | [ckpt](https://huggingface.co/CK0607/qwen3.5-9b-blogprovider-rlsd-e3) | [traces](https://huggingface.co/datasets/CK0607/qwen3.5-9b-blogprovider-traces/tree/main/qwen3.5-9b-blogprovider-rlsd-e3) |
| RLSD-E3b (trust-region anchor) | + KL trust-region anchor on RLSD | collapse PREVENTED but gain killed (flat 0.18–0.25) | [wandb](https://wandb.ai/ChinmayK0604/blog-author-id-rl/runs/14zovr5c) | — | (weights not retained) |
| E-LEX (lexical, 40-step) | Hint-free lexical/style prompt, control GRPO, base init | 0.137→0.262 climb (single-seed, NOT reproduced) | [wandb](https://wandb.ai/ChinmayK0604/blog-author-id-rl/runs/rvlk8sgr) | — | (weights not retained) |
| E-LEX-LONG (lexical, 100-step) | Same, 100 steps — reseed | REFUTES climb: →0.041, trunc ~77% | [wandb](https://wandb.ai/ChinmayK0604/blog-author-id-rl/runs/brn85thu) | — | (weights not retained) |
| **Cold-start SFT (Phase 1)** | 60-step SFT warm-up on STaR traces (the RL init) | val 0.696 / ood 0.682 | [wandb](https://wandb.ai/ChinmayK0604/blog-author-id-rl/runs/92svhbn0) | [ckpt](https://huggingface.co/CK0607/qwen3.5-9b-blogprovider-coldstart-sft) | [traces](https://huggingface.co/datasets/CK0607/qwen3.5-9b-blogprovider-traces/tree/main/coldstart-sft) |
| **Cold-start RL (Phase 2) ★** | Control GRPO from SFT-60 (the headline) | **0.67→0.90, 0% trunc, NO collapse**; plain **0.911/0.919** | [wandb](https://wandb.ai/ChinmayK0604/blog-author-id-rl/runs/92svhbn0) | [ckpt](https://huggingface.co/CK0607/qwen3.5-9b-blogprovider-coldstart-rl) | [traces](https://huggingface.co/datasets/CK0607/qwen3.5-9b-blogprovider-traces/tree/main/coldstart-rl) |

NOTES: Cold-start RL (star) = only reasoning-channel run that climbs + never collapses; reproduced
2 seeds (0.670->0.895, 0.652->0.905, both 0% trunc). RLSD-E3b / E-LEX / E-LEX-LONG = weights NOT
retained (reclaimed for disk); curves in W&B + log narrative. RLSD-E3 + OPCD-E2 re-run from scratch
to regenerate pushable checkpoints (2026-06-27).

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
