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
