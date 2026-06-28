# Reproducing the blog — RL on the 3-way AI-provider-style task

This fork of [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) holds **all the code,
configs, and write-ups** behind the blog/log on training **Qwen3.5-9B (thinking OFF)** to identify
which AI assistant — **CLAUDE / CHATGPT / GEMINI** — authored a ~3000-word blog post, run on a single
8×H100 node.

The headline finding: **answer-only SFT solves the task (val / val_ood = 1.000)**, while
**reasoning-channel RL** from a base init reliably *collapses* (truncation → 100%, accuracy → ~0) —
**unless** the policy is first given a light reasoning **cold-start**, after which control GRPO climbs
to ~0.91 and never collapses. The providers are separable by **style, not content**: ~40 stylometric
features give a perfect linear probe (1.000 / 1.000) while semantic/content embeddings intermix them.

## Start here

| Document | What it is |
|---|---|
| [`EXPERIMENT_BLOG.md`](EXPERIMENT_BLOG.md) | The narrative blog — full arc, Parts I–XVI, every run and what it taught us. |
| [`RL_LOG.md`](RL_LOG.md) | The dense engineering log — run-by-run diagnostics, decisions, numbers. |
| **Part XV** (in both) | **Full run ladder**: every run, what changed, W&B link, HF checkpoint + traces links, result. |
| **Part XIV / XVI** (in both) | What the models learned about the three providers' styles (semantic-space study + trace-level prose analysis). |

If you only read one table, read **Part XV** — it is the index to every run and its artifacts.

## Published artifacts (Hugging Face)

| Artifact | Link |
|---|---|
| Source blog corpus | [`anonymousNeurIPS2026submission4281/copilot-sdk-blogs`](https://huggingface.co/datasets/anonymousNeurIPS2026submission4281/copilot-sdk-blogs) |
| Trained models (collection) | [`CK0607`](https://huggingface.co/CK0607) — 10 `qwen3.5-9b-blogprovider-*` repos |
| Inference traces + style analysis | [`CK0607/qwen3.5-9b-blogprovider-traces`](https://huggingface.co/datasets/CK0607/qwen3.5-9b-blogprovider-traces) |

Pushed models: `coldstart-rl`, `coldstart-sft`, `star-selfdistill`, `selfgated-answeronly`,
`sft-goldcond`, `rl-cheatsheet`, `rl-entropydecay`, `rl-pureacc-peak`, `rlsd-e3`, `opcd-e2`
(all under `CK0607/qwen3.5-9b-blogprovider-<name>`).

## Repository layout (the blog-specific parts)

```
examples/blog_author_id/*.toml      # every run config (RL + SFT), one file per experiment
deps/research-environments/environments/blog_author_id/
                                    # the verifiers environment (prompt, parser, reward, label schemes)
data/build_blog_*.py                # dataset builders (HF corpus -> train/val/val_ood on disk)
scripts/                            # eval, trace inference, analysis, model-push tooling
blog-eval/analysis/                 # style-separation study (figures + JSON) — committed, lightweight
EXPERIMENT_BLOG.md / RL_LOG.md      # the write-ups
```

Heavy artifacts are **not committed** (see `.gitignore`) and live on the Hub instead: the ~300 MB of
inference traces are in the traces dataset, and the ~1.5 GB of built datasets are regenerated locally
with the `data/build_blog_*.py` scripts.

## The task / environment

`deps/research-environments/environments/blog_author_id/blog_author_id.py` is a single-turn
[`verifiers`](https://github.com/willccbb/verifiers) environment:

- **Input**: one blog body (`question`). Frontmatter is stripped (it leaks the model).
- **Output**: a `<reason_why>…</reason_why>` block followed by an `<answer>` tag naming the provider.
- **Reward**: binary exact-match of the parsed `<answer>` against the gold provider (weight 1.0);
  `parsed_ok` is a weight-0 diagnostic. `label_scheme="provider3"` selects the 3-way CLAUDE/CHATGPT/GEMINI task.
- **Prompt variants** (`prompt_variant` env arg): `SYSTEM_PROMPT_3WAY` (default, generic), `lexical`,
  `cheatsheet`, etc. — used by the various ablations.

## Quickstart

```bash
# 1. Environment (uv-managed, same as upstream prime-rl)
uv sync && uv pip install -e deps/research-environments/environments/blog_author_id

# 2. Build the data splits from the HF corpus (writes data/blog_author_id_3way/{train,val,val_ood})
uv run python data/build_blog_split_3way.py

# 3. Launch a run (8x H100; trainer + inference colocated, RL server on port 8300)
uv run rl @ examples/blog_author_id/rl_3way_coldstart.toml --output-dir outputs/coldstart_rl

# 4. Trace a checkpoint on val + val_ood (writes blog-eval/traces/<name>/)
uv run python scripts/infer_traces_allmodels.py \
    --repo outputs/coldstart_rl/weights/step_40 \
    --short qwen3.5-9b-blogprovider-coldstart-rl \
    --render_model Qwen/Qwen3.5-9B
```

> **Note on paths.** The committed configs use **absolute** `data_dir` /init-weight paths
> (`/home/ubuntu/blogger/prime-rl/...`) and SFT-init configs expect a prior checkpoint under
> `outputs/`. Adjust `data_dir` to your checkout and run the prerequisite SFT phase first where a
> config's `[model] name` points at a local `outputs/.../weights/step_N`. The main 3-way runs point at
> a `data/blog_author_id_3way_v2` variant of the canonical `build_blog_split_3way.py` split (balanced
> by provider, with `val_ood` held out).

## Run → config map

Every experiment in the blog's run ladder corresponds to a config in `examples/blog_author_id/`. Each
config carries a header comment explaining its hypothesis and exact setup. The most important:

| Run (blog name) | Config | One-line description |
|---|---|---|
| **Cold-start SFT** (Phase 1) ★ | `sft_coldstart.toml` | 60-step SFT warm-up on verifier-passed STaR traces — the RL init. |
| **Cold-start RL** (Phase 2) ★ | `rl_3way_coldstart.toml` | Control GRPO (DPPO + Dr.GRPO) from SFT-60 — climbs to ~0.91, **0% truncation, no collapse**. |
| Answer-only SFT | `sft_warmup.toml` | Gold-conditioned answer-only SFT — **solves the task (1.000 / 1.000)**. |
| STaR self-distill | `sft_star_e1.toml` / `sft_star_e1_ans.toml` | Self-distillation on the model's own verifier-passed reasoning. |
| Pure-accuracy RL | `rl_3way_v2_pure_acc.toml` | Accuracy-only GRPO, no reasoning shaping — peaks then mode-collapses onto one class. |
| Cheatsheet GRPO | `rl_3way_trio_cheatsheet.toml` | Cheatsheet-elicited GRPO. |
| Entropy-decay GRPO | `rl_3way_trio_entdecay.toml` | Entropy-decay schedule to tame late collapse. |
| Trio contrastive | `rl_3way_trio.toml` | Triowise structural constraint on the marginal. |
| OPCD (E2) | `rl_3way_opcd.toml` | On-policy context distillation (cheatsheet teacher → plain student). |
| RLSD (E3) | `rl_3way_rlsd.toml` | Verifier-anchored self-distillation (sign × teacher-gap, PPO-clip). |
| RLSD-anchored (E3b) | `rl_3way_rlsd_anchored.toml` | RLSD + a trust-region / length anchor. |
| E-LEX / E-LEX-LONG | `rl_3way_lexical.toml` / `rl_3way_lexical_long.toml` | Hint-free lexical-style prompt, 40 / 100 steps. |
| Diverse-data ablation | `rl_3way_div_default.toml` / `rl_3way_div_entropy.toml` | Single-variable entropy on/off on diverse data. |
| PVG variants | `rl_3way_pvg_*.toml` | Prover-verifier-game variants (cispo, entropy, entdecay, maxrl, reasongate). |
| Base / overfit / eval | `rl_3way.toml`, `overfit*.toml`, `eval_ood_best.toml` | Base configs, overfit sanity checks, OOD eval. |

See **Part XV** of the blog/log for each run's W&B link, result, and Hub checkpoint/traces links.

## Data builders (`data/`)

All builders read the HF corpus `anonymousNeurIPS2026submission4281/copilot-sdk-blogs` and write HF
datasets to disk under `data/…/{train,val,val_ood}` with columns `question` / `answer` / `info`.

| Builder | Output | Notes |
|---|---|---|
| `build_blog_split_3way.py` | `data/blog_author_id_3way` | **Canonical 3-way split**, balanced by provider, val = 3 hardest categories held out, `val_ood` extra holdout. |
| `build_blog_split.py` | `data/blog_author_id` | Original 2-way (CLAUDE/CHATGPT) split. |
| `build_blog_3way_aug.py`, `build_blog_curriculum_*.py` | various | Augmentation / curriculum / contrastive / balanced ablation variants. |
| `build_blog_split_local.py` | local variant | Build from a local corpus copy. |

## Key scripts (`scripts/`)

| Script | Purpose |
|---|---|
| `infer_traces_allmodels.py` | Run a checkpoint (local path or Hub repo) over val + val_ood, write `<reason_why>/<answer>` traces + `summary.json`. |
| `push_models_hub.py` | Push selected checkpoints to the Hub (backfills the Qwen3.5 VLM processor files). |
| `provider_semantic_viz.py` | Style-separation study: semantic vs. stylometric UMAP + linear probe (Part XIV). |
| `analyze_trace_quirks.py` | Trace-level prose analysis: what each training method learned to notice/say (Part XVI). |
| `passk_fullcorpus.py` | pass@k over the full corpus (train + val). |
| `gen_star_e1.py`, `build_sft_warmup.py`, `build_cheatsheet_splice.py`, `derive_train_cheatsheet.py` | Generate the SFT/STaR/cheatsheet training data. |
| `diag_teacher_student.py`, `eval_sft_ckpts.py`, `style_probe_eval.py` | Teacher/student diagnostics, checkpoint eval, style probe. |

## Reproducibility notes

- **Hardware**: 1× node, 8× H100. RL configs colocate trainer + vLLM inference (`num_train_gpus` +
  `num_infer_gpus` = 8); the inference server uses **port 8300** (8000 is reserved on the dev box).
- **Model**: `Qwen/Qwen3.5-9B`, thinking OFF (`reasoning_parser = "None"`), `seq_len = 16384`.
  It is a VLM, so pushed/loaded checkpoints need the processor files backfilled — `push_models_hub.py`
  handles this; trace inference renders chat with `--render_model Qwen/Qwen3.5-9B`.
- **Source-code changes** vs upstream prime-rl (committed here): teacher-logprob transport, RLSD/OPCD
  loss paths, and orchestrator/dispatcher tweaks needed by the distillation runs — see
  `src/prime_rl/trainer/rl/loss.py`, `src/prime_rl/orchestrator/*`, `src/prime_rl/transport/types.py`.
- Some collapsed runs' raw weights were not retained (disk); their curves live in W&B and the run
  ladder marks them "weights not retained". All pushed checkpoints are reproducible from these configs.

For upstream prime-rl installation, training internals, and general usage, see the original project
documentation referenced from the main [`README.md`](README.md).
