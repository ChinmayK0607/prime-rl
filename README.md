<p align="center">
  <img src="https://github.com/user-attachments/assets/40c36e38-c5bd-4c5a-9cb3-f7b902cd155d#gh-light-mode-only" alt="Prime Intellect" width="180">
  <img src="https://github.com/user-attachments/assets/6414bc9b-126b-41ca-9307-9e982430cde8#gh-dark-mode-only"  alt="Prime Intellect" width="180">
</p>

<h1 align="center">Who wrote this blog?</h1>
<h3 align="center">RL on the writing style of CLAUDE · CHATGPT · GEMINI</h3>

<p align="center"><i>A research fork of <a href="https://github.com/PrimeIntellect-ai/prime-rl">prime-rl</a> — framework setup is preserved in <a href="#setup-built-on-prime-rl">Setup</a> below.</i></p>

---

We train **Qwen3.5-9B (thinking OFF)** on a single 8×H100 node to read a ~3000-word blog post and name
which assistant wrote it — **CLAUDE**, **CHATGPT**, or **GEMINI** — and study what each training recipe
actually learns about their writing styles.

📖 **Full write-up:** [`RL_LOG.md`](RL_LOG.md) (the engineering blog, Parts I–XVI) &nbsp;·&nbsp;
🧭 **Exhaustive run/config/script index:** [`REPRODUCING_THE_BLOG.md`](REPRODUCING_THE_BLOG.md)

## Headline findings

- **Answer-only SFT solves the task** — val / val_ood = **1.000 / 1.000**.
- **Reasoning-channel RL from a base init reliably collapses** (truncation → 100%, accuracy → ~0) on
  every distillation/shaping variant we tried (OPCD, RLSD, cheatsheet, lexical).
- **A light reasoning cold-start fixes it**: 60 SFT steps → control GRPO climbs to **~0.91 with 0 %
  truncation and never collapses** (reproduced across seeds).
- **It's style, not content**: ~40 stylometric features give a **perfect linear probe (1.000 / 1.000)**
  while semantic/content embeddings intermix the providers — Claude = em-dashes + sincere essay voice;
  ChatGPT = hedged, header/list structure; Gemini = intensifiers + ASCII/LaTeX density.

## Results at a glance

| Model | Training | val | val_ood |
|---|---|---|---|
| `selfgated-answeronly` / `sft-goldcond` | answer-only / gold-conditioned SFT | **1.000** | **1.000** |
| `star-selfdistill` | STaR self-distillation | 0.952 | 0.960 |
| `coldstart-rl` ★ | SFT-60 cold-start → control GRPO | 0.911 | 0.919 |
| `coldstart-sft` | 60-step SFT only (the RL init) | 0.696 | 0.682 |
| `rl-pureacc` / `rl-cheatsheet` / `rl-entropydecay` | reasoning-channel RL (base init) | 0.29–0.38 | 0.30–0.38 |
| `rlsd-e3` / `opcd-e2` | verifier-/context-distillation (collapsed) | ~0.00 | ~0.00 |

Full ladder with W&B + Hub links is **Part XV** of [`RL_LOG.md`](RL_LOG.md).

## Artifacts on the Hugging Face Hub

| Artifact | Link |
|---|---|
| Source blog corpus | [`Samarth0710/copilot-sdk-blogs`](https://huggingface.co/datasets/Samarth0710/copilot-sdk-blogs) |
| Trained models (10 repos) | [`CK0607`](https://huggingface.co/CK0607) — `qwen3.5-9b-blogprovider-*` |
| Inference traces + style study | [`CK0607/qwen3.5-9b-blogprovider-traces`](https://huggingface.co/datasets/CK0607/qwen3.5-9b-blogprovider-traces) |

## Quickstart (the experiments)

```bash
# 0. Framework setup — see "Setup (built on prime-rl)" below, then install the env:
uv pip install -e deps/research-environments/environments/blog_author_id

# 1. Build the data splits from the HF corpus (writes data/blog_author_id_3way/{train,val,val_ood})
uv run python data/build_blog_split_3way.py

# 2. Launch a run (8x H100; trainer + vLLM colocated, RL inference server on port 8300)
uv run rl @ examples/blog_author_id/rl_3way_coldstart.toml --output-dir outputs/coldstart_rl

# 3. Trace a checkpoint on val + val_ood (writes blog-eval/traces/<name>/)
uv run python scripts/infer_traces_allmodels.py \
    --repo outputs/coldstart_rl/weights/step_40 \
    --short qwen3.5-9b-blogprovider-coldstart-rl --render_model Qwen/Qwen3.5-9B
```

Every run's config lives in [`examples/blog_author_id/`](examples/blog_author_id) and the
[reproduction guide](REPRODUCING_THE_BLOG.md) maps each one to its hypothesis, data builder, scripts,
and Hub artifacts.

---

## Setup (built on prime-rl)

This repo is a fork of [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) (async RL at scale); the experiments above use its SFT/RL trainer and vLLM inference unchanged except for the distillation hooks noted in the [reproduction guide](REPRODUCING_THE_BLOG.md). The original prime-rl setup follows.

> *We develop and test on NVIDIA RTX 3090/4090/5090, A100, H100, H200, and B200. If your setup fails, please create an [issue](https://github.com/PrimeIntellect-ai/prime-rl/issues).*

### Prerequisites

Currently, you **need at least one NVIDIA GPU to use PRIME-RL**. If you don't already have access to one, we recommend our [compute platform](https://app.primeintellect.ai) for everything from renting on-demand single GPUs for developing, debugging and small ablations, to [reserving 1000+ GPU clusters](https://app.primeintellect.ai/dashboard/quotes) for production-scale training.

### Quick Setup

Set up PRIME-RL in a single command.

```bash
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
```

<details>
<summary>
Manual Setup
</summary>
<br>

1. Clone the repository

```bash
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl
```

2. Initialize submodules

```bash
git submodule update --init -- deps/verifiers deps/renderers deps/research-environments deps/pydantic-config
```

3. Install [uv](https://docs.astral.sh/uv/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

4. Install dependencies from the lock file

```bash
uv sync --all-extras
```

4.1. On aarch64 hosts: build flash-attn from source for your GPU

> *NOTE*: aarch64 has no prebuilt flash-attn wheel. This step compiles the CUDA extension for your local GPU (~20-30 minutes). Compute capability is auto-detected from `nvidia-smi`; override with `TORCH_CUDA_ARCH_LIST=9.0` (Hopper) / `10.0` (Blackwell) if needed.
> *NOTE*: After this step, you can't run `uv sync --all-extras` or `uv run` as it will uninstall the package, you can avoid it by running `uv sync --inexact` or `uv run --no-sync`.

```bash
bash scripts/docker-arm64-post-install.sh
```

3.1. Optional: Install Flash Attention 3 (on Hopper GPUs only, for flash_attention_3 attention backend)

> *NOTE*: This step will take a while, as it builds the Flash Attention 3 extension from source, as it has no wheels prebuilt.
> *NOTE*: After this step, you can't run `uv sync --all-extras` or `uv run` as it will uninstall the package, you can avoid it by running `uv sync --inexact` or `uv run --no-sync`

```bash
uv pip install "flash-attn-3 @ git+https://github.com/Dao-AILab/flash-attention.git@main#subdirectory=hopper" --no-build-isolation
```

</details>

<details>
<summary>
Validate your environment setup
</summary>
<br>

1. Check that the environment uses Python 3.12

```bash
uv run python -V
```

2. Check that `flash-attn` is installed

```bash
uv run python -c "import flash_attn"
```

3. Check that you can run SFT trainer  (*this requires 1 GPU*)

```bash
uv run sft @ configs/debug/sft/train.toml
```

4. Check that you can run the RL trainer (*this requires 1 GPU*)

```bash
uv run trainer @ configs/debug/rl/train.toml
```

5. Check that you can run the inference server (*this requires 1 GPU*)

```bash
uv run inference @ configs/debug/infer.toml
```

*Keep the inference server running in the background for the next steps.*

5.1. Check that you can run the orchestrator against the inference server

```bash
uv run orchestrator @ configs/debug/orch.toml
```

5.2. Check that you can run evals against the inference server

```bash
uv run eval @ configs/debug/eval.toml
```

</details>

### Additional Setup

1. If you want to log your runs to [W&B](https://wandb.ai), log in

```bash
uv run wandb login
# Or set `export WANDB_API_KEY=...`
```

2. If you require gated/ private models or datasets from [HuggingFace](https://huggingface.co), log in

```bash
uv run hf auth login
# Or set `export HF_TOKEN=...`
```

## Training Examples
We provide end-to-end training examples in the [`examples`](examples) directory to highlight features of the framework and guide you through the process of training your own models.

### Basic Training: 1 to 8 GPUs

Follow this guide to learn the basics of Prime-RL. You can train your own models on 1 to 8 GPUs. Ideal for getting started and exploring the capabilities of the framework. These guides cover most use cases -- single-turn, multi-turn, tool calling, etc. -- on toy environments and small models.

1. [**Reverse Text**](examples/reverse_text/README.md): Train `Qwen3-0.6B` to reverse a small chunk of text. Demonstrates tiny-scale single-turn SFT and RL training. Can be trained on a single consumer GPU in a few minutes, and is ideal for getting started.
2. [**Wordle**](examples/wordle/README.md): Train `Qwen3-1.7B` to play Wordle. A fun example of multi-turn SFT and RL training. Can be trained on a 2-4 H100 GPUs in a few hours. Ideal for exploring the multi-turn training capabilities of the framework.
3. [**Alphabet Sort**](examples/alphabet_sort/README.md): Train `Qwen3-4B-Instruct-2507` to sort names alphabetically. Demonstrates multi-turn RL training via LoRA without SFT warmup. Can be trained on a single H100 GPU in just over an hour. Ideal for exploring LoRA-based training.
4. [**Wiki Search**](examples/wiki_search/README.md): Train `Qwen3-4B-Instruct-2507` to answer trivia questions by searching through a Wikipedia. Demonstrates multi-turn with web search tool use.
5. [**Hendrycks Sanity**](examples/hendrycks_sanity/README.md): Run a sanity check experiment on `DeepSeek-R1-Distill-Qwen-1.5B` using a filtered subset of MATH where the model already partially solves 20-80% of problems. Useful for algorithm ablations.

### Advanced Training: 32 - 2048 GPUs:

Follow this guide to train large models on hard reasoning and agentic / swe environments.
These guides are designed to be run from a Slurm cluster but can also be adapted to k8s deployments.

1. [**Qwen 3 30B - A3B Math**](examples/qwen30b_math/README.md): Train `Qwen3-30B-A3B` to solve hard math problems.
2. [**Qwen 3 30B - A3B SWE**](examples/qwen30b_swe/README.md): Train `Qwen3-30B-A3B` to solve hard SWE problems.
3. [**Intellect-3.1**](examples/Intellect-3.1/README.md): Reproduce our `INTELLECT-3.1` training run.
4. [**MiniMax-M2.5 SWE**](examples/minimax_m2.5_swe/README.md): Train `MiniMax-M2.5` on agentic SWE tasks.
5. [**High-throughput GLM-5**](examples/glm5_pd_disag/README.md): Train `GLM-5` with PD disaggregation and FP8 inference on SWE.

## Docs

Check out the [docs](docs) directory for in-depth guides on how to use PRIME-RL.

- [**Overview**](docs/overview.md) - Architecture, install, and a copy-pasteable end-to-end RL run
- [**Configuration**](docs/configuration.md) - TOML composition, CLI overrides, env vars, validation
- [**Training**](docs/training.md) - RL, SFT, evals, checkpointing, observability, rules of thumb
- [**Scaling**](docs/scaling.md) - Single-GPU through multi-node, FSDP/EP/CP, SLURM, benchmarking
- [**Algorithms**](docs/algorithms.md) - Async/off-policy training, the AIPO loss, advantage and filter plugins, trajectory merging
- [**Advanced**](docs/advanced.md) - Custom modeling, multimodal training, LoRA, multi-tenant training
- [**Development**](docs/development.md) - Test suite, pre-commit hooks, adding a new model

## Contributing

We warmly welcome community contributions! We use [issues](https://github.com/PrimeIntellect-ai/prime-rl/issues) to track bugs, feature requests, and share our internal roadmap. If you encounter bugs, have pain points during development, or have ideas for new features, please open an issue.

Contributions are welcome via PR. Please follow these guidelines:
1. Install the [pre-commit hooks](#pre-commit-hooks) to ensure your code is formatted correctly.
2. Please keep your PR in "Draft" until it is ready for review.
3. If your PR resolves an issue, please link the issue in the PR description
4. If you can, try running the [test suite](#tests) locally to ensure your changes are working as expected.

### Pre-Commit Hooks

Please install the [pre-commit](https://pre-commit.com) hooks to ensure your code is formatted correctly.

```bash
uv run pre-commit install
```

### Tests

```bash
uv run pytest -v                    # everything
uv run pytest tests/unit -v         # unit only
uv run pytest tests/integration -v  # integration only
uv run pytest -v -m "not gpu"       # CPU-only (inverse of the gpu marker)
```

## License

This project is licensed under the Apache 2.0 license, as found in the [License](LICENSE) file.

## Citation

If you find our work useful, feel free to cite it using

```tex
@misc{primeintellect2025prime-rl,
  author = {Prime Intellect},
  title = {PRIME-RL},
  url = {https://github.com/PrimeIntellect-ai/prime-rl},
  year = {2025}
}
```
