import inspect
from dataclasses import dataclass
from typing import Any, Callable

import torch
from beartype import beartype as typechecker
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor

from prime_rl.configs.trainer import CustomLossConfig, DefaultLossConfig, LossConfig
from prime_rl.utils.utils import import_object


@dataclass
class LossInputs:
    """Inputs for computing loss on a single sample."""

    trainer_logprobs: Float[Tensor, " seq"]
    inference_logprobs: Float[Tensor, " seq"]
    teacher_logprobs: Float[Tensor, " seq"] | None
    advantages: Float[Tensor, " seq"]
    loss_mask: Bool[Tensor, " seq"]


@dataclass
class LossOutputs:
    """Outputs from computing loss on a single sample."""

    loss: Float[Tensor, ""]
    metrics: dict[str, Tensor]


LossFn = Callable[..., LossOutputs]
"""Type for a per-sample loss function.

Expected signature:
    def my_loss(inputs: LossInputs, **kwargs) -> LossOutputs:
        ...
"""


@jaxtyped(typechecker=typechecker)
@torch.compile(dynamic=True)
def selective_log_softmax(
    logits: Float[Tensor, "batch seq vocab"], index: Int[Tensor, "batch seq"]
) -> Float[Tensor, "batch seq"]:
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


@jaxtyped(typechecker=typechecker)
@torch.compile(dynamic=True)
def compute_entropy(shifted_logits: Float[Tensor, "batch seq vocab"]) -> Float[Tensor, "batch seq"]:
    with torch.no_grad():
        pd = torch.nn.functional.softmax(shifted_logits, dim=-1)
        entropy = torch.logsumexp(shifted_logits, dim=-1) - torch.sum(pd * shifted_logits, dim=-1)
    return entropy


@jaxtyped(typechecker=typechecker)
def shift_logits(
    logits: Float[Tensor, "batch seq vocab"], left_pad_logit: Float[Tensor, "batch 1 vocab"] | None = None
) -> Float[Tensor, "batch seq vocab"]:
    """Removes final token logits and adds a left pad logit for the first token."""
    # We drop the last logit because it corresponds to the next token that will be sampled but is not here yet
    batch, seq, vocab = logits.shape
    logits = logits[:, :-1, :]  # (batch, seq-1, vocab)
    if left_pad_logit is None:
        left_pad_logit = torch.zeros(batch, 1, vocab, device=logits.device, dtype=logits.dtype)  # (batch, 1, vocab)
    logits = torch.cat([left_pad_logit, logits], dim=1)  # (batch, seq, vocab)
    return logits


def shift_tensor_left(t: Float[Tensor, "batch seq"]) -> Float[Tensor, "batch seq"]:
    """Shifts the tensor one token to the left.

    Used to create labels from input_ids: labels[i] = input_ids[i+1].
    The last position is padded with 0 (a valid token index) since this value
    will be shifted off by shift_tensor_right and never used.
    """
    return torch.cat([t[:, 1:], torch.full((t.shape[0], 1), 0, device=t.device, dtype=t.dtype)], dim=1)


def shift_tensor_right(t: Float[Tensor, "batch seq"], pad_value: float | None = None) -> Float[Tensor, "batch seq"]:
    """Shifts the tensor one token to the right, prepending a padding value.

    Used to realign logprobs/entropy after computing with shifted labels.
    After shift: result[i] = t[i-1], result[0] = pad_value.
    This converts from "predict next token" convention to "probability of current token" convention.

    Args:
        t: Tensor to shift right
        pad_value: Value to use for position 0. If None, uses 0.0 for backward compatibility.
                   For logprobs, should be log(1/vocab_size) to represent uniform distribution.
                   For entropy, should be log(vocab_size) to represent maximum entropy.
    """
    if pad_value is None:
        pad_value = 0.0
    return torch.cat([torch.full((t.shape[0], 1), pad_value, device=t.device, dtype=t.dtype), t[:, :-1]], dim=1)


def _safe_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean of values over a boolean mask; returns 0 when mask is empty."""
    denom = torch.clamp_min(mask.sum(), 1)
    return values[mask].sum() / denom


def compute_importance_ratio_and_mismatch_kl(
    trainer_logprobs: Tensor, inference_logprobs: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    log_importance_ratio = trainer_logprobs - inference_logprobs
    importance_ratio = torch.exp(log_importance_ratio)
    mismatch_kl = importance_ratio - log_importance_ratio - 1
    return log_importance_ratio, importance_ratio, mismatch_kl


def default_loss_fn(inputs: LossInputs, loss_config: DefaultLossConfig) -> LossOutputs:
    """
    DPPO+KL loss for RL training, combining:
    - DPPO-Binary TV Loss (https://arxiv.org/pdf/2602.04879)
    - Kimi-K2.5 KL Loss (https://arxiv.org/pdf/2602.02276)

    The mask is conditioned on the advantage sign: for positive advantages,
    we mask tokens whose probability increased too much (trust region violation
    in the upweight direction); for negative advantages, we mask tokens whose
    probability decreased too much (trust region violation in the downweight
    direction).
    """
    trainer_logprobs = inputs.trainer_logprobs
    inference_logprobs = inputs.inference_logprobs
    advantages = inputs.advantages
    loss_mask = inputs.loss_mask

    log_importance_ratio, importance_ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
        trainer_logprobs, inference_logprobs
    )

    probs_diff = torch.exp(trainer_logprobs) - torch.exp(inference_logprobs)
    dppo_invalid_mask_high = probs_diff > loss_config.dppo_mask_high
    dppo_invalid_mask_low = probs_diff < -loss_config.dppo_mask_low
    positive_advantages = advantages > 0
    negative_advantages = advantages < 0
    dppo_invalid_mask = torch.where(positive_advantages, dppo_invalid_mask_high, dppo_invalid_mask_low)

    is_masked = dppo_invalid_mask
    is_masked_high = positive_advantages & dppo_invalid_mask_high
    is_masked_low = negative_advantages & dppo_invalid_mask_low
    drop_mask = loss_mask & is_masked
    keep_mask = loss_mask & ~is_masked

    advantages = loss_config.adv_tau * advantages
    pg_loss = keep_mask * advantages * importance_ratio
    kl_loss = loss_mask * log_importance_ratio**2
    loss = (-pg_loss + loss_config.kl_tau * kl_loss).sum()

    metrics = {
        "masked_mismatch_kl": _safe_mean(mismatch_kl, loss_mask & is_masked),  # all trainable, masked tokens
        "unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),  # all trainable, unmasked tokens
        "is_masked": _safe_mean(is_masked, loss_mask),
        "is_masked_low": _safe_mean(is_masked_low, loss_mask),
        "is_masked_high": _safe_mean(is_masked_high, loss_mask),
        "masked_advantage_positive": _safe_mean(positive_advantages, drop_mask),
        "masked_advantage_negative": _safe_mean(negative_advantages, drop_mask),
    }

    return LossOutputs(loss=loss, metrics=metrics)


def surprisal_entropy_loss_fn(
    inputs: LossInputs,
    *,
    entropy_coef: float = 0.02,
    dppo_mask_low: float = 0.2,
    dppo_mask_high: float = 0.2,
    adv_tau: float = 1.0,
    kl_tau: float = 1e-3,
) -> LossOutputs:
    """Default DPPO+KL loss PLUS a detached-surprisal ENTROPY-bonus policy-gradient term.

    Why (anti-collapse for the cross-prompt class MARGINAL): correctness-only GRPO with no
    entropy/reference term leaves the single-text marginal prediction prior unconstrained, so
    the favored class drifts (an under-predicted class can fall to ~5%). This adds an entropy
    incentive that REWARDS generating low-probability tokens, i.e. it up-weights the rare
    tokens — including the answer-label token of an under-predicted class — keeping that class
    sampled so it can keep receiving corrective gradient.

    Form (fused-LM-head compatible — uses ONLY sampled-token logprobs, no entropy backward,
    no teacher/reference model): treat per-token surprisal under the ROLLOUT policy,
    ``s_t = -inference_logprobs_t`` (>= 0, detached), as an extra advantage folded into the
    policy-gradient term:

        shaped_adv = adv_tau * advantage + entropy_coef * s_t
        pg_loss    = keep_mask * shaped_adv * importance_ratio

    Minimizing ``-pg_loss`` then INCREASES the trainer probability of high-surprisal (rare)
    tokens -> raises entropy. This is the standard surprisal/intrinsic-entropy PG estimator,
    not the (degenerate) ``+beta*logprob`` form: it correctly UP-weights rare actions instead
    of pushing the sampled token down, and at on-policy it is non-trivial (unlike the
    expectation-zero logprob term). Applied on ``keep_mask`` so it respects the DPPO trust
    region (never reintroduces gradient on trust-region-masked tokens). ``entropy_coef=0``
    reproduces ``default_loss_fn`` exactly.
    """
    trainer_logprobs = inputs.trainer_logprobs
    inference_logprobs = inputs.inference_logprobs
    advantages = inputs.advantages
    loss_mask = inputs.loss_mask

    log_importance_ratio, importance_ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
        trainer_logprobs, inference_logprobs
    )

    probs_diff = torch.exp(trainer_logprobs) - torch.exp(inference_logprobs)
    dppo_invalid_mask_high = probs_diff > dppo_mask_high
    dppo_invalid_mask_low = probs_diff < -dppo_mask_low
    positive_advantages = advantages > 0
    negative_advantages = advantages < 0
    dppo_invalid_mask = torch.where(positive_advantages, dppo_invalid_mask_high, dppo_invalid_mask_low)

    is_masked = dppo_invalid_mask
    is_masked_high = positive_advantages & dppo_invalid_mask_high
    is_masked_low = negative_advantages & dppo_invalid_mask_low
    drop_mask = loss_mask & is_masked
    keep_mask = loss_mask & ~is_masked

    # Detached surprisal under the rollout policy (>= 0): the per-token entropy incentive.
    surprisal = torch.clamp_min((-inference_logprobs).detach(), 0.0)
    shaped_advantages = adv_tau * advantages + entropy_coef * surprisal

    pg_loss = keep_mask * shaped_advantages * importance_ratio
    kl_loss = loss_mask * log_importance_ratio**2
    loss = (-pg_loss + kl_tau * kl_loss).sum()

    metrics = {
        "masked_mismatch_kl": _safe_mean(mismatch_kl, loss_mask & is_masked),
        "unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),
        "is_masked": _safe_mean(is_masked, loss_mask),
        "is_masked_low": _safe_mean(is_masked_low, loss_mask),
        "is_masked_high": _safe_mean(is_masked_high, loss_mask),
        "masked_advantage_positive": _safe_mean(positive_advantages, drop_mask),
        "masked_advantage_negative": _safe_mean(negative_advantages, drop_mask),
        "surprisal": _safe_mean(surprisal, keep_mask),
        "entropy_bonus": _safe_mean(entropy_coef * surprisal, keep_mask),
    }

    return LossOutputs(loss=loss, metrics=metrics)


def _entropy_decay_coef(
    entropy_coef: float,
    step: int,
    max_steps: int | None,
    hold_frac: float,
    end_frac: float,
    floor: float,
) -> float:
    """Schedule for the entropy bonus: hold the full coef early (to break the truncation/
    repetition loop and keep every class alive through the brevity breakthrough), then ramp it
    DOWN to ``floor`` so it cannot, late in training, push the policy BACK INTO the high-surprisal
    repetition loop (the Run-14 failure: unchecked entropy collapsed val 0.36->0.02 / truncation
    1.5%->91.5% over steps 28->40). Full coef while ``step/max_steps <= hold_frac``, linear decay
    to ``floor`` by ``end_frac``, ``floor`` thereafter. With ``max_steps=None`` the coef is constant
    (reproduces ``surprisal_entropy_loss_fn``)."""
    if max_steps is None or max_steps <= 0:
        return entropy_coef
    f = step / max_steps
    if f <= hold_frac:
        return entropy_coef
    if f >= end_frac or end_frac <= hold_frac:
        return floor
    return floor + (entropy_coef - floor) * (end_frac - f) / (end_frac - hold_frac)


def surprisal_entropy_decay_loss_fn(
    inputs: LossInputs,
    *,
    entropy_coef: float = 0.02,
    hold_frac: float = 0.4,
    end_frac: float = 0.8,
    floor: float = 0.0,
    dppo_mask_low: float = 0.2,
    dppo_mask_high: float = 0.2,
    adv_tau: float = 1.0,
    kl_tau: float = 1e-3,
    step: int = 0,
    max_steps: int | None = None,
) -> LossOutputs:
    """``surprisal_entropy_loss_fn`` with a DECAYING entropy coefficient (see ``_entropy_decay_coef``).

    Identical mechanics to the proven entropy recipe — detached-surprisal entropy bonus folded into
    the advantage on DPPO ``keep_mask`` tokens, plus Kimi KL — except ``entropy_coef`` is annealed
    over training via the ``(step, max_steps)`` the trainer now passes through ``compute_loss``. This
    keeps the anti-collapse / anti-truncation benefit of the bonus during the early/mid phase while
    removing it before the late over-training regime where it became destabilizing.
    """
    eff_coef = _entropy_decay_coef(entropy_coef, step, max_steps, hold_frac, end_frac, floor)

    trainer_logprobs = inputs.trainer_logprobs
    inference_logprobs = inputs.inference_logprobs
    advantages = inputs.advantages
    loss_mask = inputs.loss_mask

    log_importance_ratio, importance_ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
        trainer_logprobs, inference_logprobs
    )

    probs_diff = torch.exp(trainer_logprobs) - torch.exp(inference_logprobs)
    dppo_invalid_mask_high = probs_diff > dppo_mask_high
    dppo_invalid_mask_low = probs_diff < -dppo_mask_low
    positive_advantages = advantages > 0
    negative_advantages = advantages < 0
    dppo_invalid_mask = torch.where(positive_advantages, dppo_invalid_mask_high, dppo_invalid_mask_low)

    is_masked = dppo_invalid_mask
    is_masked_high = positive_advantages & dppo_invalid_mask_high
    is_masked_low = negative_advantages & dppo_invalid_mask_low
    drop_mask = loss_mask & is_masked
    keep_mask = loss_mask & ~is_masked

    surprisal = torch.clamp_min((-inference_logprobs).detach(), 0.0)
    shaped_advantages = adv_tau * advantages + eff_coef * surprisal

    pg_loss = keep_mask * shaped_advantages * importance_ratio
    kl_loss = loss_mask * log_importance_ratio**2
    loss = (-pg_loss + kl_tau * kl_loss).sum()

    metrics = {
        "masked_mismatch_kl": _safe_mean(mismatch_kl, loss_mask & is_masked),
        "unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),
        "is_masked": _safe_mean(is_masked, loss_mask),
        "is_masked_low": _safe_mean(is_masked_low, loss_mask),
        "is_masked_high": _safe_mean(is_masked_high, loss_mask),
        "surprisal": _safe_mean(surprisal, keep_mask),
        "entropy_coef": torch.tensor(float(eff_coef)),
        "entropy_bonus": _safe_mean(eff_coef * surprisal, keep_mask),
    }

    return LossOutputs(loss=loss, metrics=metrics)


def cispo_loss_fn(
    inputs: LossInputs,
    *,
    eps_low: float = 10.0,
    eps_high: float = 0.28,
    adv_tau: float = 1.0,
    kl_tau: float = 1e-3,
) -> LossOutputs:
    """CISPO loss (MiniMax-M1, https://arxiv.org/abs/2506.13585) + Kimi-style KL.

    Why (anti-collapse via gradient retention): PPO/DPPO-style trust regions DROP the
    gradient on any token whose probability moved too far (``keep_mask`` zeroes it). Those
    are exactly the rare, high-advantage tokens that matter most for lifting an
    under-represented class (e.g. the GEMINI answer-label token as the policy first starts
    to sample it). Killing their gradient at the moment they start to grow is what starves
    the abandoned class. CISPO never masks: it clips only the importance-sampling WEIGHT and
    stop-gradients it, so the per-token gradient ``sg(clip(rho)) * A * grad log pi`` FLOWS for
    every trainable token while the clipped weight still bounds the off-policy variance.

    Form (fused-LM-head compatible — uses ONLY sampled-token logprobs, no logits, no
    teacher/reference model). With ``rho_t = pi_theta / mu`` (trainer over rollout policy):

        rho_hat = clip(rho, 1 - eps_low, 1 + eps_high)            # stop-gradient
        pg_loss = loss_mask * sg(rho_hat) * (adv_tau * A) * log pi_theta

    Minimizing ``-pg_loss`` increases ``log pi_theta`` on positive-advantage tokens and
    decreases it on negative ones — the REINFORCE direction, weighted by the bounded IS
    weight. Following MiniMax, only the UPPER clip ``eps_high`` is active; ``eps_low`` is set
    large (default 10.0) so the lower clip never binds. The Kimi mismatch-KL term
    (``kl_tau * log_rho^2``) is kept for trainer/inference drift control, matching the other
    losses in this file. Unlike ``default_loss_fn`` there is no DPPO probability-difference
    mask, so no token is ever dropped from the gradient.
    """
    trainer_logprobs = inputs.trainer_logprobs
    inference_logprobs = inputs.inference_logprobs
    advantages = inputs.advantages
    loss_mask = inputs.loss_mask

    log_importance_ratio, importance_ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
        trainer_logprobs, inference_logprobs
    )

    # Clip the IS weight and stop-gradient it: bounds variance without deleting any update.
    clipped_ratio = torch.clamp(importance_ratio, 1.0 - eps_low, 1.0 + eps_high).detach()
    positive_advantages = advantages > 0
    negative_advantages = advantages < 0

    advantages = adv_tau * advantages
    # REINFORCE-style PG: gradient flows through log pi_theta (every trainable token), the
    # detached clipped ratio is just a bounded per-token weight.
    pg_loss = loss_mask * clipped_ratio * advantages * trainer_logprobs
    kl_loss = loss_mask * log_importance_ratio**2
    loss = (-pg_loss + kl_tau * kl_loss).sum()

    # Tokens whose true (un-stop-gradient) ratio left the trust region: kept here, dropped by DPPO.
    upper_clipped = (importance_ratio > 1.0 + eps_high) & loss_mask
    lower_clipped = (importance_ratio < 1.0 - eps_low) & loss_mask

    metrics = {
        "masked_mismatch_kl": _safe_mean(mismatch_kl, upper_clipped | lower_clipped),
        "unmasked_mismatch_kl": _safe_mean(mismatch_kl, loss_mask),
        "clip_frac_high": _safe_mean(upper_clipped, loss_mask),
        "clip_frac_low": _safe_mean(lower_clipped, loss_mask),
        "importance_ratio": _safe_mean(importance_ratio, loss_mask),
        "advantage_positive": _safe_mean(positive_advantages, loss_mask),
        "advantage_negative": _safe_mean(negative_advantages, loss_mask),
    }

    return LossOutputs(loss=loss, metrics=metrics)


def opd_loss_fn(inputs: LossInputs) -> LossOutputs:
    """
    On-policy distillation loss: the default DPPO+KL math with the tau knobs
    hardcoded to drop the reward signal and use the teacher KL as the
    per-token policy-gradient signal.
    """
    trainer_logprobs = inputs.trainer_logprobs
    inference_logprobs = inputs.inference_logprobs
    teacher_logprobs = inputs.teacher_logprobs
    advantages = inputs.advantages
    loss_mask = inputs.loss_mask

    if teacher_logprobs is None:
        raise ValueError("opd_loss_fn requires teacher_logprobs - configure a teacher for opd mode.")

    log_importance_ratio, importance_ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
        trainer_logprobs, inference_logprobs
    )

    probs_diff = torch.exp(trainer_logprobs) - torch.exp(inference_logprobs)
    dppo_invalid_mask_high = probs_diff > 0.2
    dppo_invalid_mask_low = probs_diff < -0.2
    positive_advantages = advantages > 0
    negative_advantages = advantages < 0
    dppo_invalid_mask = torch.where(positive_advantages, dppo_invalid_mask_high, dppo_invalid_mask_low)

    is_masked = dppo_invalid_mask
    is_masked_high = positive_advantages & dppo_invalid_mask_high
    is_masked_low = negative_advantages & dppo_invalid_mask_low
    drop_mask = loss_mask & is_masked
    keep_mask = loss_mask & ~is_masked

    teacher_kl = teacher_logprobs - trainer_logprobs
    advantages = 0.0 * advantages + 1.0 * teacher_kl.detach()

    pg_loss = keep_mask * advantages * importance_ratio
    kl_loss = loss_mask * log_importance_ratio**2
    loss = (-pg_loss + 1e-3 * kl_loss).sum()

    metrics = {
        "masked_mismatch_kl": _safe_mean(mismatch_kl, loss_mask & is_masked),
        "unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),
        "is_masked": _safe_mean(is_masked, loss_mask),
        "is_masked_low": _safe_mean(is_masked_low, loss_mask),
        "is_masked_high": _safe_mean(is_masked_high, loss_mask),
        "masked_advantage_positive": _safe_mean(positive_advantages, drop_mask),
        "masked_advantage_negative": _safe_mean(negative_advantages, drop_mask),
        "teacher_kl": _safe_mean(teacher_kl, loss_mask),
    }

    return LossOutputs(loss=loss, metrics=metrics)


def sft_loss_fn(inputs: LossInputs) -> LossOutputs:
    """SFT-style masked negative log-likelihood over trainable tokens."""
    trainer_logprobs = inputs.trainer_logprobs
    loss_mask = inputs.loss_mask

    loss = -(trainer_logprobs[loss_mask]).sum()
    metrics = {
        "nll": _safe_mean(-trainer_logprobs, loss_mask),
    }
    return LossOutputs(loss=loss, metrics=metrics)


def setup_loss_fns(loss_config: LossConfig) -> dict[str, LossFn]:
    """Build the per-training-mode loss fn dispatch table.

    Always returns all three modes - the trainer is mode-agnostic and routes
    per batch from ``TrainingSample.training_mode``:

    - ``"sft"`` → ``sft_loss_fn`` (masked NLL on teacher tokens)
    - ``"opd"`` → ``opd_loss_fn`` (teacher KL as gradient signal, hardcoded
      DPPO + KL knobs)
    - ``"rl"``  → ``default_loss_fn(loss_config)`` for ``DefaultLossConfig``,
      or the imported function for ``CustomLossConfig``.

    ``trainer.loss`` only affects the rl path - opd and sft are independent.
    """
    if isinstance(loss_config, CustomLossConfig):
        custom_fn = import_object(loss_config.import_path)
        kwargs = loss_config.kwargs

        # Forward dynamic per-step context (step, max_steps) ONLY to custom loss fns that declare
        # it (e.g. surprisal_entropy_decay_loss_fn), so existing custom losses are unaffected.
        _params = inspect.signature(custom_fn).parameters
        _has_varkw = any(p.kind == p.VAR_KEYWORD for p in _params.values())
        _accepts = set(_params)

        def rl_fn(inputs: LossInputs, **dyn) -> LossOutputs:
            fwd = dyn if _has_varkw else {k: v for k, v in dyn.items() if k in _accepts}
            return custom_fn(inputs, **kwargs, **fwd)
    else:

        def rl_fn(inputs: LossInputs, **dyn) -> LossOutputs:
            return default_loss_fn(inputs, loss_config)

    return {"sft": sft_loss_fn, "opd": opd_loss_fn, "rl": rl_fn}


def compute_loss(
    trainer_logprobs: list[Float[Tensor, " seq_i"]],
    inference_logprobs: list[Float[Tensor, " seq_i"]],
    teacher_logprobs: list[Float[Tensor, " seq_i"]] | None,
    advantages: list[Float[Tensor, " seq_i"]],
    loss_mask: list[Bool[Tensor, " seq_i"]],
    loss_fns: dict[str, LossFn],
    loss_scale: int,
    training_mode: str = "rl",
    step: int = 0,
    max_steps: int | None = None,
) -> tuple[Float[Tensor, ""], dict[str, Any]]:
    """
    Compute loss for packed sequences (batch size = 1, multiple sequences packed along sequence dimension).

    Loss dispatch is batch-driven: ``training_mode`` selects the loss fn from
    ``loss_fns`` (built by ``setup_loss_fns``). sft → sft_loss_fn, opd →
    opd_loss_fn, rl → the configured default/custom loss.

    Args:
        trainer_logprobs: Log probabilities for each sequence
        inference_logprobs: Reference log probabilities for each sequence
        teacher_logprobs: Teacher log probabilities for each sequence, or None
        advantages: Advantages for each sequence
        loss_mask: Loss mask for each sequence
        loss_fns: Per-mode loss fn dispatch table from setup_loss_fns()
        loss_scale: Scale factor to normalize the loss
        training_mode: Selects which loss fn to apply

    Returns:
        Tuple of (scaled_loss, aggregated_metrics)
    """
    try:
        effective_loss_fn = loss_fns[training_mode]
    except KeyError:
        raise ValueError(
            f"No loss fn available for training_mode={training_mode!r} "
            f"(available: {sorted(loss_fns)}). Check trainer.loss.type."
        )

    total_loss = 0.0
    all_metrics: dict[str, list[Tensor]] = {}

    if teacher_logprobs is None:
        teacher_logprobs = [None] * len(trainer_logprobs)

    for t_logp, i_logp, teach_logp, adv, mask in zip(
        trainer_logprobs,
        inference_logprobs,
        teacher_logprobs,
        advantages,
        loss_mask,
    ):
        inputs = LossInputs(
            trainer_logprobs=t_logp,
            inference_logprobs=i_logp,
            teacher_logprobs=teach_logp,
            advantages=adv,
            loss_mask=mask,
        )

        result = effective_loss_fn(inputs, step=step, max_steps=max_steps) if training_mode == "rl" else effective_loss_fn(inputs)

        total_loss = total_loss + result.loss

        for k, v in result.metrics.items():
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append(v)

    scaled_loss = total_loss / loss_scale

    aggregated: dict[str, Any] = {}
    for k, v in all_metrics.items():
        if v[0].dim() == 0:
            aggregated[k] = torch.stack(v)
        else:
            aggregated[k] = torch.cat(v)

    return scaled_loss, aggregated
