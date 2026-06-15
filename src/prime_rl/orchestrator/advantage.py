from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch
import verifiers as vf
from jaxtyping import Float
from torch import Tensor

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import TrainRollout

from prime_rl.configs.orchestrator import (
    AdvantageConfig,
    CustomAdvantageConfig,
    LengthPenaltyConfig,
    TokensLengthPenaltyConfig,
    TurnsLengthPenaltyConfig,
)
from prime_rl.orchestrator.utils import get_model_completion_len, get_tool_response_len
from prime_rl.utils.utils import import_object


@dataclass
class AdvantageInputs:
    """Inputs for advantage computation of a single group (one example × N rollouts)."""

    rollouts: list[vf.RolloutOutput]


@dataclass
class AdvantageOutputs:
    """Outputs from advantage computation of a single group."""

    advantages: list[float]


AdvantageFn = Callable[..., AdvantageOutputs]
"""Type for an advantage function.

Expected signature:
    def my_advantage(inputs: AdvantageInputs, **kwargs) -> AdvantageOutputs:
        ...

The function receives a single group and returns a list of advantages with one
entry per rollout. `assign_advantages` calls it on one already-grouped cohort.
"""


def default_advantage_fn(
    inputs: AdvantageInputs,
    length_penalty: LengthPenaltyConfig | None = None,
) -> AdvantageOutputs:
    """Default GRPO advantage for a single group: reward minus per-group baseline.

    `length_penalty` enables correctness-gated efficiency shaping over a per-rollout
    cost: tokens (weighted completion + tool-response) or trajectory turn count.
    """
    rewards = torch.tensor([r["reward"] for r in inputs.rollouts], dtype=torch.float32)

    if isinstance(length_penalty, TokensLengthPenaltyConfig):
        w_c = length_penalty.completion_weight
        w_t = length_penalty.tool_response_weight
        costs = torch.tensor(
            [w_c * get_model_completion_len(r) + w_t * get_tool_response_len(r) for r in inputs.rollouts],
            dtype=rewards.dtype,
        )
        return AdvantageOutputs(advantages=_efficiency_shaping(rewards, costs).tolist())
    if isinstance(length_penalty, TurnsLengthPenaltyConfig):
        costs = torch.tensor([len(r["trajectory"]) for r in inputs.rollouts], dtype=rewards.dtype)
        return AdvantageOutputs(advantages=_efficiency_shaping(rewards, costs).tolist())

    return AdvantageOutputs(advantages=(rewards - rewards.mean()).tolist())


def _efficiency_shaping(
    rewards: Float[Tensor, "group_size"],
    costs: Float[Tensor, "group_size"],
) -> Float[Tensor, "group_size"]:
    """Correctness-gated efficiency shaping with bounded advantages.

    Shapes rewards with a bounded efficiency bonus before standard GRPO subtraction,
    preserving zero-mean advantages within the group. `costs` is a per-rollout cost
    (e.g., completion length in tokens or number of turns).

    Correct rollouts get reward amplified by up to 2x based on relative efficiency.
    Incorrect rollouts are untouched. Lower-cost correct rollouts get higher advantage.
    """
    max_reward = rewards.max()
    correct_mask = rewards >= max_reward
    num_correct = correct_mask.sum()

    # No shaping when max reward is 0 — no correct rollouts to differentiate
    if max_reward <= 0:
        return rewards - rewards.mean()

    # Mean cost of correct rollouts
    mean_correct_cost = (costs * correct_mask).sum() / num_correct.clamp(min=1)

    # Bounded efficiency bonus: [0, 1], positive for below-average cost, zero for above.
    # When mean_correct_cost is 0 (e.g. tool-only shaping with no harness metric, or
    # all-zero turn counts), no rollouts can be differentiated — fall back to no bonus.
    if mean_correct_cost <= 0:
        return rewards - rewards.mean()

    bonus = (1 - costs / mean_correct_cost).clamp(0, 1)

    # Shape rewards: correct rollouts amplified by up to 2x, incorrect untouched
    shaped_rewards = rewards * (1 + bonus * correct_mask)
    return shaped_rewards - shaped_rewards.mean()


def maxrl_advantage(
    inputs: AdvantageInputs,
    *,
    success_threshold: float = 0.5,
    adv_clip: float = 6.0,
    eps: float = 1e-8,
) -> AdvantageOutputs:
    """MaxRL advantage (Tajwar et al. 2026, "Maximum Likelihood RL").

    Why (pass@k + hard-class emphasis): standard GRPO optimizes expected reward (pass@1) and
    spreads signal evenly, so easy prompts (CLAUDE/CHATGPT) dominate and the hard class
    (GEMINI) is under-weighted exactly when it most needs reinforcement. MaxRL's on-policy
    estimator averages the score functions of the SUCCESSFUL rollouts only; in zero-mean
    control-variate form its effective per-rollout advantage is

        A_i  ∝  (r_i - r_hat) / r_hat,     r_hat = K / N   (K = #successes, N = group size)

    so a success on a prompt the model rarely solves (small r_hat) is up-weighted strongly,
    while near-solved prompts (r_hat → 1) get little extra push. This provably approximates
    maximum-likelihood training (an infinite harmonic mixture of pass@k gradients, not just
    pass@1), improves pass@k, and preserves output diversity better than GRPO — i.e. it
    pushes against marginal collapse rather than toward it.

    Implementation notes:
    - Binary reward: a rollout is a "success" iff ``reward > success_threshold``.
    - K == 0 (no success in the group) → all-zero advantages. MaxRL, like every outcome-based
      method, provides no gradient on a prompt the model never solves; the curriculum is what
      supplies the first GEMINI successes that this estimator then amplifies.
    - The form is zero-mean within the group (a valid baseline). ``adv_clip`` caps the
      magnitude on ultra-hard prompts (r_hat = 1/N gives a raw success advantage of N-1) to
      keep the update inside a sane range; it is symmetric so it barely perturbs the baseline.
    - Scale is O(1) (the proportional ``(r_i - r_hat)/r_hat`` form), matching the default GRPO
      advantage so the existing LR / trust region transfer without retuning.
    """
    rewards = [float(r["reward"]) for r in inputs.rollouts]
    n = len(rewards)
    if n == 0:
        return AdvantageOutputs(advantages=[])
    k = sum(1 for r in rewards if r > success_threshold)
    if k == 0:
        return AdvantageOutputs(advantages=[0.0] * n)
    r_hat = k / n
    advantages = []
    for r in rewards:
        a = (r - r_hat) / (r_hat + eps)
        a = max(-adv_clip, min(adv_clip, a))
        advantages.append(a)
    # Re-center after clipping so the group stays zero-mean (a valid baseline). Clipping the
    # raw (r - r_hat)/r_hat on ultra-hard prompts (e.g. n=12,k=1: [+11 -> +6, -1, ...]) would
    # otherwise leave a net-negative group sum, biasing the update against the very rare-success
    # groups we want to amplify. Re-centering restores the clean control-variate form.
    mean_a = sum(advantages) / n
    advantages = [a - mean_a for a in advantages]
    return AdvantageOutputs(advantages=advantages)


def truncation_penalty_advantage(
    inputs: AdvantageInputs,
    *,
    truncation_penalty: float = 0.5,
) -> AdvantageOutputs:
    """Default Dr.GRPO advantage PLUS a mild, additive penalty on TRUNCATED rollouts.

    Why (anti-runaway, deliverable-safe): truncation here is a genuine pathology — a degenerate
    repetition loop that burns the whole completion budget WITHOUT emitting ``<answer>`` (reward
    0). The entropy recipe already drives truncation 38%->~2.5%, but the few remaining runaway
    rollouts still waste decode and risk re-seeding the loop. This adds a small NEGATIVE advantage
    to exactly those rollouts so the policy is pushed off the runaway trajectory.

    Crucially this is NOT a general length/overlong penalty (which would push the model toward
    ever-shorter outputs and accelerate the empty-``<reason_why>`` reward hack). It fires ONLY on
    ``is_truncated`` rollouts. The brevity-vs-substance balance is owned by the reason-gated reward,
    not by this term.

    Implementation (rubber-duck-reviewed): start from the standard zero-mean GRPO advantage
    ``r - r.mean()``, then ADD ``-truncation_penalty`` to truncated rollouts WITHOUT re-centering.
    Not re-centering is deliberate: re-centering would raise every other rollout's advantage by a
    positive constant, leaking POSITIVE advantage onto short-but-wrong rollouts (reinforcing quick
    wrong/invalid answers as an escape from the loop). With the additive form, in an all-wrong group
    (base advantages all 0) only the truncated rollouts go negative and every other rollout stays at
    0 — so no incorrect rollout is ever positively reinforced, and zero-advantage filtering of fully
    non-truncated all-wrong groups is preserved.
    """
    rewards = torch.tensor([float(r["reward"]) for r in inputs.rollouts], dtype=torch.float32)
    base = rewards - rewards.mean()
    penalties = torch.tensor(
        [-truncation_penalty if bool(r.get("is_truncated", False)) else 0.0 for r in inputs.rollouts],
        dtype=torch.float32,
    )
    return AdvantageOutputs(advantages=(base + penalties).tolist())


def setup_advantage_fn(config: AdvantageConfig) -> AdvantageFn:
    """Setup advantage function from config."""
    if isinstance(config, CustomAdvantageConfig):
        custom_fn = import_object(config.import_path)
        kwargs = config.kwargs

        def advantage_fn(inputs: AdvantageInputs) -> AdvantageOutputs:
            return custom_fn(inputs, **kwargs)

        return advantage_fn

    def advantage_fn(inputs: AdvantageInputs) -> AdvantageOutputs:
        return default_advantage_fn(inputs, length_penalty=config.length_penalty)

    return advantage_fn


def assign_advantages(
    rollouts: list["TrainRollout"],  # noqa: F821 (forward ref)
    advantage_fn: AdvantageFn | None,
) -> None:
    """Compute and assign advantages for one finished group of rollouts
    (``TrainSink.process_group`` hands in a single group's surviving rollouts).
    ``advantage_fn=None`` is the trivial case (advantage = reward); a custom
    ``advantage_fn`` receives the raw ``vf.RolloutOutput``\\ s via
    ``AdvantageInputs.rollouts``.
    """
    if advantage_fn is None:
        for rollout in rollouts:
            rollout.advantage = rollout.reward
        return
    result = advantage_fn(AdvantageInputs(rollouts=[r.raw for r in rollouts]))
    for rollout, advantage in zip(rollouts, result.advantages):
        rollout.advantage = advantage
