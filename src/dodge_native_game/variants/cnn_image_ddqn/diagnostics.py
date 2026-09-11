"""Pure diagnostics for the CNN image Double-DQN variant.

All helpers here are NumPy-only and side-effect free so they can be unit
tested without a native lane or a torch model. The training loop in
``run.py`` owns wiring; this module only computes numbers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

PASS_MIN_STEPS = 5000
LOCK_IN_BALANCE_THRESHOLD = 0.05
SPARSE_ZERO_SHARE_THRESHOLD = 0.95
GENERALIZATION_GAP_THRESHOLD = 0.5
# Sources that run the pure greedy policy. The counterfactual probe forces
# the least-used action on move one, so it is off-policy and never ranks.
GREEDY_EVAL_SOURCES = ("inner", "holdout", "eval")

__all__ = [
    "GENERALIZATION_GAP_THRESHOLD",
    "GREEDY_EVAL_SOURCES",
    "LOCK_IN_BALANCE_THRESHOLD",
    "PASS_MIN_STEPS",
    "SPARSE_ZERO_SHARE_THRESHOLD",
    "action_balance_ratio",
    "action_histogram",
    "best_greedy_episode",
    "dead_fraction",
    "decide_gate",
    "effective_decay_steps",
    "eval_seed_list",
    "greedy_eval_rows",
    "holdout_eval_seeds",
    "inner_eval_seeds",
    "q_spread_stats",
    "reward_mix_stats",
    "td_error_stats",
]


def reward_mix_stats(rewards: Sequence[float]) -> dict[str, float]:
    """Describe how sparse a reward window is.

    Returns zero-share, mean over all steps, mean over non-zero steps, and
    standard deviation. Empty input yields zeros instead of raising so the
    training loop can log before the first step completes.
    """

    values = np.asarray(list(rewards), dtype=np.float64)
    if values.size == 0:
        return {
            "zero_share": 1.0,
            "mean": 0.0,
            "mean_nonzero": 0.0,
            "std": 0.0,
        }
    zero_share = float(np.mean(values == 0.0))
    nonzero = values[values != 0.0]
    return {
        "zero_share": zero_share,
        "mean": float(values.mean()),
        "mean_nonzero": float(nonzero.mean()) if nonzero.size else 0.0,
        "std": float(values.std()) if values.size > 1 else 0.0,
    }


def td_error_stats(
    chosen_q: Sequence[float], targets: Sequence[float]
) -> dict[str, float]:
    """Mean and spread of the TD error ``targets - chosen_q``."""

    chosen = np.asarray(list(chosen_q), dtype=np.float64)
    aimed = np.asarray(list(targets), dtype=np.float64)
    if chosen.size == 0 or chosen.shape != aimed.shape:
        return {"mean": 0.0, "std": 0.0}
    errors = aimed - chosen
    return {
        "mean": float(errors.mean()),
        "std": float(errors.std()) if errors.size > 1 else 0.0,
    }


def q_spread_stats(q_values: Sequence[Sequence[float]]) -> dict[str, float]:
    """Summarize a (N, A) Q-value array without storing the full matrix."""

    values = np.asarray(q_values, dtype=np.float64)
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "gap": 0.0}
    if values.ndim == 1:
        values = values.reshape(1, -1)
    action_means = values.mean(axis=0)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()) if values.size > 1 else 0.0,
        "gap": float(action_means.max() - action_means.min())
        if action_means.size
        else 0.0,
    }


def action_histogram(actions: Sequence[int], num_actions: int) -> dict[str, object]:
    """Count action use and flag lock-in risk via min/max balance."""

    counts = [0] * max(1, int(num_actions))
    for action in actions:
        index = int(action)
        if 0 <= index < len(counts):
            counts[index] += 1
    total = sum(counts)
    ratio = action_balance_ratio(counts)
    least_used = min(range(len(counts)), key=lambda i: counts[i])
    return {
        "counts": counts,
        "total": total,
        "balance_ratio": float(ratio),
        "least_used_action": int(least_used),
    }


def action_balance_ratio(counts: Sequence[int]) -> float:
    """Balance ratio for already-aggregated counts; 0 for an unused action."""

    values = [int(value) for value in counts]
    nonzero = [value for value in values if value > 0]
    if not nonzero:
        return 0.0
    if any(value == 0 for value in values):
        return 0.0
    return float(min(nonzero) / max(nonzero))


def dead_fraction(activations: Sequence[float] | object) -> float:
    """Share of activations that are exactly zero (dead ReLU probe)."""

    values = np.asarray(activations)
    if values.size == 0:
        return 0.0
    return float(np.mean(values == 0.0))


def eval_seed_list(base_seed: int, count: int, *, offset: int) -> list[int]:
    """Deterministic eval seeds clamped to the native seed range."""

    seeds: list[int] = []
    for index in range(max(0, int(count))):
        seeds.append(min(32_767, int(base_seed) + int(offset) + index))
    return seeds


def inner_eval_seeds(base_seed: int, count: int) -> list[int]:
    """Seeds historically used by ``run._evaluate`` (offset 10_000)."""

    return eval_seed_list(base_seed, count, offset=10_000)


def holdout_eval_seeds(base_seed: int, count: int) -> list[int]:
    """Report-only holdout seeds (offset 20_000, disjoint from inner)."""

    return eval_seed_list(base_seed, count, offset=20_000)


def greedy_eval_rows(evaluation: Mapping[str, object]) -> list[dict[str, object]]:
    """Pool greedy-policy eval episodes into comparable rows.

    Only inner + holdout seeds (plus the legacy flat ``eval`` layout) count:
    every row ran the final checkpoint's pure greedy policy, so rows compare
    like with like. Each row carries source, seed, eval_reward, and
    survival_frames (0 when the section did not record any).
    """

    rows: list[dict[str, object]] = []
    for source in GREEDY_EVAL_SOURCES:
        section = evaluation.get(source)
        if not isinstance(section, Mapping):
            continue
        seeds = section.get("seeds")
        rewards = section.get("rewards")
        survival = section.get("survival_frames", [])
        if not isinstance(seeds, Sequence) or not isinstance(rewards, Sequence):
            continue
        for index, (seed, reward) in enumerate(zip(seeds, rewards, strict=False)):
            try:
                rows.append(
                    {
                        "source": str(source),
                        "seed": int(seed),
                        "eval_reward": float(reward),
                        "survival_frames": int(survival[index])
                        if index < len(survival)
                        else 0,
                    }
                )
            except (TypeError, ValueError):
                continue
    return rows


def _episode_rank(row: Mapping[str, object]) -> tuple[float, float, float]:
    """Order episodes by reward, then survival, then lowest seed first.

    Higher reward wins; longer survival breaks reward ties; the lower seed
    wins any remaining tie so rankings are deterministic. ``max`` over this
    key is the run's best final episode.
    """

    return (
        float(row.get("eval_reward", 0.0)),
        float(row.get("survival_frames", 0)),
        -float(row.get("seed", 0)),
    )


def best_greedy_episode(
    evaluation: Mapping[str, object],
) -> dict[str, object] | None:
    """Return the run's best final episode, or None when there is none.

    This is the reported "best": one greedy-policy episode from the frozen
    final evaluation of the saved checkpoint. It deliberately ignores the
    training-curve maximum, which was collected under exploration by an
    older, unsaved policy and can never be replayed.
    """

    rows = greedy_eval_rows(evaluation)
    if not rows:
        return None
    return dict(max(rows, key=_episode_rank))


def effective_decay_steps(configured: int, steps: int) -> int:
    """Return the validated global epsilon schedule.

    ``steps`` remains in the signature for compatibility with existing
    callers, but segment length must not implicitly rescale the schedule.
    Bounded runs that need faster exploitation must pass an explicit shorter
    configured schedule. Always returns at least 1.
    """

    del steps
    return max(1, int(configured))


def decide_gate(
    *,
    steps: int,
    warmup_steps: int,
    target_sync_interval: int,
    decay_configured: int,
    decay_effective: int,
    balance_ratio: float,
    zero_share: float,
    train_mean: float,
    holdout_mean: float | None,
    updates: int,
    target_sync_count: int,
    evaluation_censored_share: float = 0.0,
) -> tuple[str, list[str]]:
    """Pick a quality gate and human-readable reasons.

    Small bounded runs always stay at ``warn`` with reason
    ``bounded-smoke``; ``pass`` requires ``PASS_MIN_STEPS`` steps and no
    warning reasons. ``fail`` is reserved for callers handling exceptions.
    """

    reasons: list[str] = []
    if int(warmup_steps) > int(steps):
        reasons.append("warmup-no-updates")
    elif int(updates) == 0 and int(steps) >= int(warmup_steps):
        reasons.append("no-optimizer-updates")
    if int(target_sync_count) <= 0:
        reasons.append("target-never-synced")
    if int(decay_configured) > int(decay_effective):
        reasons.append("epsilon-decay-scaled")
    if float(zero_share) >= SPARSE_ZERO_SHARE_THRESHOLD:
        reasons.append("sparse-reward")
    if int(steps) > int(warmup_steps) and float(balance_ratio) < (
        LOCK_IN_BALANCE_THRESHOLD
    ):
        reasons.append("lock-in-risk")
    if holdout_mean is not None:
        denominator = abs(float(train_mean)) + 1.0
        gap = abs(float(train_mean) - float(holdout_mean)) / denominator
        if gap > GENERALIZATION_GAP_THRESHOLD:
            reasons.append("generalization-gap")
    if float(evaluation_censored_share) > 0.0:
        reasons.append("evaluation-censored")
    if int(steps) < PASS_MIN_STEPS:
        reasons.append("bounded-smoke")
        return "warn", reasons
    if reasons:
        return "warn", reasons
    return "pass", ["steady"]
