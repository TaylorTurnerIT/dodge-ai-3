"""Pure, JSON-safe evaluation and episode metrics.

The training loop owns environment interaction and decides when to call these
helpers.  This module only pairs already-completed evaluation outcomes and
accumulates one episode at a time.  In particular, an episode record is
returned only when ``terminated`` or ``truncated`` is true; a caller cannot
mistake a logging snapshot of an in-progress episode for a completed return.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import fsum, isfinite, sqrt
from typing import TypeAlias, TypedDict

SeedReward: TypeAlias = tuple[int, float]
RewardSamples: TypeAlias = Mapping[int, float] | Iterable[SeedReward]


class PairedRewardRow(TypedDict):
    """One baseline/counterfactual outcome on one identical seed."""

    seed: int
    baseline_reward: float
    counterfactual_reward: float
    delta: float


class PairedRewardSummary(TypedDict):
    """Descriptive statistics for a paired reward comparison.

    Standard deviations are population standard deviations over the supplied
    evaluation seeds.  They describe this explicit set of scenarios rather
    than estimating training-seed uncertainty.
    """

    count: int
    baseline_mean: float
    baseline_std: float
    counterfactual_mean: float
    counterfactual_std: float
    mean_delta: float
    delta_std: float
    min_delta: float
    max_delta: float
    counterfactual_better_count: int
    counterfactual_worse_count: int
    tie_count: int
    counterfactual_better_share: float


class PairedRewardReport(TypedDict):
    """Versioned, JSON-safe result of :func:`paired_reward_deltas`."""

    protocol: str
    rows: list[PairedRewardRow]
    summary: PairedRewardSummary


class EpisodeReturnRecord(TypedDict):
    """A complete episode outcome emitted at termination or truncation."""

    episode: int
    episode_return: float
    episode_steps: int
    survival_frames: int
    terminated: bool
    truncated: bool


@dataclass(frozen=True, slots=True)
class EpisodeReturnState:
    """Immutable accumulator state between environment decisions.

    ``episode`` is zero-based by default.  A non-terminal step returns a new
    state and no record.  A terminal step returns the completed record and a
    reset state for the next episode.  If a run stops before termination, the
    caller simply has no record for that partial episode.
    """

    episode: int = 0
    episode_return: float = 0.0
    episode_steps: int = 0
    survival_frames: int = 0


def _as_seed(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} seed must be an integer")
    try:
        seed = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} seed must be an integer") from error
    return int(seed)


def _as_finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise TypeError(f"{name} must be a finite number")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be a finite number") from error
    if not isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _as_nonnegative_int(value: object, *, name: str) -> int:
    converted = _as_seed(value, name=name)
    if converted < 0:
        raise ValueError(f"{name} must be non-negative")
    return converted


def _normalize_samples(
    samples: RewardSamples,
    *,
    name: str,
) -> dict[int, float]:
    """Normalize seed/reward samples while rejecting ambiguous duplicates."""

    items: Iterable[tuple[object, object]] = (
        samples.items() if isinstance(samples, Mapping) else samples
    )

    normalized: dict[int, float] = {}
    for item in items:
        try:
            raw_seed, raw_reward = item
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{name} samples must contain (seed, reward) pairs"
            ) from error
        seed = _as_seed(raw_seed, name=name)
        if seed in normalized:
            raise ValueError(f"{name} contains duplicate seed {seed}")
        normalized[seed] = _as_finite_float(raw_reward, name=f"{name} reward")
    return normalized


def _population_stats(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = fsum(values) / len(values)
    variance = fsum((value - mean) ** 2 for value in values) / len(values)
    result = (float(mean), float(sqrt(variance)))
    if not all(isfinite(value) for value in result):
        raise ValueError("reward summary is not finite")
    return result


def paired_reward_deltas(
    baseline: RewardSamples,
    counterfactual: RewardSamples,
) -> PairedRewardReport:
    """Pair baseline and forced-action rewards by identical evaluation seed.

    ``baseline`` and ``counterfactual`` may be seed-keyed mappings or
    iterables of ``(seed, reward)`` pairs.  The seed sets must match exactly;
    silently zipping differently ordered or differently sized evaluations
    would make the resulting deltas uninterpretable.
    """

    baseline_by_seed = _normalize_samples(baseline, name="baseline")
    counterfactual_by_seed = _normalize_samples(
        counterfactual, name="counterfactual"
    )
    if baseline_by_seed.keys() != counterfactual_by_seed.keys():
        missing_from_counterfactual = sorted(
            baseline_by_seed.keys() - counterfactual_by_seed.keys()
        )
        missing_from_baseline = sorted(
            counterfactual_by_seed.keys() - baseline_by_seed.keys()
        )
        raise ValueError(
            "baseline and counterfactual seeds must match exactly; "
            f"missing_from_counterfactual={missing_from_counterfactual}, "
            f"missing_from_baseline={missing_from_baseline}"
        )

    rows: list[PairedRewardRow] = []
    for seed in sorted(baseline_by_seed):
        baseline_reward = baseline_by_seed[seed]
        counterfactual_reward = counterfactual_by_seed[seed]
        delta = _as_finite_float(
            counterfactual_reward - baseline_reward,
            name="reward delta",
        )
        rows.append(
            {
                "seed": int(seed),
                "baseline_reward": float(baseline_reward),
                "counterfactual_reward": float(counterfactual_reward),
                "delta": delta,
            }
        )

    baseline_values = [row["baseline_reward"] for row in rows]
    counterfactual_values = [row["counterfactual_reward"] for row in rows]
    deltas = [row["delta"] for row in rows]
    baseline_mean, baseline_std = _population_stats(baseline_values)
    counterfactual_mean, counterfactual_std = _population_stats(
        counterfactual_values
    )
    mean_delta, delta_std = _population_stats(deltas)
    count = len(rows)
    better_count = sum(delta > 0.0 for delta in deltas)
    worse_count = sum(delta < 0.0 for delta in deltas)
    tie_count = count - better_count - worse_count
    summary: PairedRewardSummary = {
        "count": count,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "counterfactual_mean": counterfactual_mean,
        "counterfactual_std": counterfactual_std,
        "mean_delta": mean_delta,
        "delta_std": delta_std,
        "min_delta": min(deltas) if deltas else 0.0,
        "max_delta": max(deltas) if deltas else 0.0,
        "counterfactual_better_count": better_count,
        "counterfactual_worse_count": worse_count,
        "tie_count": tie_count,
        "counterfactual_better_share": better_count / count if count else 0.0,
    }
    return {
        "protocol": "paired-reward-delta-v1",
        "rows": rows,
        "summary": summary,
    }


def _evaluation_samples(
    evaluation: Mapping[str, object],
    *,
    name: str,
) -> list[SeedReward]:
    seeds = evaluation.get("seeds")
    rewards = evaluation.get("rewards")
    if (
        isinstance(seeds, (str, bytes))
        or isinstance(rewards, (str, bytes))
        or not isinstance(seeds, Sequence)
        or not isinstance(rewards, Sequence)
    ):
        raise TypeError(f"{name} evaluation must contain seed/reward sequences")
    if len(seeds) != len(rewards):
        raise ValueError(f"{name} seeds and rewards must have equal lengths")
    return list(zip(seeds, rewards, strict=True))  # type: ignore[return-value]


def paired_evaluation_deltas(
    baseline_evaluation: Mapping[str, object],
    counterfactual_evaluation: Mapping[str, object],
) -> PairedRewardReport:
    """Pair the ``seeds``/``rewards`` fields of two evaluation sections."""

    return paired_reward_deltas(
        _evaluation_samples(baseline_evaluation, name="baseline"),
        _evaluation_samples(counterfactual_evaluation, name="counterfactual"),
    )


def record_episode_step(
    state: EpisodeReturnState,
    *,
    reward: float,
    terminated: bool,
    truncated: bool,
    survival_frames: int = 0,
) -> tuple[EpisodeReturnState, EpisodeReturnRecord | None]:
    """Accumulate one step and emit a record only at episode end.

    The returned state is reset to an empty next episode after an end signal.
    Both Gymnasium end signals are retained in the record; either one is
    sufficient to mark the return complete.  No flush operation is provided,
    so a run that stops mid-episode cannot publish a partial return by using
    this helper.
    """

    if not isinstance(state, EpisodeReturnState):
        raise TypeError("state must be an EpisodeReturnState")
    if not isinstance(terminated, bool) or not isinstance(truncated, bool):
        raise TypeError("terminated and truncated must be booleans")
    episode = _as_nonnegative_int(state.episode, name="state episode")
    episode_steps = _as_nonnegative_int(
        state.episode_steps,
        name="state episode_steps",
    )
    accumulated_frames = _as_nonnegative_int(
        state.survival_frames,
        name="state survival_frames",
    )
    episode_return = _as_finite_float(
        state.episode_return,
        name="state episode return",
    )
    step_reward = _as_finite_float(reward, name="reward")
    frame_delta = _as_nonnegative_int(survival_frames, name="survival_frames")
    total_return = _as_finite_float(
        episode_return + step_reward,
        name="episode return",
    )
    total_steps = episode_steps + 1
    total_frames = accumulated_frames + frame_delta
    if not (terminated or truncated):
        return (
            EpisodeReturnState(
                episode=episode,
                episode_return=total_return,
                episode_steps=total_steps,
                survival_frames=total_frames,
            ),
            None,
        )

    record: EpisodeReturnRecord = {
        "episode": episode,
        "episode_return": total_return,
        "episode_steps": int(total_steps),
        "survival_frames": int(total_frames),
        "terminated": terminated,
        "truncated": truncated,
    }
    return EpisodeReturnState(episode=episode + 1), record


__all__ = [
    "EpisodeReturnRecord",
    "EpisodeReturnState",
    "PairedRewardReport",
    "PairedRewardRow",
    "PairedRewardSummary",
    "RewardSamples",
    "SeedReward",
    "paired_evaluation_deltas",
    "paired_reward_deltas",
    "record_episode_step",
]
