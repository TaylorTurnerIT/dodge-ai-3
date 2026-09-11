from __future__ import annotations

import json

import pytest

from dodge_native_game.variants.cnn_image_ddqn.episode_metrics import (
    EpisodeReturnState,
    paired_evaluation_deltas,
    paired_reward_deltas,
    record_episode_step,
)


def test_paired_rewards_are_seed_aligned_and_json_safe() -> None:
    report = paired_reward_deltas(
        [(12, 10.0), (10, 4.0), (11, 7.0)],
        {11: 8.0, 12: 9.0, 10: 4.0},
    )

    assert report["protocol"] == "paired-reward-delta-v1"
    assert report["rows"] == [
        {
            "seed": 10,
            "baseline_reward": 4.0,
            "counterfactual_reward": 4.0,
            "delta": 0.0,
        },
        {
            "seed": 11,
            "baseline_reward": 7.0,
            "counterfactual_reward": 8.0,
            "delta": 1.0,
        },
        {
            "seed": 12,
            "baseline_reward": 10.0,
            "counterfactual_reward": 9.0,
            "delta": -1.0,
        },
    ]
    assert report["summary"] == {
        "count": 3,
        "baseline_mean": 7.0,
        "baseline_std": 2.449489742783178,
        "counterfactual_mean": 7.0,
        "counterfactual_std": 2.160246899469287,
        "mean_delta": 0.0,
        "delta_std": 0.816496580927726,
        "min_delta": -1.0,
        "max_delta": 1.0,
        "counterfactual_better_count": 1,
        "counterfactual_worse_count": 1,
        "tie_count": 1,
        "counterfactual_better_share": 1 / 3,
    }
    json.dumps(report, allow_nan=False)


def test_paired_evaluation_sections_require_matching_seed_sets() -> None:
    report = paired_evaluation_deltas(
        {"seeds": [4, 5], "rewards": [20.0, 10.0]},
        {"seeds": [5, 4], "rewards": [12.0, 21.0]},
    )

    assert [row["delta"] for row in report["rows"]] == [1.0, 2.0]
    assert report["summary"]["mean_delta"] == 1.5

    with pytest.raises(ValueError, match="must match exactly"):
        paired_reward_deltas({4: 1.0}, {5: 1.0})
    with pytest.raises(ValueError, match="duplicate seed"):
        paired_reward_deltas([(4, 1.0), (4, 2.0)], [(4, 1.0)])
    with pytest.raises(ValueError, match="equal lengths"):
        paired_evaluation_deltas(
            {"seeds": [4], "rewards": []},
            {"seeds": [4], "rewards": [1.0]},
        )


def test_episode_records_are_emitted_only_after_terminal_step() -> None:
    state = EpisodeReturnState()
    state, record = record_episode_step(
        state,
        reward=1.5,
        terminated=False,
        truncated=False,
        survival_frames=4,
    )
    assert record is None
    assert state == EpisodeReturnState(
        episode=0,
        episode_return=1.5,
        episode_steps=1,
        survival_frames=4,
    )

    state, record = record_episode_step(
        state,
        reward=-0.5,
        terminated=True,
        truncated=False,
        survival_frames=4,
    )
    assert record == {
        "episode": 0,
        "episode_return": 1.0,
        "episode_steps": 2,
        "survival_frames": 8,
        "terminated": True,
        "truncated": False,
    }
    assert state == EpisodeReturnState(episode=1)
    json.dumps(record, allow_nan=False)


def test_truncation_completes_episode_and_partial_state_is_not_flushable() -> None:
    state, record = record_episode_step(
        EpisodeReturnState(episode=7),
        reward=3.0,
        terminated=False,
        truncated=True,
        survival_frames=2,
    )

    assert record is not None
    assert record["episode"] == 7
    assert record["truncated"] is True
    assert record["terminated"] is False
    assert state == EpisodeReturnState(episode=8)

    with pytest.raises(ValueError, match="finite"):
        record_episode_step(
            EpisodeReturnState(),
            reward=float("nan"),
            terminated=False,
            truncated=False,
        )
