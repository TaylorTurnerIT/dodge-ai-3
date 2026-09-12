from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from dodge_native_game.variants.cnn_image_ddqn import run
from dodge_native_game.variants.cnn_image_ddqn.agent import DoubleDQNAgent
from dodge_native_game.variants.cnn_image_ddqn.env import CNNImageDDQNEnv
from dodge_native_game.variants.cnn_image_ddqn.pixels import (
    COLLISION_PROFILE,
    RGB_PROFILE,
)
from dodge_native_game.variants.cnn_image_ddqn.run import _evaluate, _evaluate_batched
from dodge_native_game.variants.cnn_image_ddqn.vector_env import (
    CNNImageDDQNVectorEnv,
)

from .test_rgb_run import _TinyQNetwork


@pytest.mark.parametrize("profile", [COLLISION_PROFILE, RGB_PROFILE])
def test_vector_lane_one_matches_single_lane_trace(profile: str) -> None:
    single = CNNImageDDQNEnv(stack_size=2, observation_profile=profile)
    vector = CNNImageDDQNVectorEnv(1, stack_size=2, observation_profile=profile)
    actions = [0, 1, 4, 8, 3]
    try:
        single_observation, _ = single.reset(seed=73)
        vector_observation, _ = vector.reset([73])
        np.testing.assert_array_equal(vector_observation[0], single_observation)
        for action in actions:
            single_observation, reward, done, _, _ = single.step(action)
            vector_observation, rewards, dones, _ = vector.step([action])
            np.testing.assert_array_equal(vector_observation[0], single_observation)
            assert rewards.tolist() == [reward]
            assert dones.tolist() == [done]
            if done:
                break
    finally:
        single.close()
        vector.close()


def test_active_vector_step_freezes_excluded_lane() -> None:
    vector = CNNImageDDQNVectorEnv(2, stack_size=1, observation_profile=RGB_PROFILE)
    try:
        before, _ = vector.reset([11, 12])
        after, rewards, dones, result = vector.step(
            [0, 0], np.asarray([True, False], dtype=np.bool_)
        )
        assert result.lane_ids.tolist() == [0]
        assert rewards.shape == dones.shape == (1,)
        assert after.shape == (1, 3, 128, 128)
        np.testing.assert_array_equal(vector.observations([1])[0], before[1])
    finally:
        vector.close()


def test_batched_evaluation_matches_serial_episode_results() -> None:
    torch.manual_seed(7)
    agent = DoubleDQNAgent(9, observation_shape=(1, 84, 84))
    single = CNNImageDDQNEnv(stack_size=1, observation_profile=COLLISION_PROFILE)
    try:
        serial = _evaluate(
            agent, single, seed=5, episodes=3, max_steps=8, seed_offset=10_000
        )
    finally:
        single.close()
    batched = _evaluate_batched(
        agent,
        seed=5,
        episodes=3,
        max_steps=8,
        seed_offset=10_000,
        batch_size=3,
        env_kwargs={
            "stack_size": 1,
            "step_frames": 4,
            "difficulty": 2,
            "patterns": True,
            "powerups": True,
            "observation_profile": COLLISION_PROFILE,
            "execution": "parallel",
        },
    )
    for key in (
        "seeds",
        "rewards",
        "survival_frames",
        "action_counts",
        "terminated",
        "censored",
    ):
        assert batched[key] == serial[key]
    for key in (
        "mean_reward",
        "mean_survival_frames",
        "q_mean",
        "q_std",
        "q_gap",
        "dead_units_mean",
        "action_balance",
        "censored_share",
    ):
        assert batched[key] == pytest.approx(serial[key], abs=1e-7)


@pytest.mark.parametrize("profile", [COLLISION_PROFILE, RGB_PROFILE])
def test_multilane_training_counts_native_transitions_exactly(
    tmp_path, monkeypatch, profile: str
) -> None:
    monkeypatch.setattr(run, "AtariCnnQNetwork", _TinyQNetwork)
    root = run.train_run(
        history_root=tmp_path,
        run_id=f"vector-{profile}",
        steps=8,
        seed=7,
        stack_size=1,
        observation_profile=profile,
        replay_capacity=16,
        batch_size=2,
        warmup_steps=2,
        update_every=2,
        log_interval=4,
        eval_episodes=2,
        eval_steps=2,
        eval_batch_size=2,
        collector_lanes=4,
        collector_execution="parallel",
        device="cpu",
    )
    config = json.loads((root / "config.json").read_text())
    report = json.loads((root / "report.json").read_text())
    assert config["run"]["collector_lanes"] == 4
    assert config["run"]["collector_execution"] == "parallel"
    assert report["final_metrics"]["step"] == 8
    assert report["final_metrics"]["replay_size"] == 8
    assert report["final_metrics"]["segment_optimizer_step"] == 4
    assert sum(report["diagnostics"]["training_seed_steps"].values()) == 8
