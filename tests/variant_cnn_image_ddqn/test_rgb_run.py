from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from dodge_native_game.variants.cnn_image_ddqn import run
from dodge_native_game.variants.cnn_image_ddqn.pixels import (
    COLLISION_PROFILE,
    RGB_PROFILE,
    observation_shape,
)
from dodge_native_game.variants.cnn_image_ddqn.replay import ReplayBatch


class _RgbEnv:
    observation_profile = RGB_PROFILE

    def __init__(self, calls: list[dict[str, object]]) -> None:
        self.calls = calls
        self.closed = False
        self.observation = np.zeros((12, 128, 128), dtype=np.uint8)

    def reset(self, *, seed: int | None = None):
        return self.observation.copy(), {
            "native_seed": seed,
            "native_frame": 13,
            "native_frames_advanced": 13,
        }

    def step(self, action: int):
        del action
        return self.observation.copy(), 0.0, False, False, {
            "native_frame": 17,
            "native_frames_advanced": 4,
            "native_event_flags": 0,
            "native_mode": 2,
            "native_shattered": 0,
            "native_score": 0.0,
        }

    def close(self) -> None:
        self.closed = True


class _TinyQNetwork(nn.Module):
    def __init__(
        self,
        num_actions: int,
        *,
        dueling: bool = True,
        input_channels: int = 4,
        input_size: int = 84,
    ) -> None:
        super().__init__()
        del dueling
        self.observation_shape = (input_channels, input_size, input_size)
        self.bias = nn.Parameter(torch.zeros(num_actions))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.bias.unsqueeze(0).expand(observations.shape[0], -1)


class _TinyPixelReplay:
    constructed = False
    packed_samples = 0

    @classmethod
    def estimated_storage_bytes(cls, capacity: int, stack_size: int) -> int:
        return int(capacity * stack_size)

    def __init__(
        self,
        capacity: int,
        *,
        seed: int,
        num_actions: int,
        stack_size: int,
    ) -> None:
        del seed, num_actions, stack_size
        type(self).constructed = True
        self.capacity = capacity
        self.items: list[tuple[object, int, float, object, bool]] = []

    def __len__(self) -> int:
        return len(self.items)

    def reset(self, observation: object) -> None:
        del observation

    def add(
        self,
        observation: object,
        action: int,
        reward: float,
        next_observation: object,
        done: bool,
    ) -> None:
        self.items.append((observation, action, reward, next_observation, done))

    def sample(self, batch_size: int) -> ReplayBatch:
        rows = self.items[:batch_size]
        return ReplayBatch(
            observations=np.asarray([row[0] for row in rows], dtype=np.uint8),
            actions=np.asarray([row[1] for row in rows], dtype=np.int64),
            rewards=np.asarray([row[2] for row in rows], dtype=np.float32),
            next_observations=np.asarray([row[3] for row in rows], dtype=np.uint8),
            dones=np.asarray([row[4] for row in rows], dtype=np.bool_),
        )

    def sample_packed(self, batch_size: int) -> ReplayBatch:
        type(self).packed_samples += 1
        return self.sample(batch_size)


def test_native_rgb_run_records_profile_shape_and_packed_replay(
    tmp_path, monkeypatch
) -> None:
    calls: list[dict[str, object]] = []

    def factory(**kwargs):
        calls.append(kwargs)
        return _RgbEnv(calls)

    monkeypatch.setattr(run, "AtariCnnQNetwork", _TinyQNetwork)
    monkeypatch.setattr(run, "_pixel_replay_class", lambda: _TinyPixelReplay)
    monkeypatch.setattr(run, "_available_host_memory_bytes", lambda: 1 << 30)
    _TinyPixelReplay.constructed = False
    _TinyPixelReplay.packed_samples = 0

    root = run.train_run(
        history_root=tmp_path,
        run_id="rgb-run",
        steps=2,
        seed=7,
        stack_size=4,
        observation_profile=RGB_PROFILE,
        replay_capacity=32,
        batch_size=2,
        warmup_steps=2,
        update_every=2,
        log_interval=1,
        eval_episodes=1,
        eval_steps=1,
        device="cpu",
        env_factory=factory,
    )

    manifest = json.loads((root / "manifest.json").read_text())
    config = json.loads((root / "config.json").read_text())
    checkpoint = torch.load(
        root / "checkpoints" / "step-2.pt", map_location="cpu", weights_only=False
    )
    expected_shape = list(observation_shape(RGB_PROFILE, 4))
    assert _TinyPixelReplay.constructed
    assert _TinyPixelReplay.packed_samples == 1
    assert calls and all(call["observation_profile"] == RGB_PROFILE for call in calls)
    assert manifest["observation_profile"] == RGB_PROFILE
    assert manifest["observation"]["shape"] == expected_shape
    assert config["observation"]["profile"] == RGB_PROFILE
    assert config["observation"]["shape"] == expected_shape
    assert config["model"]["input_channels"] == 12
    assert config["model"]["input_size"] == 128
    assert config["replay"]["capacity"] == 32
    assert config["replay"]["encoding"] == "native-palette4-frame-ring-v1"
    assert checkpoint["observation_profile"] == RGB_PROFILE
    assert checkpoint["stack_size"] == 4
    assert checkpoint["model_input_size"] == 128
    assert checkpoint["observation_shape"] == expected_shape


def test_missing_profile_is_legacy_collision_and_profile_mismatch_is_rejected() -> None:
    legacy = {"observation_shape": [1, 84, 84]}
    run._validate_checkpoint_observation(
        legacy,
        observation_profile=COLLISION_PROFILE,
        stack_size=1,
        expected_shape=(1, 84, 84),
    )
    with pytest.raises(ValueError, match="observation profile"):
        run._validate_checkpoint_observation(
            {
                "observation_profile": RGB_PROFILE,
                "stack_size": 4,
                "observation_shape": [4, 84, 84],
            },
            observation_profile=COLLISION_PROFILE,
            stack_size=4,
            expected_shape=(4, 84, 84),
        )


def test_rgb_replay_memory_gate_runs_before_allocation(monkeypatch) -> None:
    _TinyPixelReplay.constructed = False
    monkeypatch.setattr(run, "_pixel_replay_class", lambda: _TinyPixelReplay)
    monkeypatch.setattr(run, "_available_host_memory_bytes", lambda: 100)
    with pytest.raises(MemoryError, match="headroom"):
        run._replay_storage_plan(
            observation_profile=RGB_PROFILE,
            capacity=100,
            stack_size=4,
            observation_shape_value=(12, 128, 128),
        )
    assert not _TinyPixelReplay.constructed


def test_non_divisible_update_and_log_intervals_sample_latest_update() -> None:
    steps = 16
    samples: list[int] = []
    for step in range(4, steps + 1, 4):
        if run._diagnostics_requested(
            step=step,
            update_every=4,
            log_interval=6,
            steps=steps,
            optimizer_step=step // 4 - 1,
            optimizer_step_start=0,
        ):
            samples.append(step)
    assert samples == [4, 12, 16]


def test_uint8_rgb_observations_skip_float_round_trip() -> None:
    value = np.full((12, 128, 128), 7, dtype=np.uint8)
    converted = run._observation_to_uint8(value, value.shape)
    assert converted.dtype == np.uint8
    assert converted.flags.c_contiguous
    np.testing.assert_array_equal(converted, value)
