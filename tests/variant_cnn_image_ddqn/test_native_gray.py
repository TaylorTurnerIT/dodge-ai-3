from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from dodge_native_game.variants.cnn_image_ddqn.env import CNNImageDDQNEnv
from dodge_native_game.variants.cnn_image_ddqn.image import NativeBatchResultError
from dodge_native_game.variants.cnn_image_ddqn.pixel_replay import (
    NativePixelReplayBuffer,
    estimated_storage_bytes,
)
from dodge_native_game.variants.cnn_image_ddqn.pixels import (
    COLLISION_PROFILE,
    GRAY_PROFILE,
    OBSERVATION_PROFILES,
    PICO8_LUMA_PALETTE,
    PICO8_PALETTE,
    RGB_PROFILE,
    native_gray_from_result,
    observation_shape,
)
from dodge_native_game.variants.cnn_image_ddqn.temporal import TemporalFrameStack


def _palette_indices(offset: int = 0) -> np.ndarray:
    return (
        np.arange(128 * 128, dtype=np.uint8).reshape(128, 128) + offset
    ) % 16


def _gray_frame(offset: int = 0) -> np.ndarray:
    luma = np.asarray(PICO8_LUMA_PALETTE, dtype=np.uint8)
    return luma[_palette_indices(offset)]


def _gray_stack(frame: np.ndarray, stack_size: int = 4) -> np.ndarray:
    return np.stack([frame] * stack_size, axis=0).astype(np.uint8, copy=True)


def _successor(observation: np.ndarray, frame: np.ndarray) -> np.ndarray:
    return np.concatenate((observation[1:], frame[None, ...]), axis=0)


def test_native_gray_derives_fixed_luma_from_full_framebuffer() -> None:
    indices = _palette_indices()
    rgb_palette = np.asarray(PICO8_PALETTE, dtype=np.uint32)
    rgb = rgb_palette[indices]
    expected = (
        299 * rgb[..., 0]
        + 587 * rgb[..., 1]
        + 114 * rgb[..., 2]
        + 500
    ) // 1000

    gray = native_gray_from_result({"pixels": indices[None, ...]})

    assert gray.shape == (1, 128, 128)
    assert gray.dtype == np.uint8
    np.testing.assert_array_equal(gray[0], expected)
    assert not np.shares_memory(gray, indices)


def test_native_gray_requires_native_pixels_and_keeps_effects_payload() -> None:
    indices = _palette_indices(3)
    with pytest.raises(NativeBatchResultError):
        native_gray_from_result(
            {"pixels": None, "collision_image": np.zeros((84, 84), np.uint8)}
        )

    gray = native_gray_from_result({"pixels": indices})
    assert gray.shape == (1, 128, 128)


@dataclass
class FakeGrayResult:
    pixels: np.ndarray
    rewards: np.ndarray
    done: np.ndarray
    frames: np.ndarray
    frames_advanced: np.ndarray


class FakeGrayNative:
    def __init__(self, *, done_after: int | None = None) -> None:
        self.done_after = done_after
        self.actions: list[int] = []
        self._seed = 0
        self._steps = 0
        self.closed = False

    def reset_batch(self, seeds: object, *, startup: bool = False) -> FakeGrayResult:
        del startup
        self._seed = int(np.asarray(seeds).reshape(-1)[0])
        self._steps = 0
        return self._result(reward=0.0, done=False)

    def step_batch(self, actions: object) -> FakeGrayResult:
        self.actions.append(int(np.asarray(actions).reshape(-1)[0]))
        self._steps += 1
        done = self.done_after is not None and self._steps >= self.done_after
        return self._result(reward=float(self._steps), done=done)

    def close(self) -> None:
        self.closed = True

    def _result(self, *, reward: float, done: bool) -> FakeGrayResult:
        return FakeGrayResult(
            pixels=_palette_indices(self._seed + self._steps)[None, ...],
            rewards=np.asarray([reward], dtype=np.float32),
            done=np.asarray([done], dtype=np.bool_),
            frames=np.asarray([10 + self._steps * 4], dtype=np.uint32),
            frames_advanced=np.asarray(
                [0 if self._steps == 0 else 4], dtype=np.uint32
            ),
        )


def test_gray_env_preserves_native_action_reward_done_and_temporal_order() -> None:
    native = FakeGrayNative(done_after=2)
    env = CNNImageDDQNEnv(
        stack_size=4,
        observation_profile=GRAY_PROFILE,
        native_environment=native,
    )
    try:
        observation, _ = env.reset(seed=7)
        assert env.observation_space.shape == (4, 128, 128)
        assert env.observation_space.dtype == np.dtype(np.uint8)
        assert observation.dtype == np.uint8
        expected_initial = native_gray_from_result(
            {"pixels": _palette_indices(7)[None, ...]}
        )[0]
        np.testing.assert_array_equal(observation, _gray_stack(expected_initial))

        previous = observation
        stepped, reward, terminated, truncated, info = env.step(3)
        assert reward == 1.0
        assert not terminated
        assert not truncated
        assert info["native_frames_advanced"] == 4
        assert native.actions == [3]
        np.testing.assert_array_equal(stepped[:-1], previous[1:])
        expected_step = native_gray_from_result(
            {"pixels": _palette_indices(8)[None, ...]}
        )[0]
        np.testing.assert_array_equal(stepped[-1], expected_step)

        _, reward, terminated, truncated, info = env.step(4)
        assert reward == 2.0
        assert terminated
        assert not truncated
        assert info["native_done"]
        assert native.actions == [3, 4]
    finally:
        env.close()


def test_old_profiles_and_collision_temporal_dtype_remain_unchanged() -> None:
    assert OBSERVATION_PROFILES[:2] == (COLLISION_PROFILE, RGB_PROFILE)
    assert observation_shape(COLLISION_PROFILE, 4) == (4, 84, 84)
    assert observation_shape(RGB_PROFILE, 4) == (12, 128, 128)
    assert observation_shape(GRAY_PROFILE, 4) == (4, 128, 128)

    collision = TemporalFrameStack(4)
    collision_observation = collision.reset(
        np.zeros((1, 84, 84), dtype=np.float32)
    )
    assert collision_observation.dtype == np.float32


def test_gray_temporal_stack_owns_uint8_frames_and_orders_oldest_to_newest() -> None:
    stack = TemporalFrameStack(4, frame_shape=(1, 128, 128))
    first = np.full((1, 128, 128), 43, dtype=np.uint8)
    second = np.full((1, 128, 128), 244, dtype=np.uint8)
    initial = stack.reset(first)
    updated = stack.append(second)

    assert initial.shape == (4, 128, 128)
    assert initial.dtype == np.uint8
    assert np.all(initial == 43)
    assert np.all(updated[:-1] == 43)
    assert np.all(updated[-1] == 244)
    second.fill(0)
    assert np.all(updated[-1] == 244)


def test_gray_native_pixel_replay_round_trips_packed_luma_and_profile_shape() -> None:
    frame0 = _gray_frame()
    frame1 = _gray_frame(1)
    observation = _gray_stack(frame0)
    next_observation = _successor(observation, frame1)
    replay = NativePixelReplayBuffer(
        capacity=4,
        stack_size=4,
        seed=11,
        num_actions=9,
        observation_profile=GRAY_PROFILE,
    )

    replay.reset(observation)
    replay.add(observation, 2, 1.25, next_observation, False)
    batch = replay.sample(1)
    packed = replay.sample_packed(1)

    assert replay.observation_profile == GRAY_PROFILE
    assert replay.observation_shape == (4, 128, 128)
    np.testing.assert_array_equal(batch.observations[0], observation)
    np.testing.assert_array_equal(batch.next_observations[0], next_observation)
    assert batch.observations.dtype == np.uint8
    assert packed.observation_profile == GRAY_PROFILE
    assert packed.observation_shape == (4, 128, 128)
    assert packed.logical_shape == (4, 128, 128)
    assert packed.observations.shape == (1, 4, 8192)


def test_gray_packed_replay_keeps_full_capacity_memory_bound() -> None:
    rgb_bytes = estimated_storage_bytes(100_000, 4, RGB_PROFILE)
    gray_bytes = estimated_storage_bytes(100_000, 4, GRAY_PROFILE)
    assert gray_bytes == rgb_bytes
    assert gray_bytes < 2_000_000_000
