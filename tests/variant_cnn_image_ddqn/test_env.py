from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pytest

from dodge_native_game.variants.cnn_image_ddqn import (
    CNNImageDDQNEnv,
    NativeCollisionImageUnavailable,
    TemporalFrameStack,
)


@dataclass
class FakeBatchResult:
    collision_image: np.ndarray
    rewards: np.ndarray
    done: np.ndarray
    frames: np.ndarray
    frames_advanced: np.ndarray
    seeds: np.ndarray


class FakeNativeBatch:
    def __init__(self, *, done_after: int | None = None) -> None:
        self.step_frames = 4
        self.done_after = done_after
        self.reset_seeds: list[int] = []
        self.actions: list[int] = []
        self._seed = 0
        self._steps = 0
        self.closed = False

    def reset_batch(self, seeds: object, *, startup: bool = False) -> FakeBatchResult:
        del startup
        self._seed = int(np.asarray(seeds).reshape(-1)[0])
        self._steps = 0
        self.reset_seeds.append(self._seed)
        return self._result(done=False, reward=0.0)

    def step_batch(self, actions: object) -> FakeBatchResult:
        self.actions.append(int(np.asarray(actions).reshape(-1)[0]))
        self._steps += 1
        done = self.done_after is not None and self._steps >= self.done_after
        return self._result(done=done, reward=float(self._steps))

    def close(self) -> None:
        self.closed = True

    def _result(self, *, done: bool, reward: float) -> FakeBatchResult:
        value = (self._seed + self._steps) % 256
        image = np.full((1, 84, 84), value, dtype=np.uint8)
        return FakeBatchResult(
            collision_image=image,
            rewards=np.asarray([reward], dtype=np.float32),
            done=np.asarray([done], dtype=np.bool_),
            frames=np.asarray([13 + self._steps * 4], dtype=np.uint32),
            frames_advanced=np.asarray([0 if self._steps == 0 else 4], dtype=np.uint32),
            seeds=np.asarray([self._seed], dtype=np.uint32),
        )


def make_env(
    *, stack_size: int = 4, done_after: int | None = None
) -> tuple[CNNImageDDQNEnv, FakeNativeBatch]:
    native = FakeNativeBatch(done_after=done_after)
    return (
        CNNImageDDQNEnv(
            stack_size=stack_size,
            step_frames=4,
            native_environment=native,
        ),
        native,
    )


@pytest.mark.parametrize("stack_size", [1, 2, 4, 8])
def test_observation_shape_dtype_bounds_and_reset_fill(stack_size: int) -> None:
    env, native = make_env(stack_size=stack_size)
    try:
        observation, info = env.reset(seed=7)
        assert env.action_space == gym.spaces.Discrete(9)
        assert env.step_frames == 4
        assert env.observation_space.shape == (stack_size, 84, 84)
        assert env.observation_space.dtype == np.dtype(np.float32)
        assert observation.shape == (stack_size, 84, 84)
        assert observation.dtype == np.dtype(np.float32)
        assert env.observation_space.contains(observation)
        assert np.all(observation == np.float32(7 / 255))
        assert env.frame_count == stack_size
        assert info["native_seed"] == 7
        assert native.reset_seeds == [7]
    finally:
        env.close()


def test_step_appends_one_frame_and_preserves_native_result() -> None:
    env, native = make_env()
    try:
        initial, _ = env.reset(seed=7)
        observation, reward, terminated, truncated, info = env.step(3)
        assert observation.shape == (4, 84, 84)
        assert np.all(initial[:-1] == observation[:-1])
        assert np.all(observation[-1] == np.float32(8 / 255))
        assert env.frame_count == 4
        assert reward == 1.0
        assert not terminated
        assert not truncated
        assert info["native_frames_advanced"] == 4
        assert native.actions == [3]
    finally:
        env.close()


def test_same_seed_and_action_trace_is_deterministic() -> None:
    left, _ = make_env()
    right, _ = make_env()
    try:
        left_observation, _ = left.reset(seed=42)
        right_observation, _ = right.reset(seed=42)
        np.testing.assert_array_equal(left_observation, right_observation)
        for action in (0, 8, 3, 6):
            left_result = left.step(action)
            right_result = right.step(action)
            np.testing.assert_array_equal(left_result[0], right_result[0])
            assert left_result[1:] == right_result[1:]
    finally:
        left.close()
        right.close()


def test_mapping_batch_result_with_collision_image_is_supported() -> None:
    class MappingNative(FakeNativeBatch):
        def reset_batch(
            self, seeds: object, *, startup: bool = False
        ) -> dict[str, np.ndarray]:
            result = super().reset_batch(seeds, startup=startup)
            return vars(result)

        def step_batch(self, actions: object) -> dict[str, np.ndarray]:
            result = super().step_batch(actions)
            return vars(result)

    native = MappingNative()
    env = CNNImageDDQNEnv(native_environment=native)
    try:
        observation, _ = env.reset(seed=3)
        next_observation, reward, terminated, truncated, _ = env.step(1)
        assert observation.shape == (4, 84, 84)
        assert next_observation.shape == (4, 84, 84)
        assert reward == 1.0
        assert not terminated
        assert not truncated
    finally:
        env.close()


def test_native_done_maps_to_terminated_without_python_truncation() -> None:
    env, _ = make_env(done_after=2)
    try:
        env.reset(seed=5)
        _, _, terminated, truncated, info = env.step(0)
        assert not terminated
        assert not truncated
        assert not info["native_done"]
        _, reward, terminated, truncated, info = env.step(0)
        assert reward == 2.0
        assert terminated
        assert not truncated
        assert info["native_done"]
        with pytest.raises(RuntimeError, match="terminated"):
            env.step(0)
    finally:
        env.close()


def test_temporal_reset_fills_and_append_replaces_exactly_one_slot() -> None:
    stack = TemporalFrameStack(stack_size=4)
    first = np.full((1, 84, 84), 0.25, dtype=np.float32)
    second = np.full((1, 84, 84), 0.75, dtype=np.float32)
    initial = stack.reset(first)
    updated = stack.append(second)
    assert stack.frame_count == 4
    assert len(stack.frames) == 4
    assert np.all(initial == 0.25)
    assert np.all(updated[:-1] == 0.25)
    assert np.all(updated[-1] == 0.75)


def test_missing_collision_image_is_explicit() -> None:
    class MissingImageNative(FakeNativeBatch):
        def reset_batch(self, seeds: object, *, startup: bool = False) -> object:
            del seeds, startup
            return {
                "rewards": np.asarray([0.0]),
                "done": np.asarray([False]),
                "frames": np.asarray([13]),
                "frames_advanced": np.asarray([0]),
            }

    env = CNNImageDDQNEnv(native_environment=MissingImageNative())
    try:
        with pytest.raises(NativeCollisionImageUnavailable, match="collision_image"):
            env.reset(seed=1)
    finally:
        env.close()


def test_gymnasium_checker_accepts_the_environment() -> None:
    from gymnasium.utils.env_checker import check_env

    env, _ = make_env()
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_live_native_boundary_is_ready_when_collision_image_is_exposed() -> None:
    from dodge_native_game import NativeBatchEnvironment

    native = NativeBatchEnvironment(
        step_frames=4,
        board=False,
        pixels=False,
        collision_image=True,
    )
    try:
        result = native.reset_batch([42])
        if not hasattr(result, "collision_image"):
            pytest.skip("current native wheel does not expose collision_image yet")
        env = CNNImageDDQNEnv(native_environment=native)
        try:
            observation, _ = env.reset(seed=42)
            assert observation.shape == (4, 84, 84)
            assert observation.dtype == np.dtype(np.float32)
        finally:
            env.close()
    finally:
        native.close()
