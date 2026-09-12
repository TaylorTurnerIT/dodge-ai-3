from __future__ import annotations

import numpy as np
import pytest

from dodge_native_game.variants.cnn_image_ddqn.env import CNNImageDDQNEnv
from dodge_native_game.variants.cnn_image_ddqn.image import NativeBatchResultError
from dodge_native_game.variants.cnn_image_ddqn.pixels import (
    PICO8_PALETTE,
    RGB_PROFILE,
    native_rgb_from_result,
)
from dodge_native_game.variants.cnn_image_ddqn.temporal import TemporalFrameStack


def as_bytes(value: np.ndarray) -> np.ndarray:
    return value if value.dtype == np.uint8 else np.rint(value * 255).astype(np.uint8)


def test_all_native_pixels_and_palette_colors_are_preserved() -> None:
    indices = (np.arange(128 * 128).reshape(1, 128, 128) % 16).astype(np.uint8)
    rgb = native_rgb_from_result({"pixels": indices})
    assert rgb.shape == (3, 128, 128)
    assert rgb.dtype == np.uint8
    expected = np.asarray(PICO8_PALETTE, dtype=np.uint8)[indices[0]]
    np.testing.assert_array_equal(as_bytes(rgb).transpose(1, 2, 0), expected)
    assert not np.shares_memory(indices, rgb)


@pytest.mark.parametrize(
    "payload", [None, np.zeros((84, 84), np.uint8), np.full((128, 128), 16, np.uint8)]
)
def test_rgb_never_falls_back_to_collision_image(payload: object) -> None:
    with pytest.raises(NativeBatchResultError):
        native_rgb_from_result(
            {"pixels": payload, "collision_image": np.zeros((84, 84), np.uint8)}
        )


def test_rgb_stack_appends_whole_rgb_triplets_and_owns_frames() -> None:
    stack = TemporalFrameStack(4, frame_shape=(3, 128, 128))
    first = np.zeros((3, 128, 128), dtype=np.float32)
    first[0] = 1
    reset = stack.reset(first)
    np.testing.assert_array_equal(reset, np.concatenate([first] * 4))
    second = np.zeros_like(first)
    second[1] = 1
    stepped = stack.append(second)
    np.testing.assert_array_equal(stepped[:9], reset[3:])
    np.testing.assert_array_equal(stepped[-3:], second)
    second[:] = 0
    assert stepped[-2].min() == 1
    assert stack.frame_count == stack.stack_size == 4


def test_live_rgb_matches_entire_native_display_and_transition() -> None:
    pytest.importorskip("dodge_native")
    from dodge_native_game.batch import NativeBatchEnvironment

    env = CNNImageDDQNEnv(observation_profile=RGB_PROFILE)
    oracle = NativeBatchEnvironment(
        step_frames=4,
        pixels=True,
        board=False,
        full_state=True,
        difficulty=2,
        patterns_enabled=True,
        powerups_enabled=True,
    )
    palette = np.asarray(PICO8_PALETTE, dtype=np.uint8)
    try:
        observation, info = env.reset(seed=42)
        native = oracle.reset_batch(np.asarray([42], dtype=np.uint32))
        for index in range(128):
            assert env.observation_space.contains(observation)
            np.testing.assert_array_equal(
                as_bytes(observation[-3:]).transpose(1, 2, 0),
                palette[native.pixels[0]],
            )
            np.testing.assert_array_equal(
                info["native_palette_indices"], native.pixels[0]
            )
            assert info["native_frame"] == int(native.frames[0])
            action = index % 9
            previous = observation
            observation, reward, done, truncated, info = env.step(action)
            native = oracle.step_batch(np.asarray([action], dtype=np.uint8))
            np.testing.assert_array_equal(observation[:9], previous[3:])
            assert reward == float(native.rewards[0])
            assert done == bool(native.done[0])
            assert not truncated
            if done:
                break
    finally:
        env.close()
        oracle.close()
