"""Independent chronological reference for compact native pixel replay."""

from __future__ import annotations

import numpy as np
import pytest

from dodge_native_game.variants.cnn_image_ddqn.pixel_replay import (
    NativePixelReplayBuffer,
)
from dodge_native_game.variants.cnn_image_ddqn.pixels import PICO8_PALETTE


def rgb(color: int) -> np.ndarray:
    return np.broadcast_to(
        np.asarray(PICO8_PALETTE[color], dtype=np.uint8)[:, None, None], (3, 128, 128)
    ).copy()


@pytest.mark.parametrize("capacity", [1, 2, 7])
@pytest.mark.parametrize("stack_size", [1, 4, 8])
def test_identical_frames_keep_time_and_ring_wrap_preserves_every_transition(
    capacity: int, stack_size: int
) -> None:
    replay = NativePixelReplayBuffer(capacity, stack_size=stack_size, seed=42)
    expected = {}
    observation = np.concatenate([rgb(0)] * stack_size)
    for step in range(60):
        # Pairs of equal frames after a change must advance temporal history.
        color = ((step + 1) // 2) % 16
        next_observation = np.concatenate((observation[3:], rgb(color)))
        terminal = step % 11 in (0, 1, 10)
        replay.add(observation, step % 9, float(step), next_observation, terminal)
        expected[step] = (observation.copy(), next_observation.copy(), terminal)
        expected = {
            key: value for key, value in expected.items() if key > step - capacity
        }
        batch = replay.sample(len(replay))
        assert set(batch.rewards.tolist()) == set(expected)
        for index, reward in enumerate(batch.rewards):
            before, after, done = expected[int(reward)]
            np.testing.assert_array_equal(batch.observations[index], before)
            np.testing.assert_array_equal(batch.next_observations[index], after)
            assert batch.dones[index] == done
        observation = (
            np.concatenate([rgb(step % 16)] * stack_size)
            if terminal
            else (next_observation)
        )
