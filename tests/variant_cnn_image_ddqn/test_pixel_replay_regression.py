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


def palette_ids(color: int) -> np.ndarray:
    return np.full((128, 128), color, dtype=np.uint8)


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


def test_native_palette_ingest_matches_rgb_reference_and_keeps_samples_owned() -> None:
    reference = NativePixelReplayBuffer(16, stack_size=4, seed=19)
    direct = NativePixelReplayBuffer(16, stack_size=4, seed=19)
    observation = np.concatenate([rgb(0)] * 4)
    direct.reset_palette_ids(palette_ids(0))

    for step in range(8):
        color = (step + 1) % 16
        next_observation = np.concatenate((observation[3:], rgb(color)))
        terminal = step == 3
        reference.add(
            observation,
            step % 9,
            float(step),
            next_observation,
            terminal,
        )
        direct.add_palette_ids(
            palette_ids(color),
            step % 9,
            float(step),
            terminal,
        )
        if terminal:
            reset_color = 9
            observation = np.concatenate([rgb(reset_color)] * 4)
            direct.reset_palette_ids(palette_ids(reset_color))
        else:
            observation = next_observation

    expected = reference.sample(len(reference))
    actual = direct.sample(len(direct))
    np.testing.assert_array_equal(actual.observations, expected.observations)
    np.testing.assert_array_equal(actual.next_observations, expected.next_observations)
    np.testing.assert_array_equal(actual.actions, expected.actions)
    np.testing.assert_array_equal(actual.rewards, expected.rewards)
    np.testing.assert_array_equal(actual.dones, expected.dones)

    stored = direct._packed_frames.copy()
    packed = direct.sample_packed(len(direct))
    packed.observations.fill(0)
    packed.next_observations.fill(0)
    np.testing.assert_array_equal(direct._packed_frames, stored)


def test_interleaved_palette_streams_keep_independent_temporal_chains() -> None:
    replay = NativePixelReplayBuffer(16, stack_size=2, seed=31)
    replay.reset_palette_ids(palette_ids(1), stream_id=0)
    replay.reset_palette_ids(palette_ids(8), stream_id=1)
    expected: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    stacks = {
        0: np.concatenate([rgb(1), rgb(1)]),
        1: np.concatenate([rgb(8), rgb(8)]),
    }
    for tick in range(4):
        for stream in (0, 1):
            color = 2 + tick if stream == 0 else 9 + tick
            following = np.concatenate([stacks[stream][3:], rgb(color)])
            reward = float(10 * stream + tick)
            replay.add_palette_ids(
                palette_ids(color), stream, reward, False, stream_id=stream
            )
            expected[reward] = (stacks[stream].copy(), following.copy())
            stacks[stream] = following

    batch = replay.sample(len(replay))
    for index, reward in enumerate(batch.rewards):
        before, after = expected[float(reward)]
        np.testing.assert_array_equal(batch.observations[index], before)
        np.testing.assert_array_equal(batch.next_observations[index], after)
