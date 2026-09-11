from __future__ import annotations

import numpy as np
import pytest

from dodge_native_game.variants.cnn_image_ddqn.model import IMAGE_SHAPE
from dodge_native_game.variants.cnn_image_ddqn.replay import ReplayBuffer


def _add_transitions(buffer: ReplayBuffer, count: int = 5) -> None:
    for index in range(count):
        observation = np.full(IMAGE_SHAPE, index, dtype=np.uint8)
        next_observation = np.full(IMAGE_SHAPE, index + 1, dtype=np.uint8)
        buffer.add(
            observation,
            index % 3,
            float(index),
            next_observation,
            index % 2 == 0,
        )


def test_replay_sample_shapes_and_frame_validation() -> None:
    buffer = ReplayBuffer(capacity=4, seed=7, num_actions=3)
    _add_transitions(buffer)

    batch = buffer.sample(3)

    assert len(buffer) == 4
    assert batch.size == 3
    assert batch.observations.shape == (3, *IMAGE_SHAPE)
    assert batch.next_observations.shape == (3, *IMAGE_SHAPE)
    assert batch.actions.shape == (3,)
    assert batch.rewards.shape == (3,)
    assert batch.dones.shape == (3,)
    assert batch.observations.dtype == np.uint8
    assert batch.next_observations.dtype == np.uint8
    assert batch.observations.flags.owndata
    assert batch.next_observations.flags.owndata
    assert batch.dones.dtype == np.bool_

    with pytest.raises(ValueError, match="shape"):
        buffer.add(
            np.zeros((4, 84, 83), dtype=np.uint8),
            0,
            0.0,
            np.zeros(IMAGE_SHAPE, dtype=np.uint8),
            False,
        )


def test_replay_owns_uint8_frames_without_normalizing_storage() -> None:
    buffer = ReplayBuffer(capacity=2, seed=9, num_actions=3)
    source = np.full(IMAGE_SHAPE, 128, dtype=np.float32)
    next_source = np.full(IMAGE_SHAPE, 64, dtype=np.float32)

    buffer.add(source, 0, 1.0, next_source, False)
    source.fill(0)
    next_source.fill(0)
    batch = buffer.sample(1)

    assert batch.observations[0, 0, 0, 0] == 128
    assert batch.next_observations[0, 0, 0, 0] == 64
    assert batch.observations.dtype == np.uint8


def test_replay_sampling_is_deterministic_for_equal_seeds() -> None:
    left = ReplayBuffer(capacity=8, seed=123, num_actions=3)
    right = ReplayBuffer(capacity=8, seed=123, num_actions=3)
    _add_transitions(left, count=8)
    _add_transitions(right, count=8)

    left_batch = left.sample(5)
    right_batch = right.sample(5)

    np.testing.assert_array_equal(left_batch.observations, right_batch.observations)
    np.testing.assert_array_equal(left_batch.actions, right_batch.actions)
    np.testing.assert_array_equal(left_batch.rewards, right_batch.rewards)
    np.testing.assert_array_equal(
        left_batch.next_observations, right_batch.next_observations
    )
    np.testing.assert_array_equal(left_batch.dones, right_batch.dones)


def test_replay_supports_a_non_default_temporal_stack() -> None:
    buffer = ReplayBuffer(
        capacity=2,
        seed=3,
        num_actions=9,
        observation_shape=(1, 84, 84),
    )
    frame = np.full((1, 84, 84), 7, dtype=np.uint8)
    buffer.add(frame, 0, 1.0, frame, False)

    batch = buffer.sample(1)

    assert buffer.observation_shape == (1, 84, 84)
    assert batch.observations.shape == (1, 1, 84, 84)
    assert batch.observation_shape == (1, 84, 84)
