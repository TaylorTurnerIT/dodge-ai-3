from __future__ import annotations

import math

import numpy as np
import torch

from dodge_native_game.variants.cnn_image_ddqn.agent import DoubleDQNAgent
from dodge_native_game.variants.cnn_image_ddqn.model import (
    IMAGE_SHAPE,
    AtariCnnQNetwork,
)
from dodge_native_game.variants.cnn_image_ddqn.pixel_replay import (
    NativePixelReplayBuffer,
    estimated_storage_bytes,
)
from dodge_native_game.variants.cnn_image_ddqn.pixels import PICO8_PALETTE
from dodge_native_game.variants.cnn_image_ddqn.replay import ReplayBatch, ReplayBuffer


def _palette_frame(offset: int = 0) -> np.ndarray:
    indices = (
        np.arange(128 * 128, dtype=np.uint8).reshape(128, 128) + offset
    ) % 16
    palette = np.asarray(PICO8_PALETTE, dtype=np.uint8)
    return palette[indices].transpose(2, 0, 1).copy()


def _successor(observation: np.ndarray, frame: np.ndarray) -> np.ndarray:
    return np.concatenate((observation[3:], frame), axis=0)


def _decode_packed(packed: np.ndarray) -> np.ndarray:
    batch_size, stack_size, packed_bytes = packed.shape
    assert packed_bytes == 8192
    palette_ids = np.empty(
        (batch_size, stack_size, 128 * 128), dtype=np.uint8
    )
    palette_ids[..., 0::2] = packed >> 4
    palette_ids[..., 1::2] = packed & 0x0F
    rgb = np.asarray(PICO8_PALETTE, dtype=np.uint8)[palette_ids]
    rgb = rgb.reshape(batch_size, stack_size, 128, 128, 3)
    return rgb.transpose(0, 1, 4, 2, 3).reshape(
        batch_size, 3 * stack_size, 128, 128
    )


def test_rgb_model_forward_identity_and_spatial_contract() -> None:
    model = AtariCnnQNetwork(num_actions=9, input_channels=12, input_size=128)
    observations = torch.randint(0, 256, (2, 12, 128, 128), dtype=torch.uint8)

    with torch.no_grad():
        raw_outputs = model(observations)
        normalized_outputs = model(observations.float() / 255.0)

    assert model.observation_shape == (12, 128, 128)
    assert model.features[0].in_channels == 12
    assert tuple(model.features[-2].weight.shape) == (64, 64, 3, 3)
    assert model.shared[1].in_features == 64 * 12 * 12
    assert raw_outputs.shape == (2, 9)
    torch.testing.assert_close(raw_outputs, normalized_outputs)


def test_rgb_model_update_uses_shape_derived_default_network() -> None:
    agent = DoubleDQNAgent(
        num_actions=3,
        observation_shape=(12, 128, 128),
        learning_rate=1e-4,
        seed=7,
    )
    observations = np.zeros((2, 12, 128, 128), dtype=np.uint8)
    next_observations = np.full_like(observations, 255)
    batch = ReplayBatch(
        observations=observations,
        actions=np.asarray([0, 1], dtype=np.int64),
        rewards=np.asarray([1.0, 0.0], dtype=np.float32),
        next_observations=next_observations,
        dones=np.asarray([False, True], dtype=np.bool_),
    )

    result = agent.update(batch)

    assert agent.online_network.observation_shape == (12, 128, 128)
    assert result.optimizer_step == 1
    assert math.isfinite(result.loss)


def test_default_model_checkpoint_contract_is_unchanged() -> None:
    model = AtariCnnQNetwork(num_actions=5)
    state = model.state_dict()

    assert model.observation_shape == IMAGE_SHAPE
    assert model.features[0].in_channels == 4
    assert model.shared[1].in_features == 3136
    assert list(state) == [
        "features.0.weight",
        "features.0.bias",
        "features.2.weight",
        "features.2.bias",
        "features.4.weight",
        "features.4.bias",
        "shared.1.weight",
        "shared.1.bias",
        "value_stream.weight",
        "value_stream.bias",
        "advantage_stream.weight",
        "advantage_stream.bias",
    ]
    assert tuple(state["shared.1.weight"].shape) == (512, 3136)


def test_dense_replay_supports_rgb_shape_and_owns_uint8_frames() -> None:
    shape = (12, 128, 128)
    buffer = ReplayBuffer(capacity=2, seed=3, num_actions=9, observation_shape=shape)
    observation = np.zeros(shape, dtype=np.uint8)
    next_observation = np.full(shape, 17, dtype=np.uint8)
    buffer.add(observation, 1, 0.5, next_observation, False)
    observation.fill(255)
    next_observation.fill(255)

    batch = buffer.sample(1)

    assert batch.observation_shape == shape
    assert batch.observations.dtype == np.uint8
    assert batch.next_observations.dtype == np.uint8
    assert batch.observations.flags.owndata
    assert batch.next_observations.flags.owndata
    assert batch.observations[0, 0, 0, 0] == 0
    assert batch.next_observations[0, 0, 0, 0] == 17


def test_native_pixel_replay_round_trips_palette_and_stack() -> None:
    frame0 = _palette_frame()
    frame1 = _palette_frame(1)
    observation = np.concatenate([frame0] * 4, axis=0)
    next_observation = _successor(observation, frame1)
    buffer = NativePixelReplayBuffer(capacity=4, stack_size=4, seed=11, num_actions=9)

    buffer.reset(observation)
    buffer.add(observation, 2, 1.25, next_observation, False)
    batch = buffer.sample(1)

    np.testing.assert_array_equal(batch.observations[0], observation)
    np.testing.assert_array_equal(batch.next_observations[0], next_observation)
    assert batch.observations.dtype == np.uint8
    assert batch.next_observations.dtype == np.uint8
    assert batch.observations.flags.owndata
    assert batch.next_observations.flags.owndata


def test_native_pixel_replay_owns_inputs_and_reuses_identical_newest_frame() -> None:
    frame = _palette_frame(2)
    observation = np.concatenate([frame] * 4, axis=0)
    source = observation.copy()
    buffer = NativePixelReplayBuffer(capacity=2, stack_size=4, seed=5)

    buffer.reset(source)
    source.fill(0)
    buffer.add(observation, 0, 0.0, observation.copy(), False)
    sampled = buffer.sample(1)
    sampled.observations.fill(0)
    sampled.next_observations.fill(0)
    again = buffer.sample(1)

    np.testing.assert_array_equal(again.observations[0], observation)
    np.testing.assert_array_equal(again.next_observations[0], observation)
    assert buffer._next_frame_id == 3


def test_native_pixel_replay_requires_repeated_reset_and_exact_continuity() -> None:
    frame0 = _palette_frame(3)
    frame1 = _palette_frame(4)
    observation = np.concatenate([frame0] * 4, axis=0)
    next_observation = _successor(observation, frame1)
    buffer = NativePixelReplayBuffer(capacity=2, stack_size=4)
    buffer.reset(observation)
    assert buffer._next_frame_id == 1

    with np.testing.assert_raises(ValueError):
        buffer.add(observation, 0, 0.0, np.concatenate([frame1] * 4, axis=0), False)
    assert buffer._next_frame_id == 1
    buffer.add(observation, 0, 0.0, next_observation, True)
    with np.testing.assert_raises(RuntimeError):
        buffer.add(next_observation, 0, 0.0, next_observation, False)
    with np.testing.assert_raises(ValueError):
        buffer.reset(next_observation)


def test_repeated_reset_staging_does_not_evict_stored_transitions() -> None:
    buffer = NativePixelReplayBuffer(capacity=4, stack_size=4, seed=13)
    frames = [_palette_frame(offset) for offset in range(5)]
    observations = [np.concatenate([frames[0]] * 4, axis=0)]
    transitions: list[tuple[np.ndarray, np.ndarray]] = []
    for frame in frames[1:]:
        next_observation = _successor(observations[-1], frame)
        transitions.append((observations[-1], next_observation))
        observations.append(next_observation)

    buffer.reset(observations[0])
    for observation, next_observation in transitions:
        buffer.add(observation, 0, 0.0, next_observation, False)
    frame_id_before_resets = buffer._next_frame_id

    for _ in range(100):
        buffer.reset(observations[0])

    assert buffer._next_frame_id == frame_id_before_resets
    batch = buffer.sample(4)
    expected_observations = [observation for observation, _ in transitions]
    expected_next_observations = [
        next_observation for _, next_observation in transitions
    ]
    for sampled in batch.observations:
        assert any(
            np.array_equal(sampled, expected) for expected in expected_observations
        )
    for sampled in batch.next_observations:
        assert any(
            np.array_equal(sampled, expected) for expected in expected_next_observations
        )


def test_packed_sample_matches_rgb_sample_with_equal_rng_and_stays_owned() -> None:
    frames = [_palette_frame(offset) for offset in range(5)]
    observations = [np.concatenate([frames[0]] * 4, axis=0)]
    transitions: list[tuple[np.ndarray, np.ndarray]] = []
    for frame in frames[1:]:
        next_observation = _successor(observations[-1], frame)
        transitions.append((observations[-1], next_observation))
        observations.append(next_observation)

    rgb_buffer = NativePixelReplayBuffer(capacity=8, stack_size=4, seed=23)
    packed_buffer = NativePixelReplayBuffer(capacity=8, stack_size=4, seed=23)
    for buffer in (rgb_buffer, packed_buffer):
        buffer.reset(observations[0])
        for observation, next_observation in transitions:
            buffer.add(observation, 0, 0.0, next_observation, False)

    rgb_batch = rgb_buffer.sample(4)
    packed_batch = packed_buffer.sample_packed(4)

    np.testing.assert_array_equal(
        _decode_packed(packed_batch.observations), rgb_batch.observations
    )
    np.testing.assert_array_equal(
        _decode_packed(packed_batch.next_observations), rgb_batch.next_observations
    )
    np.testing.assert_array_equal(packed_batch.actions, rgb_batch.actions)
    assert packed_batch.observation_shape == (12, 128, 128)
    assert packed_batch.observations.flags.owndata
    assert packed_batch.next_observations.flags.owndata


def test_native_pixel_replay_memory_estimate_does_not_expand_rgb_stacks() -> None:
    estimated = estimated_storage_bytes(100_000, 4)
    small = NativePixelReplayBuffer(capacity=2, stack_size=4)

    assert estimated < 2_000_000_000
    assert small.allocated_bytes == estimated_storage_bytes(2, 4)
