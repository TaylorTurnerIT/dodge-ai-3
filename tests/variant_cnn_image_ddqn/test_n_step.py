from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from dodge_native_game.variants.cnn_image_ddqn.agent import DoubleDQNAgent
from dodge_native_game.variants.cnn_image_ddqn.model import IMAGE_SHAPE
from dodge_native_game.variants.cnn_image_ddqn.n_step import NStepAccumulator
from dodge_native_game.variants.cnn_image_ddqn.pixels import (
    GRAY_PROFILE,
    PICO8_LUMA_PALETTE,
)
from dodge_native_game.variants.cnn_image_ddqn.replay import ReplayBatch, ReplayBuffer


def _observation(value: int) -> np.ndarray:
    return np.full((1,), value, dtype=np.uint8)


def _add(
    accumulator: NStepAccumulator,
    value: int,
    reward: float,
    *,
    terminated: bool = False,
    truncated: bool = False,
) -> list:
    return accumulator.add(
        _observation(value),
        value % 2,
        reward,
        _observation(value + 1),
        terminated,
        truncated,
    )


def test_three_step_return_and_segment_flush_have_exact_horizons() -> None:
    accumulator = NStepAccumulator(n_step=3, gamma=0.5)

    assert _add(accumulator, 0, 1.0) == []
    assert _add(accumulator, 1, 2.0) == []
    ready = _add(accumulator, 2, 3.0)

    assert len(ready) == 1
    assert ready[0].n_steps == 3
    assert ready[0].reward == pytest.approx(2.75)
    assert ready[0].bootstrap_discount == pytest.approx(0.125)
    assert not ready[0].done

    suffixes = accumulator.flush()
    assert [transition.n_steps for transition in suffixes] == [2, 1]
    assert [transition.reward for transition in suffixes] == pytest.approx([3.5, 3.0])
    assert [transition.bootstrap_discount for transition in suffixes] == pytest.approx(
        [0.25, 0.5]
    )
    assert all(not transition.done for transition in suffixes)
    assert len(accumulator) == 0


def test_terminal_step_flushes_all_suffixes_and_does_not_cross_episode() -> None:
    accumulator = NStepAccumulator(n_step=3, gamma=0.5)

    _add(accumulator, 0, 1.0)
    _add(accumulator, 1, 2.0)
    ready = _add(accumulator, 2, 3.0, terminated=True)

    assert [transition.n_steps for transition in ready] == [3, 2, 1]
    assert [transition.reward for transition in ready] == pytest.approx(
        [2.75, 3.5, 3.0]
    )
    assert [transition.bootstrap_discount for transition in ready] == pytest.approx(
        [0.125, 0.25, 0.5]
    )
    assert all(transition.done for transition in ready)
    assert len(accumulator) == 0

    # The next step starts a fresh episode and cannot be folded into the
    # terminal suffix emitted above.
    next_episode = _add(accumulator, 10, 10.0)
    assert next_episode == []
    flushed = accumulator.flush()
    assert len(flushed) == 1
    assert flushed[0].reward == pytest.approx(10.0)
    assert flushed[0].n_steps == 1


def test_truncation_flushes_suffixes_but_keeps_bootstrap_nonterminal() -> None:
    accumulator = NStepAccumulator(n_step=3, gamma=0.9)

    _add(accumulator, 0, 1.0)
    _add(accumulator, 1, 2.0)
    ready = _add(accumulator, 2, 3.0, truncated=True)

    assert [transition.n_steps for transition in ready] == [3, 2, 1]
    assert [transition.reward for transition in ready] == pytest.approx(
        [1.0 + 0.9 * 2.0 + 0.9**2 * 3.0, 2.0 + 0.9 * 3.0, 3.0]
    )
    assert all(not transition.done for transition in ready)
    assert [transition.bootstrap_discount for transition in ready] == pytest.approx(
        [0.9**3, 0.9**2, 0.9]
    )
    assert len(accumulator) == 0


def test_accumulator_owns_observations() -> None:
    accumulator = NStepAccumulator(n_step=1, gamma=0.99)
    observation = _observation(7)
    next_observation = _observation(8)

    emitted = accumulator.add(observation, 0, 1.0, next_observation)
    observation.fill(0)
    next_observation.fill(0)

    assert emitted[0].observation.flags.owndata
    assert emitted[0].next_observation.flags.owndata
    assert emitted[0].observation[0] == 7
    assert emitted[0].next_observation[0] == 8


def test_replay_discount_storage_is_opt_in_and_owned() -> None:
    shape = (1, 84, 84)
    frame = np.full(shape, 4, dtype=np.uint8)
    buffer = ReplayBuffer(
        capacity=2,
        seed=2,
        num_actions=2,
        observation_shape=shape,
        store_discounts=True,
    )
    buffer.add(frame, 0, 1.0, frame, False, discount=0.125)
    batch = buffer.sample(1)

    assert batch.discounts is not None
    assert batch.discounts.flags.owndata
    assert batch.discounts[0] == pytest.approx(0.125)

    legacy = ReplayBuffer(capacity=1, seed=2, num_actions=2, observation_shape=shape)
    legacy.add(frame, 0, 1.0, frame, False)
    assert legacy.sample(1).discounts is None
    with pytest.raises(ValueError, match="store_discounts"):
        legacy.add(frame, 0, 1.0, frame, False, discount=0.5)
    with pytest.raises(ValueError, match="required"):
        buffer.add(frame, 0, 1.0, frame, False)


class _BiasQNetwork(nn.Module):
    def __init__(self, num_actions: int, scale: float = 1.0) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_actions))
        self.scale = scale

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.scale * self.bias.unsqueeze(0).expand(observations.shape[0], -1)


def test_ddqn_uses_per_transition_bootstrap_discounts() -> None:
    online = _BiasQNetwork(num_actions=2, scale=100.0)
    target = _BiasQNetwork(num_actions=2)
    with torch.no_grad():
        online.bias.copy_(torch.tensor([5.0, 1.0]))
        target.bias.copy_(torch.tensor([10.0, 20.0]))

    agent = DoubleDQNAgent(
        num_actions=2,
        gamma=0.5,
        learning_rate=1e-3,
        online_network=online,
        target_network=target,
        seed=4,
    )
    zeros = np.zeros((2, *IMAGE_SHAPE), dtype=np.uint8)
    batch = ReplayBatch(
        observations=zeros,
        actions=np.array([0, 0], dtype=np.int64),
        rewards=np.array([0.0, 2.0], dtype=np.float32),
        next_observations=zeros.copy(),
        dones=np.array([False, True], dtype=np.bool_),
        discounts=np.array([0.125, 0.25], dtype=np.float32),
    )

    result = agent.update(batch)

    # Online selects action 0 (Q=5), so target Q is 10. The first target is
    # 0 + .125*10 = 1.25; the terminal second target is exactly its reward 2.
    assert result.mean_target == pytest.approx((1.25 + 2.0) / 2.0)
    assert math.isfinite(result.loss)


def test_packed_gray_decoder_uses_exact_profile_and_cached_luma_palette() -> None:
    agent = DoubleDQNAgent(
        num_actions=2,
        observation_shape=(4, 128, 128),
        online_network=_BiasQNetwork(2),
    )
    packed = np.zeros((1, 4, 8192), dtype=np.uint8)
    packed[..., 0] = np.uint8((1 << 4) | 2)

    decoded = agent._packed_observation_tensor(
        packed, observation_profile=GRAY_PROFILE
    )
    expected = np.asarray(PICO8_LUMA_PALETTE, dtype=np.float32) / 255.0
    assert decoded.shape == (1, 4, 128, 128)
    assert decoded[0, 0, 0, 0].item() == pytest.approx(expected[1])
    assert decoded[0, 0, 0, 1].item() == pytest.approx(expected[2])
    cached = agent._display_luma_palette
    agent._packed_observation_tensor(packed, observation_profile=GRAY_PROFILE)
    assert agent._display_luma_palette is cached

    with pytest.raises(ValueError, match="profile"):
        agent._packed_observation_tensor(
            packed, observation_profile="collision-image-v1"
        )
