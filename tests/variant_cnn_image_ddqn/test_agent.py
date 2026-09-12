from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from dodge_native_game.variants.cnn_image_ddqn.agent import (
    DDQNUpdate,
    DoubleDQNAgent,
)
from dodge_native_game.variants.cnn_image_ddqn.model import IMAGE_SHAPE
from dodge_native_game.variants.cnn_image_ddqn.replay import ReplayBatch
from dodge_native_game.variants.cnn_image_ddqn.run import (
    _configure_torch_backend,
    _evaluate,
)


class _BiasQNetwork(nn.Module):
    def __init__(self, num_actions: int, scale: float = 1.0) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_actions))
        self.scale = scale

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.scale * self.bias.unsqueeze(0).expand(observations.shape[0], -1)


class _CountingQNetwork(_BiasQNetwork):
    def __init__(self, num_actions: int) -> None:
        super().__init__(num_actions)
        self.forward_calls = 0

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return super().forward(observations)


class _InspectableQNetwork(nn.Module):
    def __init__(self, num_actions: int) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Flatten())
        self.head = nn.Linear(int(np.prod(IMAGE_SHAPE)), num_actions)
        self.forward_calls = 0

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return self.head(self.features(observations))


class _TwoStepEvalEnv:
    observation_profile = "collision-image-v1"

    def __init__(self) -> None:
        self.steps = 0

    def reset(self, *, seed: int):
        del seed
        self.steps = 0
        return np.zeros(IMAGE_SHAPE, dtype=np.uint8), {}

    def step(self, action: int):
        del action
        self.steps += 1
        return (
            np.zeros(IMAGE_SHAPE, dtype=np.uint8),
            1.0,
            self.steps == 2,
            False,
            {"native_frames_advanced": 4},
        )


def test_double_dqn_update_is_finite_and_masks_terminal_bootstrap() -> None:
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
    )

    result = agent.update(batch)

    assert isinstance(result, DDQNUpdate)
    assert math.isfinite(result.loss)
    assert result.loss >= 0.0
    assert result.mean_target == 3.5
    assert result.pre_clip_grad_norm > 10.0
    assert result.batch_size == 2
    assert result.optimizer_step == 1
    gradient_norm = torch.linalg.vector_norm(
        torch.stack(
            [parameter.grad.detach().norm() for parameter in online.parameters()]
        )
    )
    assert gradient_norm <= 10.0 + 1e-6

    agent.sync_target()
    for online_parameter, target_parameter in zip(
        agent.online_network.parameters(),
        agent.target_network.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(online_parameter, target_parameter)


def test_fully_random_action_skips_online_forward_and_preserves_validation() -> None:
    online = _CountingQNetwork(num_actions=3)
    target = _CountingQNetwork(num_actions=3)
    agent = DoubleDQNAgent(
        num_actions=3,
        online_network=online,
        target_network=target,
        seed=17,
    )
    observation = np.zeros(IMAGE_SHAPE, dtype=np.uint8)

    action = agent.select_action(observation, 1.0)

    assert action in range(3)
    assert online.forward_calls == 0
    with pytest.raises(ValueError, match="observations must have shape"):
        agent.select_action(np.zeros((1, 2), dtype=np.uint8), 1.0)


def test_epsilon_fast_path_preserves_seeded_action_trace() -> None:
    online = _CountingQNetwork(num_actions=3)
    target = _CountingQNetwork(num_actions=3)
    with torch.no_grad():
        online.bias.copy_(torch.tensor([0.0, 3.0, 1.0]))
    agent = DoubleDQNAgent(
        num_actions=3,
        online_network=online,
        target_network=target,
    )
    observation = np.zeros((8, *IMAGE_SHAPE), dtype=np.uint8)
    actual_rng = np.random.default_rng(91)
    reference_rng = np.random.default_rng(91)
    mask = reference_rng.random(8) < 0.5
    expected = np.full(8, 1, dtype=np.int64)
    expected[mask] = reference_rng.integers(0, 3, size=int(mask.sum()))

    actual = agent.select_action(observation, 0.5, rng=actual_rng)

    np.testing.assert_array_equal(actual, expected)
    assert online.forward_calls == int(not np.all(mask))


def test_evaluation_uses_one_online_forward_per_decision() -> None:
    online = _InspectableQNetwork(num_actions=3)
    target = _InspectableQNetwork(num_actions=3)
    agent = DoubleDQNAgent(
        num_actions=3,
        online_network=online,
        target_network=target,
    )

    result = _evaluate(
        agent,
        _TwoStepEvalEnv(),  # type: ignore[arg-type]
        seed=1,
        episodes=1,
        max_steps=3,
    )

    assert result["mean_reward"] == 2.0
    assert result["dead_units_mean"] == 1.0
    assert online.forward_calls == 2


def test_cpu_backend_disables_unavailable_nnpack_probe() -> None:
    set_flags = torch.backends.nnpack.set_flags
    previous = torch._C._get_nnpack_enabled()
    try:
        set_flags(True)
        _configure_torch_backend("cpu")
        assert not torch._C._get_nnpack_enabled()
    finally:
        set_flags(previous)


def test_cuda_learner_backend_rejects_cpu_device() -> None:
    with pytest.raises(ValueError, match="CUDA learner"):
        DoubleDQNAgent(2, learner_backend="cuda-optimized")


def test_v82_optimizer_restore_retains_selected_backend_flags() -> None:
    agent = DoubleDQNAgent(2, network_factory=_BiasQNetwork)
    state = agent.optimizer.state_dict()
    for group in state["param_groups"]:
        group["fused"] = None
        group["foreach"] = True
    agent.learner_backend = "cuda-optimized"
    agent.load_optimizer_state_dict(state)
    assert all(group["fused"] is True for group in agent.optimizer.param_groups)
    assert all(group["foreach"] is None for group in agent.optimizer.param_groups)
