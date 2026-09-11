from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from dodge_native_game.variants.cnn_image_ddqn.agent import (
    DDQNUpdate,
    DoubleDQNAgent,
)
from dodge_native_game.variants.cnn_image_ddqn.model import IMAGE_SHAPE
from dodge_native_game.variants.cnn_image_ddqn.replay import ReplayBatch
from dodge_native_game.variants.cnn_image_ddqn.run import _configure_torch_backend


class _BiasQNetwork(nn.Module):
    def __init__(self, num_actions: int, scale: float = 1.0) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_actions))
        self.scale = scale

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.scale * self.bias.unsqueeze(0).expand(observations.shape[0], -1)


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


def test_cpu_backend_disables_unavailable_nnpack_probe() -> None:
    set_flags = torch.backends.nnpack.set_flags
    previous = torch._C._get_nnpack_enabled()
    try:
        set_flags(True)
        _configure_torch_backend("cpu")
        assert not torch._C._get_nnpack_enabled()
    finally:
        set_flags(previous)
