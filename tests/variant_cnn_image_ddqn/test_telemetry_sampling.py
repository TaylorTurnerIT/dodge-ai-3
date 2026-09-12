from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import torch

from dodge_native_game.variants.cnn_image_ddqn.agent import DoubleDQNAgent
from dodge_native_game.variants.cnn_image_ddqn.model import IMAGE_SHAPE
from dodge_native_game.variants.cnn_image_ddqn.replay import ReplayBatch

from .test_agent import _BiasQNetwork


def test_diagnostic_sampling_preserves_updates_without_cpu_metric_transfers() -> None:
    network = _BiasQNetwork(2)
    measured = DoubleDQNAgent(2, online_network=deepcopy(network))
    quiet = DoubleDQNAgent(2, online_network=deepcopy(network))
    zeros = np.zeros((2, *IMAGE_SHAPE), dtype=np.uint8)
    batch = ReplayBatch(
        zeros,
        np.asarray([0, 1]),
        np.asarray([1.0, 2.0]),
        zeros.copy(),
        np.asarray([False, True]),
    )
    result = measured.update(batch)
    with patch.object(
        torch.Tensor,
        "cpu",
        side_effect=AssertionError(
            "unsampled update must not copy diagnostic scalars to CPU"
        ),
    ):
        unmeasured = quiet.update(batch, diagnostics=False)
    assert result.diagnostics_sampled
    assert not unmeasured.diagnostics_sampled
    assert np.isnan(unmeasured.loss)
    assert measured.optimizer_steps == quiet.optimizer_steps == 1
    for key, value in measured.online_network.state_dict().items():
        torch.testing.assert_close(
            value, quiet.online_network.state_dict()[key], rtol=0, atol=0
        )


def test_sampled_metrics_use_one_cpu_transfer() -> None:
    agent = DoubleDQNAgent(2, online_network=_BiasQNetwork(2))
    zeros = np.zeros((2, *IMAGE_SHAPE), dtype=np.uint8)
    batch = ReplayBatch(
        zeros,
        np.asarray([0, 1]),
        np.asarray([1.0, 2.0]),
        zeros.copy(),
        np.asarray([False, True]),
    )
    original = torch.Tensor.cpu
    calls = []

    def counted(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        calls.append(tuple(tensor.shape))
        return original(tensor, *args, **kwargs)

    with patch.object(torch.Tensor, "cpu", counted):
        result = agent.update(batch)
    assert calls == [(8,)]
    assert result.mean_target == 1.5
