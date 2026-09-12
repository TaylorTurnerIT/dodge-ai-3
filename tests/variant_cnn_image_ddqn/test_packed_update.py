from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch import nn

from dodge_native_game.variants.cnn_image_ddqn.agent import DoubleDQNAgent
from dodge_native_game.variants.cnn_image_ddqn.pixel_replay import (
    NativePixelReplayBuffer,
)
from dodge_native_game.variants.cnn_image_ddqn.pixels import PICO8_PALETTE


class MeanRGBQ(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(12, 9)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.head(observations.mean(dim=(-2, -1)))


def test_packed_device_decode_and_weight_update_equal_full_rgb() -> None:
    rgb_replay = NativePixelReplayBuffer(16, seed=42)
    packed_replay = NativePixelReplayBuffer(16, seed=42)
    palette = np.asarray(PICO8_PALETTE, dtype=np.uint8)
    grid = np.arange(128 * 128).reshape(128, 128) % 16
    first = palette[grid].transpose(2, 0, 1)
    observation = np.concatenate([first] * 4)
    for step in range(8):
        frame = palette[(grid + step) % 16].transpose(2, 0, 1)
        following = np.concatenate((observation[3:], frame))
        for replay in (rgb_replay, packed_replay):
            replay.add(observation, step, float(step), following, step == 7)
        observation = following
    dense = rgb_replay.sample(4)
    packed = packed_replay.sample_packed(4)
    np.testing.assert_array_equal(dense.actions, packed.actions)
    network = MeanRGBQ()
    rgb_agent = DoubleDQNAgent(
        9, online_network=deepcopy(network), observation_shape=(12, 128, 128)
    )
    packed_agent = DoubleDQNAgent(
        9, online_network=deepcopy(network), observation_shape=(12, 128, 128)
    )
    decoded = packed_agent._packed_observation_tensor(packed.observations)
    torch.testing.assert_close(
        decoded, torch.tensor(dense.observations).float() / 255, rtol=0, atol=0
    )
    rgb_result = rgb_agent.update(dense)
    packed_result = packed_agent.update(packed)
    assert packed_result.loss == rgb_result.loss
    for key, value in rgb_agent.online_network.state_dict().items():
        torch.testing.assert_close(
            value, packed_agent.online_network.state_dict()[key], rtol=0, atol=0
        )
