"""The fixed Atari-style convolutional Q-network used by this variant."""

from __future__ import annotations

from typing import Final

import torch
from torch import nn

IMAGE_SHAPE: Final = (4, 84, 84)
CNN_FEATURE_SIZE: Final = 64 * 7 * 7
INITIALIZATION_ID: Final = "kaiming-relu-xavier-head-v1"
SUPPORTED_SPATIAL_SIZES: Final = (84, 128)

__all__ = [
    "CNN_FEATURE_SIZE",
    "IMAGE_SHAPE",
    "INITIALIZATION_ID",
    "SUPPORTED_SPATIAL_SIZES",
    "AtariCnnQNetwork",
    "to_float_observations",
    "validate_observation_shape",
]


def to_float_observations(
    observations: torch.Tensor,
    *,
    expected_shape: tuple[int, int, int] = IMAGE_SHAPE,
) -> torch.Tensor:
    """Validate images and convert integer frames to normalized float32."""

    if not isinstance(observations, torch.Tensor):
        raise TypeError("observations must be a torch.Tensor")
    expected_shape = _validate_observation_shape(expected_shape)
    if observations.ndim != 4 or tuple(observations.shape[1:]) != expected_shape:
        raise ValueError(
            f"observations must have shape (N, {expected_shape}), "
            f"got {tuple(observations.shape)}"
        )
    was_integer = not torch.is_floating_point(observations)
    result = observations.to(dtype=torch.float32)
    if was_integer:
        result = result / 255.0
    return result


class AtariCnnQNetwork(nn.Module):
    """Map stacked image frames to action Q-values.

    The default final head is dueling: one shared 512-wide representation
    feeds a value stream and an advantage stream. A plain linear Q-head is
    available for controlled comparisons. Neither head has an output
    activation; outputs are action-value estimates, not probabilities.
    """

    def __init__(
        self,
        num_actions: int,
        *,
        dueling: bool = True,
        input_channels: int = IMAGE_SHAPE[0],
        input_size: int = 84,
    ) -> None:
        super().__init__()
        if isinstance(num_actions, bool) or num_actions < 1:
            raise ValueError("num_actions must be a positive integer")
        if isinstance(input_channels, bool) or not isinstance(input_channels, int):
            raise TypeError("input_channels must be an integer")
        if input_channels < 1:
            raise ValueError("input_channels must be positive")
        if isinstance(input_size, bool) or not isinstance(input_size, int):
            raise TypeError("input_size must be an integer")
        if input_size not in SUPPORTED_SPATIAL_SIZES:
            raise ValueError("input_size must be exactly 84 or 128")

        self.num_actions = int(num_actions)
        self.dueling = bool(dueling)
        self.input_size = int(input_size)
        self.observation_shape = (
            int(input_channels),
            self.input_size,
            self.input_size,
        )
        feature_map_size = _conv_output_size(self.input_size)
        feature_size = 64 * feature_map_size * feature_map_size
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 512),
            nn.ReLU(),
        )
        if self.dueling:
            self.value_stream = nn.Linear(512, 1)
            self.advantage_stream = nn.Linear(512, self.num_actions)
        else:
            self.q_head = nn.Linear(512, self.num_actions)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Keep activation variance stable across widths.

        PyTorch's default ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))`` gives
        ``Var(W) = 1/(3*fan_in)``, so each ReLU layer scales variance by
        ``1/6`` and four such layers shrink the signal ~1300x. He init
        (``2/fan_in``) holds ReLU gain at 1; linear Q-heads use Xavier
        (gain ~1) since they have no activation.
        """

        for module in self.features:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.kaiming_uniform_(self.shared[1].weight, nonlinearity="relu")
        nn.init.zeros_(self.shared[1].bias)
        heads: list[nn.Linear] = []
        if self.dueling:
            heads = [self.value_stream, self.advantage_stream]
        else:
            heads = [self.q_head]
        for head in heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return one Q-value vector per stacked grayscale observation."""

        features = self.shared(
            self.features(
                to_float_observations(
                    observations,
                    expected_shape=self.observation_shape,
                )
            )
        )
        if self.dueling:
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
        return self.q_head(features)


def validate_observation_shape(value: object) -> tuple[int, int, int]:
    """Validate a channel-first image shape supported by the CNN variant."""

    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError("observation shape must be a (channels, height, width) tuple")
    try:
        shape = tuple(int(item) for item in value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("observation shape must contain integers") from error
    if shape[0] < 1 or shape[1] != shape[2] or shape[1] not in SUPPORTED_SPATIAL_SIZES:
        raise ValueError(
            "observation shape must be (channels, 84, 84) or "
            "(channels, 128, 128)"
        )
    return shape


def _validate_observation_shape(value: tuple[int, int, int]) -> tuple[int, int, int]:
    """Backward-compatible private alias used by the forward boundary."""

    return validate_observation_shape(value)


def _conv_output_size(input_size: int) -> int:
    output_size = input_size
    for kernel_size, stride in ((8, 4), (4, 2), (3, 1)):
        output_size = (output_size - kernel_size) // stride + 1
    return output_size
