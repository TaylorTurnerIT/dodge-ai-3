"""Matched global-pooling controls for the frozen local-token readout.

Each mode maps a ``(B, 256, 192)`` raster token sequence back to the same
shape before handing it to :class:`LocalPatchDecoder`.  The decoder core is
therefore shared in architecture and initialization across the controls;
only the attention mode learns its zero-initialized pooling query.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Final

import torch
from torch import nn

from .spatial_readout import (
    DEFAULT_FEATURE_DIM,
    DEFAULT_GRID_SIZE,
    LocalPatchDecoder,
)

__all__ = [
    "ATTENTION",
    "CONDITIONS",
    "FEATURE_DIM",
    "GRID4",
    "GRID4_BLOCK_SIZE",
    "GRID_SIZE",
    "INIT_SEED",
    "MAX",
    "MEAN",
    "NUM_PATCH_TOKENS",
    "PoolingReadout",
    "PoolingReadoutDecoder",
    "make_pooling_decoders",
    "pool_features",
]

MEAN: Final[str] = "mean"
MAX: Final[str] = "max"
ATTENTION: Final[str] = "attention"
GRID4: Final[str] = "grid4"
CONDITIONS: Final[tuple[str, str, str, str]] = (MEAN, MAX, ATTENTION, GRID4)
FEATURE_DIM: Final[int] = DEFAULT_FEATURE_DIM
GRID_SIZE: Final[int] = DEFAULT_GRID_SIZE
NUM_PATCH_TOKENS: Final[int] = GRID_SIZE * GRID_SIZE
GRID4_BLOCK_SIZE: Final[int] = 4
INIT_SEED: Final[int] = 904


def _grid_dimensions(grid_size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(grid_size, Integral) and not isinstance(grid_size, bool):
        side = int(grid_size)
        if side < 1:
            raise ValueError("grid_size must be positive")
        return side, side
    if not isinstance(grid_size, Sequence) or len(grid_size) != 2:
        raise ValueError("grid_size must be a positive integer or (height, width)")
    height, width = (int(grid_size[0]), int(grid_size[1]))
    if height < 1 or width < 1:
        raise ValueError("grid dimensions must be positive")
    return height, width


def _validate_mode(mode: str) -> str:
    if mode not in CONDITIONS:
        raise ValueError(f"mode must be one of {CONDITIONS}, got {mode!r}")
    return mode


def _validate_features(
    features: torch.Tensor,
    *,
    grid_height: int,
    grid_width: int,
    feature_dim: int,
) -> torch.Tensor:
    expected = (grid_height * grid_width, feature_dim)
    if not isinstance(features, torch.Tensor) or features.ndim != 3:
        raise ValueError(
            f"features must have shape (batch, {expected[0]}, {expected[1]})"
        )
    if tuple(features.shape[1:]) != expected:
        raise ValueError(
            f"features must have shape (batch, {expected[0]}, {expected[1]}), "
            f"got {tuple(features.shape)}"
        )
    if not torch.is_floating_point(features):
        raise TypeError("features must be floating point")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("features contain nonfinite values")
    return features


def _pool_features(
    features: torch.Tensor,
    mode: str,
    *,
    query: torch.Tensor,
    grid_height: int,
    grid_width: int,
    feature_dim: int,
) -> torch.Tensor:
    features = _validate_features(
        features,
        grid_height=grid_height,
        grid_width=grid_width,
        feature_dim=feature_dim,
    )
    if not bool(torch.isfinite(query).all()):
        raise ValueError("attention query contains nonfinite values")
    tokens = features.shape[1]
    if mode == MEAN:
        pooled = features.mean(dim=1, keepdim=True)
        return pooled.expand(-1, tokens, -1)
    if mode == MAX:
        pooled = features.amax(dim=1, keepdim=True)
        return pooled.expand(-1, tokens, -1)
    if mode == ATTENTION:
        scores = torch.einsum("bld,d->bl", features, query) / math.sqrt(feature_dim)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.einsum("bl,bld->bd", weights, features).unsqueeze(1)
        return pooled.expand(-1, tokens, -1)
    if mode == GRID4:
        if (
            grid_height % GRID4_BLOCK_SIZE
            or grid_width % GRID4_BLOCK_SIZE
        ):
            raise ValueError("grid4 requires grid dimensions divisible by four")
        blocks = features.reshape(
            features.shape[0],
            grid_height // GRID4_BLOCK_SIZE,
            GRID4_BLOCK_SIZE,
            grid_width // GRID4_BLOCK_SIZE,
            GRID4_BLOCK_SIZE,
            feature_dim,
        ).mean(dim=(2, 4))
        return blocks.repeat_interleave(GRID4_BLOCK_SIZE, dim=1).repeat_interleave(
            GRID4_BLOCK_SIZE, dim=2
        ).reshape(features.shape[0], tokens, feature_dim)
    raise AssertionError(f"unhandled pooling mode: {mode}")


def pool_features(
    features: torch.Tensor,
    mode: str,
    *,
    query: torch.Tensor | None = None,
    grid_size: int | Sequence[int] = GRID_SIZE,
    feature_dim: int = FEATURE_DIM,
) -> torch.Tensor:
    """Pool raster tokens while retaining the original token sequence shape.

    ``mean``, ``max``, and ``attention`` produce one global vector repeated at
    every raster position.  ``grid4`` averages each non-overlapping 4x4 block
    of the 16x16 grid and repeats that vector across the block.  A supplied
    query is used only by attention; the default is a zero query, which makes
    its initial result exactly the global mean.
    """

    mode = _validate_mode(mode)
    grid_height, grid_width = _grid_dimensions(grid_size)
    if not isinstance(features, torch.Tensor):
        raise ValueError("features must be a torch.Tensor")
    if isinstance(feature_dim, bool) or not isinstance(feature_dim, Integral):
        raise TypeError("feature_dim must be an integer")
    feature_dim = int(feature_dim)
    if feature_dim < 1:
        raise ValueError("feature_dim must be positive")
    if query is None:
        query = torch.zeros(feature_dim, device=features.device, dtype=features.dtype)
    if (
        not isinstance(query, torch.Tensor)
        or query.ndim != 1
        or query.shape[0] != feature_dim
    ):
        raise ValueError(f"query must have shape ({feature_dim},)")
    if query.device != features.device or query.dtype != features.dtype:
        raise ValueError("query and features must share device and dtype")
    return _pool_features(
        features,
        mode,
        query=query,
        grid_height=grid_height,
        grid_width=grid_width,
        feature_dim=feature_dim,
    )


class PoolingReadout(nn.Module):
    """Pool final-layer patch tokens and decode them with a local core."""

    def __init__(
        self,
        mode: str,
        *,
        feature_dim: int = FEATURE_DIM,
        hidden_dim: int = 256,
        grid_size: int | Sequence[int] = GRID_SIZE,
        patch_size: int = 8,
        output_channels: int = 3,
    ) -> None:
        super().__init__()
        self.mode = _validate_mode(mode)
        self.grid_height, self.grid_width = _grid_dimensions(grid_size)
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, Integral):
            raise TypeError("feature_dim must be an integer")
        self.feature_dim = int(feature_dim)
        if self.feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        self.decoder = LocalPatchDecoder(
            feature_dim=self.feature_dim,
            hidden_dim=hidden_dim,
            grid_size=(self.grid_height, self.grid_width),
            patch_size=patch_size,
            output_channels=output_channels,
        )
        self.query = nn.Parameter(
            torch.zeros(self.feature_dim),
            requires_grad=self.mode == ATTENTION,
        )
        if self.mode == GRID4 and (
            self.grid_height % GRID4_BLOCK_SIZE
            or self.grid_width % GRID4_BLOCK_SIZE
        ):
            raise ValueError("grid4 requires grid dimensions divisible by four")

    @property
    def num_patches(self) -> int:
        return self.grid_height * self.grid_width

    @property
    def core(self) -> LocalPatchDecoder:
        """Compatibility name for the wrapped local decoder core."""

        return self.decoder

    def pooled_features(self, features: torch.Tensor) -> torch.Tensor:
        """Return the mode's pooled raster sequence before decoding."""

        return pool_features(
            features,
            self.mode,
            query=self.query,
            grid_size=(self.grid_height, self.grid_width),
            feature_dim=self.feature_dim,
        )

    def forward_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Return raw decoder logits for the pooled feature sequence."""

        return self.decoder(self.pooled_features(features))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward_logits(features)


PoolingReadoutDecoder = PoolingReadout


def _rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [device.index if device.index is not None else torch.cuda.current_device()]


def _state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _states_equal(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
    return tuple(left) == tuple(right) and all(
        torch.equal(left[name], right[name]) for name in left
    )


def make_pooling_decoders(
    device: torch.device | str = "cpu",
    seed: int = INIT_SEED,
    *,
    feature_dim: int = FEATURE_DIM,
    hidden_dim: int = 256,
    grid_size: int | Sequence[int] = GRID_SIZE,
    patch_size: int = 8,
    output_channels: int = 3,
) -> dict[str, PoolingReadout]:
    """Create four matched pooling decoders with byte-identical state."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    target = torch.device(device)
    with torch.random.fork_rng(devices=_rng_devices(target)):
        torch.manual_seed(seed)
        reference = PoolingReadout(
            MEAN,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            grid_size=grid_size,
            patch_size=patch_size,
            output_channels=output_channels,
        ).to(target)
        initial = _state(reference)
        decoders = {
            mode: PoolingReadout(
                mode,
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                grid_size=grid_size,
                patch_size=patch_size,
                output_channels=output_channels,
            ).to(target)
            for mode in CONDITIONS
        }
        for decoder in decoders.values():
            decoder.load_state_dict(initial, strict=True)
    states = {mode: _state(decoder) for mode, decoder in decoders.items()}
    if any(not _states_equal(initial, state) for state in states.values()):
        raise RuntimeError("pooling decoders did not share initialization")
    if decoders[ATTENTION].query.requires_grad is not True:
        raise RuntimeError("attention query must require gradients")
    if any(
        decoders[mode].query.requires_grad
        for mode in (MEAN, MAX, GRID4)
    ):
        raise RuntimeError("non-attention pooling queries must be frozen")
    return decoders
