"""Local patch readouts for spatial reconstruction controls.

The decoder is shared across real encoder patches, CLS broadcasts, and the
direct pixel control.  Every patch is processed independently after adding a
fixed raster-order ``(x, y)`` coordinate, so changing one input patch cannot
change another output patch.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Final

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    "DEFAULT_FEATURE_DIM",
    "DEFAULT_GRID_SIZE",
    "DEFAULT_OUTPUT_CHANNELS",
    "DEFAULT_PATCH_SIZE",
    "LocalPatchDecoder",
    "broadcast_cls",
    "pixel_patch_features",
]

DEFAULT_FEATURE_DIM: Final[int] = 192
DEFAULT_GRID_SIZE: Final[int] = 16
DEFAULT_OUTPUT_CHANNELS: Final[int] = 3
DEFAULT_PATCH_SIZE: Final[int] = 8
_DEFAULT_HIDDEN_DIM: Final[int] = 256


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _grid_dimensions(grid_size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(grid_size, Integral) and not isinstance(grid_size, bool):
        side = _positive_int("grid_size", int(grid_size))
        return side, side
    if not isinstance(grid_size, Sequence) or len(grid_size) != 2:
        raise ValueError("grid_size must be a positive integer or (height, width)")
    height = _positive_int("grid height", int(grid_size[0]))
    width = _positive_int("grid width", int(grid_size[1]))
    return height, width


def _raster_xy_grid(
    grid_height: int,
    grid_width: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return ``(x, y)`` coordinates in row-major patch order."""

    y = torch.linspace(-1.0, 1.0, grid_height, device=device)
    x = torch.linspace(-1.0, 1.0, grid_width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)


def broadcast_cls(
    cls: torch.Tensor, num_patches: int = DEFAULT_GRID_SIZE**2
) -> torch.Tensor:
    """Broadcast ``(B, D)`` CLS values over a raster patch sequence."""

    if not isinstance(cls, torch.Tensor) or cls.ndim != 2:
        raise ValueError("cls must have shape (batch, feature_dim)")
    if not torch.is_floating_point(cls):
        raise TypeError("cls must be floating point")
    num_patches = _positive_int("num_patches", num_patches)
    if not bool(torch.isfinite(cls).all()):
        raise ValueError("cls contains nonfinite values")
    return cls.unsqueeze(1).expand(-1, num_patches, -1)


def _palette_tensor(
    palette: Sequence[Sequence[int]] | torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    values = torch.as_tensor(palette)
    if values.ndim != 2 or tuple(values.shape) != (3, 3):
        raise ValueError("palette must contain exactly three RGB colors")
    if torch.is_floating_point(values):
        if not bool(torch.allclose(values, values.round(), rtol=0.0, atol=0.0)):
            raise ValueError("palette channels must be integers")
    elif values.dtype == torch.bool or values.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("palette channels must be integers")
    values = values.to(dtype=torch.int64)
    if bool(torch.any(values < 0)) or bool(torch.any(values > 255)):
        raise ValueError("palette channels must be in [0, 255]")
    packed = (
        (values[:, 0] << 16) | (values[:, 1] << 8) | values[:, 2]
    )
    if not bool(torch.all(packed[1:] > packed[:-1])):
        raise ValueError("palette must be sorted and contain unique colors")
    return values.to(device=device, dtype=torch.uint8)


def pixel_patch_features(
    pixels: torch.Tensor,
    palette: Sequence[Sequence[int]] | torch.Tensor,
    *,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> torch.Tensor:
    """Flatten exact native RGB one-hot patches in raster order.

    Input is native ``uint8`` ``(N, 3, H, W)``.  The configured three-color
    palette is looked up before patch extraction; output is ``(N, L, 192)``
    for the default 8x8 patches, with channels ordered class-major then
    row-major pixels inside each patch.
    """

    patch_size = _positive_int("patch_size", patch_size)
    if not isinstance(pixels, torch.Tensor) or pixels.ndim != 4:
        raise ValueError("pixels must have shape (batch, 3, height, width)")
    if pixels.shape[1] != 3 or pixels.dtype != torch.uint8:
        raise ValueError("pixels must be native uint8 RGB with three channels")
    height, width = (int(pixels.shape[-2]), int(pixels.shape[-1]))
    if height < 1 or width < 1 or height % patch_size or width % patch_size:
        raise ValueError("pixel height and width must be divisible by patch_size")
    colors = _palette_tensor(palette, device=pixels.device)
    matches = (
        pixels.unsqueeze(1) == colors.view(1, 3, 3, 1, 1)
    ).all(dim=2)
    covered = matches.any(dim=1)
    if not bool(torch.all(covered)):
        raise ValueError("pixels contain an RGB color outside configured palette")
    classes = matches.to(dtype=torch.int64).argmax(dim=1)
    one_hot = F.one_hot(classes, num_classes=3).permute(0, 3, 1, 2)
    one_hot = one_hot.to(dtype=torch.float32)
    patches = F.unfold(
        one_hot,
        kernel_size=(patch_size, patch_size),
        stride=(patch_size, patch_size),
    )
    return patches.transpose(1, 2).contiguous()


class LocalPatchDecoder(nn.Module):
    """Decode independent feature patches into raw RGB image logits."""

    def __init__(
        self,
        *,
        feature_dim: int = DEFAULT_FEATURE_DIM,
        hidden_dim: int = _DEFAULT_HIDDEN_DIM,
        grid_size: int | Sequence[int] = DEFAULT_GRID_SIZE,
        patch_size: int = DEFAULT_PATCH_SIZE,
        output_channels: int = DEFAULT_OUTPUT_CHANNELS,
    ) -> None:
        super().__init__()
        self.feature_dim = _positive_int("feature_dim", feature_dim)
        self.hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.grid_height, self.grid_width = _grid_dimensions(grid_size)
        self.patch_size = _positive_int("patch_size", patch_size)
        self.output_channels = _positive_int("output_channels", output_channels)
        coordinates = _raster_xy_grid(self.grid_height, self.grid_width)
        self.register_buffer("xy_grid", coordinates, persistent=False)
        self.output_dim = self.output_channels * self.patch_size * self.patch_size
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim + 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    @property
    def num_patches(self) -> int:
        return self.grid_height * self.grid_width

    @property
    def output_size(self) -> tuple[int, int]:
        return (
            self.grid_height * self.patch_size,
            self.grid_width * self.patch_size,
        )

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """Place class-major patch values into a raster ``(B,C,H,W)`` image."""

        expected = (
            self.num_patches,
            self.output_dim,
        )
        if patches.ndim != 3 or tuple(patches.shape[1:]) != expected:
            raise ValueError(
                "patches must have shape "
                f"(batch, {expected[0]}, {expected[1]}), got {tuple(patches.shape)}"
            )
        if not torch.is_floating_point(patches):
            raise TypeError("patches must be floating point")
        if not bool(torch.isfinite(patches).all()):
            raise ValueError("patch logits contain nonfinite values")
        values = patches.reshape(
            patches.shape[0],
            self.grid_height,
            self.grid_width,
            self.output_channels,
            self.patch_size,
            self.patch_size,
        )
        return values.permute(0, 3, 1, 4, 2, 5).reshape(
            patches.shape[0],
            self.output_channels,
            *self.output_size,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return raw RGB logits for a raster patch-feature sequence."""

        expected = (self.num_patches, self.feature_dim)
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
        coordinates = self.xy_grid.to(
            device=features.device,
            dtype=features.dtype,
        ).unsqueeze(0).expand(features.shape[0], -1, -1)
        values = self.mlp(torch.cat((features, coordinates), dim=-1))
        return self.unpatchify(values)
