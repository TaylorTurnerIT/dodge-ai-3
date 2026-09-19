"""Bounded RGB-palette targets for frozen pixel decoder probes.

The palette is a property of the selected training pixels.  This module keeps
palette discovery and target conversion separate from the decoder so a palette
experiment cannot accidentally derive classes from validation pixels or from a
predicted image.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

__all__ = [
    "MAX_PALETTE_SIZE",
    "PaletteCoverageError",
    "RGBPalette",
    "bce_palette_loss",
    "ce_palette_loss",
    "derive_palette",
    "forward_logits",
    "palette_indices",
    "palette_bce_loss",
    "palette_ce_loss",
    "palette_one_hot",
    "palette_to_json",
    "project_to_palette",
    "render_palette",
    "render_palette_rgb",
    "validate_palette_coverage",
]

MAX_PALETTE_SIZE = 256
DEFAULT_PALETTE_BATCH_SIZE = 64


class PaletteCoverageError(ValueError):
    """Raised when a split contains an RGB value absent from the train palette."""


@dataclass(frozen=True)
class RGBPalette:
    """Immutable, sorted RGB palette represented as a uint8 ``(K, 3)`` array."""

    colors: np.ndarray

    def __post_init__(self) -> None:
        colors = _validate_palette(self.colors)
        copied = colors.copy()
        copied.setflags(write=False)
        object.__setattr__(self, "colors", copied)

    def __len__(self) -> int:
        return int(self.colors.shape[0])

    def __iter__(self):
        return iter(self.colors)

    def tolist(self) -> list[list[int]]:
        return self.colors.tolist()


def _validate_palette(palette: Any) -> np.ndarray:
    colors = np.asarray(palette)
    if colors.dtype != np.uint8:
        raise TypeError("palette must have dtype uint8")
    if colors.ndim != 2 or colors.shape[1] != 3 or colors.shape[0] < 1:
        raise ValueError("palette must have shape (classes, 3)")
    if colors.shape[0] > MAX_PALETTE_SIZE:
        raise ValueError(f"palette exceeds cap {MAX_PALETTE_SIZE}")
    packed = _pack_colors(colors)
    if np.any(packed[1:] <= packed[:-1]):
        raise ValueError("palette colors must be strictly sorted and unique")
    return np.ascontiguousarray(colors)


def _validate_pixels(values: Any) -> np.ndarray:
    pixels = np.asarray(values)
    if pixels.dtype != np.uint8:
        raise TypeError("RGB pixels must have dtype uint8")
    if pixels.ndim != 4 or pixels.shape[1] != 3:
        raise ValueError("RGB pixels must have shape (rows, 3, height, width)")
    return pixels


def _pack_colors(colors: np.ndarray) -> np.ndarray:
    return (
        (colors[..., 0].astype(np.uint32) << 16)
        | (colors[..., 1].astype(np.uint32) << 8)
        | colors[..., 2].astype(np.uint32)
    )


def _pack_pixels(pixels: np.ndarray) -> np.ndarray:
    return (
        (pixels[:, 0].astype(np.uint32) << 16)
        | (pixels[:, 1].astype(np.uint32) << 8)
        | pixels[:, 2].astype(np.uint32)
    )


def _indices_array(indices: object, count: int) -> np.ndarray:
    if indices is None:
        return np.arange(count, dtype=np.int64)
    if isinstance(indices, range):
        return np.arange(indices.start, indices.stop, indices.step, dtype=np.int64)
    values = np.asarray(
        list(indices) if not isinstance(indices, np.ndarray) else indices
    )
    if values.ndim != 1:
        raise ValueError("row indices must be one-dimensional")
    return values.astype(np.int64, copy=False)


def _row_chunks(
    pixels: object,
    indices: object,
    batch_size: int,
) -> Iterable[np.ndarray]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
        raise TypeError("batch_size must be an integer")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    count = len(pixels)  # type: ignore[arg-type]
    selected = _indices_array(indices, count)
    if np.any(selected < 0) or np.any(selected >= count):
        raise IndexError("row index is outside pixel bank")
    for start in range(0, len(selected), int(batch_size)):
        rows = selected[start : start + int(batch_size)]
        values = _validate_pixels(pixels[rows])  # type: ignore[index]
        yield values


def _palette_from_packed(packed: np.ndarray) -> np.ndarray:
    return np.stack(
        ((packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF), axis=1
    ).astype(np.uint8, copy=False)


def derive_palette(
    pixels: object,
    indices: object = None,
    *,
    batch_size: int = DEFAULT_PALETTE_BATCH_SIZE,
    max_colors: int = MAX_PALETTE_SIZE,
) -> np.ndarray:
    """Derive a sorted exact RGB palette from selected rows in bounded chunks.

    ``pixels`` is normally the train ``ProbeBank.pixels`` map and ``indices``
    is its train range.  Validation rows must be passed to
    :func:`validate_palette_coverage` after this function returns; they are
    deliberately never consulted here.
    """

    if isinstance(max_colors, bool) or not isinstance(max_colors, (int, np.integer)):
        raise TypeError("max_colors must be an integer")
    max_colors = int(max_colors)
    if max_colors < 1 or max_colors > MAX_PALETTE_SIZE:
        raise ValueError(f"max_colors must be in 1..{MAX_PALETTE_SIZE}")
    observed = np.empty(0, dtype=np.uint32)
    for values in _row_chunks(pixels, indices, batch_size):
        packed = np.unique(_pack_pixels(values))
        observed = np.union1d(observed, packed)
        if len(observed) > max_colors:
            raise ValueError(
                f"RGB palette exceeds cap {max_colors}; "
                f"encountered at least {len(observed)} colors in train pixels"
            )
    if len(observed) == 0:
        raise ValueError("cannot derive an RGB palette from empty pixels")
    return _palette_from_packed(observed)


def validate_palette_coverage(
    pixels: object,
    indices: object,
    palette: np.ndarray,
    *,
    batch_size: int = DEFAULT_PALETTE_BATCH_SIZE,
    split: str = "validation",
) -> None:
    """Raise if any selected row contains an RGB value outside ``palette``."""

    colors = _validate_palette(palette)
    palette_packed = _pack_colors(colors)
    unknown = np.empty(0, dtype=np.uint32)
    for values in _row_chunks(pixels, indices, batch_size):
        packed = np.unique(_pack_pixels(values))
        positions = np.searchsorted(palette_packed, packed)
        covered = (positions < len(palette_packed)) & (
            palette_packed[np.minimum(positions, len(palette_packed) - 1)] == packed
        )
        unknown = np.union1d(unknown, packed[~covered])
        if len(unknown):
            sample = _palette_from_packed(unknown[:4]).tolist()
            raise PaletteCoverageError(
                f"{split} pixels contain RGB colors absent from the train palette: "
                f"{sample}"
            )


def project_to_palette(
    pixels: Any, palette: np.ndarray
) -> tuple[np.ndarray, int]:
    """Map uint8 RGB pixels onto the nearest palette color (P7 §6).

    Pixels already in ``palette`` pass through bit-identical; every other
    pixel takes the nearest palette color by RGB Euclidean distance, with
    ties resolved to the lowest palette index.  Returns the projected copy
    and the count of changed pixels.  Live inputs only: frozen banks keep
    strict coverage validation.
    """

    values = _validate_pixels(pixels)
    colors = _validate_palette(palette)
    flat = values.reshape(values.shape[0], 3, -1).transpose(0, 2, 1)
    target = colors.astype(np.int32)
    distances = (
        (flat.astype(np.int32)[:, :, None, :] - target[None, None, :, :]) ** 2
    ).sum(axis=-1)
    nearest = np.argmin(distances, axis=-1).astype(np.int64)
    projected = target[nearest].transpose(0, 2, 1).reshape(values.shape)
    projected = np.ascontiguousarray(projected, dtype=np.uint8)
    changed = int((projected != values).any(axis=1).sum())
    return projected, changed


def palette_indices(pixels: Any, palette: np.ndarray) -> np.ndarray:
    """Map exact uint8 RGB pixels to int64 palette class indices."""

    values = _validate_pixels(pixels)
    colors = _validate_palette(palette)
    original = values.shape
    packed = _pack_pixels(values).reshape(
        values.shape[0], values.shape[2], values.shape[3]
    )
    palette_packed = _pack_colors(colors)
    positions = np.searchsorted(palette_packed, packed)
    safe = np.minimum(positions, len(palette_packed) - 1)
    covered = (positions < len(palette_packed)) & (palette_packed[safe] == packed)
    if not bool(np.all(covered)):
        first = int(np.flatnonzero(~covered)[0])
        missing = int(packed.reshape(-1)[first])
        rgb = _palette_from_packed(np.asarray([missing], dtype=np.uint32))[0].tolist()
        raise PaletteCoverageError(f"RGB color {rgb} is absent from the palette")
    return positions.astype(np.int64, copy=False).reshape(
        original[0], original[2], original[3]
    )


def palette_one_hot(
    targets: torch.Tensor,
    classes: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return unweighted ``(B, K, H, W)`` one-hot palette targets."""

    if targets.ndim != 3:
        raise ValueError("palette targets must have shape (batch, height, width)")
    if isinstance(classes, bool) or not isinstance(classes, int) or classes < 1:
        raise ValueError("classes must be a positive integer")
    return F.one_hot(targets.to(dtype=torch.long), num_classes=classes).permute(
        0, 3, 1, 2
    ).to(dtype=dtype)


def ce_palette_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean per-pixel cross-entropy for palette class logits."""

    if logits.ndim != 4 or targets.ndim != 3 or logits.shape[0] != targets.shape[0]:
        raise ValueError("palette CE expects logits (B,K,H,W) and targets (B,H,W)")
    if logits.shape[2:] != targets.shape[1:]:
        raise ValueError("palette logits and targets have incompatible spatial shapes")
    return F.cross_entropy(logits, targets.to(dtype=torch.long))


def bce_palette_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Unweighted mean BCE-with-logits against palette one-hot targets."""

    if logits.ndim != 4 or targets.ndim != 3 or logits.shape[0] != targets.shape[0]:
        raise ValueError("palette BCE expects logits (B,K,H,W) and targets (B,H,W)")
    if logits.shape[2:] != targets.shape[1:]:
        raise ValueError("palette logits and targets have incompatible spatial shapes")
    return F.binary_cross_entropy_with_logits(
        logits, palette_one_hot(targets, int(logits.shape[1]), dtype=logits.dtype)
    )


palette_ce_loss = ce_palette_loss
palette_bce_loss = bce_palette_loss


def render_palette(
    values: Any,
    palette: np.ndarray,
    *,
    logits: bool = True,
) -> np.ndarray:
    """Render class indices or class logits as exact uint8 RGB ``(B,3,H,W)``."""

    colors = _validate_palette(palette)
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = np.asarray(values)
    if logits:
        if values.ndim != 4 or values.shape[1] != len(colors):
            raise ValueError(
                "palette logits must have shape (batch, classes, height, width)"
            )
        values = np.argmax(values, axis=1)
    elif values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3:
        raise ValueError("palette indices must have shape (batch, height, width)")
    if np.any(values < 0) or np.any(values >= len(colors)):
        raise ValueError("palette class index is out of range")
    return colors[values.astype(np.int64, copy=False)].transpose(0, 3, 1, 2).copy()


def render_palette_rgb(logits: torch.Tensor, palette: np.ndarray) -> torch.Tensor:
    """Render argmax palette logits as normalized float RGB tensors."""

    colors = torch.as_tensor(palette, device=logits.device, dtype=torch.float32)
    if logits.ndim != 4 or logits.shape[1] != colors.shape[0]:
        raise ValueError(
            "palette logits must have shape (batch, classes, height, width)"
        )
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("palette logits contain nonfinite values")
    indices = logits.argmax(dim=1)
    return colors[indices].permute(0, 3, 1, 2).div(255.0)


def forward_logits(decoder: torch.nn.Module, latent: torch.Tensor) -> torch.Tensor:
    """Call a decoder's raw-logit path, with a small compatibility fallback."""

    method = getattr(decoder, "forward_logits", None)
    if callable(method):
        return method(latent)
    return decoder(latent)


def palette_to_json(palette: np.ndarray | RGBPalette | None) -> list[list[int]] | None:
    """Convert palette metadata to JSON-safe nested RGB lists."""

    if palette is None:
        return None
    colors = (
        palette.colors
        if isinstance(palette, RGBPalette)
        else _validate_palette(palette)
    )
    return colors.tolist()
