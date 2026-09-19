"""Shared live-input boundary for steering policies (P7 §6).

Every frame batch fed to the frozen world model on live inputs (planning
rollouts, policy rollouts, evaluation) passes through :func:`history_batch`:
exact-black viewport artifacts mask to background (AD6.rule), then any other
out-of-palette pixel projects to the nearest training-palette color so
gameplay colors (powerup sparkles) never abort an episode.  Both counts are
reported per episode.  In-palette pixels pass bit-identical.  Frozen
banks/datasets keep strict coverage validation; RGB-arm models skip
projection (their palette is ``None``).
"""

from __future__ import annotations

from collections import deque
from typing import Final

import numpy as np
import torch

from .palette import project_to_palette

SHAKE_BLACK_RGB: Final[tuple[int, int, int]] = (0, 0, 0)
PLAYFIELD_BACKGROUND_RGB: Final[tuple[int, int, int]] = (41, 173, 255)

__all__ = [
    "PLAYFIELD_BACKGROUND_RGB",
    "SHAKE_BLACK_RGB",
    "history_batch",
    "mask_black_pixels",
    "model_palette",
]


def mask_black_pixels(stacked: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace exact-black pixels with playfield background (AD6.rule).

    Screen-shake strips and game-over text shadows render (0, 0, 0), a
    color absent from every training corpus (AD5.diagnosis).  A shake
    strip exposes out-of-view playfield whose training-time content is
    background, so exact-black maps to background blue.  Returns the
    masked copy and the masked pixel count.
    """

    black_rgb = np.asarray(SHAKE_BLACK_RGB, dtype=np.uint8).reshape(1, 3, 1, 1)
    black = (stacked == black_rgb).all(axis=1, keepdims=True)
    count = int(black.sum())
    if not count:
        return stacked.copy(), 0
    background = np.asarray(PLAYFIELD_BACKGROUND_RGB, dtype=np.uint8).reshape(
        1, 3, 1, 1
    )
    return np.where(black, background, stacked).astype(np.uint8), count


def model_palette(model: torch.nn.Module) -> np.ndarray | None:
    """Return the model's configured palette, or None for RGB models."""

    config = getattr(model, "config", None)
    palette = getattr(config, "palette_rgb", None)
    if palette is None:
        return None
    return np.asarray(palette, dtype=np.uint8)


def history_batch(
    frames: deque[np.ndarray],
    device: torch.device,
    palette: np.ndarray | None = None,
) -> tuple[torch.Tensor, int, int]:
    """Condition a frame deque into a model-ready batch plus counts.

    Returns ``(tensor, masked_pixels, projected_pixels)``.  When
    ``palette`` is None the projection step is skipped (RGB models).
    """

    stacked = np.stack(list(frames), axis=0).astype(np.uint8)
    masked, masked_count = mask_black_pixels(stacked)
    if palette is None:
        tensor = torch.from_numpy(masked).unsqueeze(0).to(device)
        return tensor, masked_count, 0
    projected, projected_count = project_to_palette(masked, palette)
    tensor = torch.from_numpy(projected).unsqueeze(0).to(device)
    return tensor, masked_count, projected_count
