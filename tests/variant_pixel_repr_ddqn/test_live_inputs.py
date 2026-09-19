from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dodge_native_game.variants.pixel_repr_ddqn import mpc_eval
from dodge_native_game.variants.pixel_repr_ddqn.live_inputs import (
    history_batch,
    model_palette,
)
from dodge_native_game.variants.pixel_repr_ddqn.mpc_eval import (
    _UnknownColorAbort,
)
from dodge_native_game.variants.pixel_repr_ddqn.palette import (
    project_to_palette,
)

NAVY = (29, 43, 83)
BLUE = (41, 173, 255)
CREAM = (255, 241, 232)
PALETTE = np.asarray([NAVY, BLUE, CREAM], dtype=np.uint8)
RED = (255, 0, 77)


def _frame(color, red_pixels: int = 0, black_pixels: int = 0) -> np.ndarray:
    frame = np.zeros((3, 8, 8), dtype=np.uint8)
    frame[:] = np.asarray(color, dtype=np.uint8).reshape(3, 1, 1)
    flat = frame.reshape(3, -1)
    for index in range(black_pixels):
        flat[:, index] = (0, 0, 0)
    for index in range(black_pixels, black_pixels + red_pixels):
        flat[:, index] = RED
    return frame


def test_project_to_palette_identity_and_count() -> None:
    pixels = np.stack([_frame(BLUE), _frame(CREAM)])
    projected, count = project_to_palette(pixels, PALETTE)
    assert count == 0
    assert (projected == pixels).all()
    assert projected.dtype == np.uint8


def test_project_to_palette_maps_red_sparkle_to_navy() -> None:
    pixels = np.stack([_frame(BLUE, red_pixels=2)])
    projected, count = project_to_palette(pixels, PALETTE)
    assert count == 2
    assert tuple(projected[0, :, 0, 0]) == NAVY
    assert tuple(projected[0, :, 0, 1]) == NAVY
    assert tuple(projected[0, :, 1, 1]) == BLUE


def test_project_to_palette_tie_goes_to_lowest_index() -> None:
    palette = np.asarray([[0, 0, 0], [0, 0, 2]], dtype=np.uint8)
    pixels = np.zeros((1, 3, 1, 1), dtype=np.uint8)
    pixels[0, :, 0, 0] = (0, 0, 1)
    projected, count = project_to_palette(pixels, palette)
    assert count == 1
    assert tuple(projected[0, :, 0, 0]) == (0, 0, 0)


def test_project_to_palette_validates_inputs() -> None:
    pixels = np.stack([_frame(BLUE)])
    with pytest.raises(TypeError, match="uint8"):
        project_to_palette(pixels.astype(np.float32), PALETTE)
    with pytest.raises(ValueError, match="rows, 3, height, width"):
        project_to_palette(pixels[0], PALETTE)
    unsorted = np.asarray([BLUE, NAVY, CREAM], dtype=np.uint8)
    with pytest.raises(ValueError, match="strictly sorted"):
        project_to_palette(pixels, unsorted)


def test_model_palette_none_for_rgb_models() -> None:
    assert model_palette(object()) is None
    assert model_palette(SimpleNamespace()) is None
    assert (
        model_palette(SimpleNamespace(config=SimpleNamespace()))
        is None
    )
    model = SimpleNamespace(
        config=SimpleNamespace(palette_rgb=(NAVY, BLUE, CREAM))
    )
    palette = model_palette(model)
    assert palette is not None
    assert (palette == PALETTE).all()


def test_history_batch_masks_then_projects() -> None:
    frames = deque([_frame(BLUE, red_pixels=2, black_pixels=3)])
    tensor, masked, projected = history_batch(
        frames, torch.device("cpu"), PALETTE
    )
    assert masked == 3
    assert projected == 2
    assert tensor.shape == (1, 1, 3, 8, 8)
    assert tuple(tensor[0, 0, :, 0, 0].tolist()) == BLUE
    assert tuple(tensor[0, 0, :, 0, 3].tolist()) == NAVY
    _, masked_only, projected_only = history_batch(
        frames, torch.device("cpu"), None
    )
    assert (masked_only, projected_only) == (3, 0)


class _StrictModel(torch.nn.Module):
    """Fake encoder that refuses off-palette pixels like the real one."""

    def __init__(self, palette: np.ndarray) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            palette_rgb=tuple(map(tuple, palette.tolist()))
        )
        self._allowed = {tuple(color) for color in palette.tolist()}

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        frames = pixels[0].to(torch.uint8).numpy()
        for index in range(frames.shape[0]):
            seen = {
                tuple(pixel)
                for pixel in frames[index].reshape(3, -1).T.tolist()
            }
            if not seen <= self._allowed:
                raise ValueError(
                    "pixels contain an RGB color outside configured palette"
                )
        return torch.zeros(1, frames.shape[0], 192)


class _SparkleAdapter:
    """Background frames with two red sparkle pixels from step 2 on."""

    def __init__(self) -> None:
        self._step = 0

    def reset(self, seed: int):
        del seed
        self._step = 0
        return _frame(BLUE)

    def step(self, action: int):
        del action
        self._step += 1
        sparkles = 2 if self._step >= 2 else 0
        done = self._step >= 4
        return _frame(BLUE, red_pixels=sparkles), 0.0, done, False

    def close(self) -> None:
        pass


def test_run_episode_faces_sparkles_without_abort() -> None:
    model = _StrictModel(PALETTE)

    def policy(history, past):
        del past
        try:
            model.encode(history)
        except ValueError as error:
            if "outside configured palette" not in str(error):
                raise
            raise _UnknownColorAbort from error
        return 0, None

    result = mpc_eval.run_episode(
        _SparkleAdapter, 0, policy,
        model=model, device=torch.device("cpu"), max_decisions=8,
    )
    assert result["outcome"] == "terminated"
    assert result["survived"] == 3
    assert result["projected_pixels"] > 0
