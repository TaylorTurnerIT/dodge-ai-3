"""Deque-backed temporal frame stacking for image observations."""

from __future__ import annotations

from collections import deque
from typing import Final

import numpy as np

from .image import COLLISION_IMAGE_SHAPE, FRAME_HEIGHT, FRAME_WIDTH

TEMPORAL_FRAME_SHAPE: Final = (FRAME_HEIGHT, FRAME_WIDTH)


def _validate_frame(frame: object) -> np.ndarray:
    value = np.asarray(frame)
    if value.shape != COLLISION_IMAGE_SHAPE:
        raise ValueError(
            f"a temporal frame must have shape (1, 84, 84); got {value.shape}"
        )
    if value.dtype != np.float32:
        raise ValueError("a temporal frame must have dtype float32")
    if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 1.0):
        raise ValueError("a temporal frame must contain finite values in [0, 1]")
    return value


class TemporalFrameStack:
    """Maintain exactly one 2-D grayscale frame per temporal position."""

    def __init__(self, stack_size: int = 4) -> None:
        if isinstance(stack_size, bool) or not isinstance(stack_size, int):
            raise TypeError("stack_size must be an integer")
        if stack_size < 1:
            raise ValueError("stack_size must be at least 1")
        self._frames: deque[np.ndarray] = deque(maxlen=stack_size)

    @property
    def stack_size(self) -> int:
        """Configured number of frames in the output."""

        return self._frames.maxlen or 0

    @property
    def frame_count(self) -> int:
        """Number of frames currently held by the deque."""

        return len(self._frames)

    @property
    def frames(self) -> tuple[np.ndarray, ...]:
        """Owned copies of the current 2-D frame history for diagnostics/tests."""

        return tuple(frame.copy() for frame in self._frames)

    def reset(self, frame: object) -> np.ndarray:
        """Fill every temporal slot with the initial native frame."""

        value = _validate_frame(frame)
        self._frames.clear()
        for _ in range(self.stack_size):
            self._frames.append(value[0].copy())
        return self._stack()

    def append(self, frame: object) -> np.ndarray:
        """Append exactly one frame and return the new stacked observation."""

        if self.frame_count != self.stack_size:
            raise RuntimeError("temporal frame stack must be reset before append")
        value = _validate_frame(frame)
        self._frames.append(value[0].copy())
        return self._stack()

    def _stack(self) -> np.ndarray:
        return np.stack(tuple(self._frames), axis=0).astype(np.float32, copy=True)
