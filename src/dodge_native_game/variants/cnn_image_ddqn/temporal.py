"""Deque-backed temporal frame stacking for image observations."""

from __future__ import annotations

from collections import deque
from typing import Final

import numpy as np

from .image import COLLISION_IMAGE_SHAPE, FRAME_HEIGHT, FRAME_WIDTH

TEMPORAL_FRAME_SHAPE: Final = (FRAME_HEIGHT, FRAME_WIDTH)
NATIVE_GRAY_FRAME_SHAPE: Final = (1, 128, 128)
NATIVE_RGB_FRAME_SHAPE: Final = (3, 128, 128)


def _validate_frame(
    frame: object, shape: tuple[int, int, int] = COLLISION_IMAGE_SHAPE
) -> np.ndarray:
    value = np.asarray(frame)
    if value.shape != shape:
        raise ValueError(
            f"a temporal frame must have shape {shape}; got {value.shape}"
        )
    if shape in (NATIVE_GRAY_FRAME_SHAPE, NATIVE_RGB_FRAME_SHAPE):
        if value.dtype == np.uint8:
            return value
        if shape == NATIVE_GRAY_FRAME_SHAPE:
            raise ValueError("a native grayscale frame must have dtype uint8")
    if value.dtype != np.float32:
        raise ValueError("a temporal frame must have dtype float32")
    if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 1.0):
        raise ValueError("a temporal frame must contain finite values in [0, 1]")
    return value


class TemporalFrameStack:
    """Maintain a fixed temporal history of grayscale or RGB frames."""

    def __init__(
        self, stack_size: int = 4, *, frame_shape: tuple[int, int, int] = (1, 84, 84)
    ) -> None:
        if isinstance(stack_size, bool) or not isinstance(stack_size, int):
            raise TypeError("stack_size must be an integer")
        if stack_size < 1:
            raise ValueError("stack_size must be at least 1")
        if frame_shape not in (
            (1, 84, 84),
            NATIVE_GRAY_FRAME_SHAPE,
            NATIVE_RGB_FRAME_SHAPE,
        ):
            raise ValueError(
                "frame_shape must be collision84, native grayscale128, "
                "or native RGB128"
            )
        self.frame_shape = frame_shape
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
        """Owned copies of the current frame history for diagnostics/tests."""

        return tuple(frame.copy() for frame in self._frames)

    def reset(self, frame: object) -> np.ndarray:
        """Fill every temporal slot with the initial native frame."""

        value = _validate_frame(frame, self.frame_shape)
        self._frames.clear()
        for _ in range(self.stack_size):
            self._frames.append(self._owned_frame(value))
        return self._stack()

    def append(self, frame: object) -> np.ndarray:
        """Append exactly one frame and return the new stacked observation."""

        if self.frame_count != self.stack_size:
            raise RuntimeError("temporal frame stack must be reset before append")
        value = _validate_frame(frame, self.frame_shape)
        self._frames.append(self._owned_frame(value))
        return self._stack()

    def _stack(self) -> np.ndarray:
        if self.frame_shape == NATIVE_GRAY_FRAME_SHAPE:
            return np.stack(tuple(self._frames), axis=0).astype(
                np.uint8, copy=True
            )
        if self.frame_shape[0] == 3:
            return np.concatenate(tuple(self._frames), axis=0)
        return np.stack(tuple(self._frames), axis=0).astype(np.float32, copy=True)

    def _owned_frame(self, value: np.ndarray) -> np.ndarray:
        return (value[0] if self.frame_shape[0] == 1 else value).copy()
