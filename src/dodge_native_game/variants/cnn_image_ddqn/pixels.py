"""Lossless native display pixels and versioned observation profiles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import numpy as np

from .image import NativeBatchResultError

COLLISION_PROFILE: Final = "collision-image-v1"
RGB_PROFILE: Final = "native-rgb-v1"
OBSERVATION_PROFILES: Final = (COLLISION_PROFILE, RGB_PROFILE)
# The native physical framebuffer already applies camera and screen palette.
# These are the display RGB values of its final PICO-8 palette indices.
PICO8_PALETTE: Final = (
    (0, 0, 0),
    (29, 43, 83),
    (126, 37, 83),
    (0, 135, 81),
    (171, 82, 54),
    (95, 87, 79),
    (194, 195, 199),
    (255, 241, 232),
    (255, 0, 77),
    (255, 163, 0),
    (255, 236, 39),
    (0, 228, 54),
    (41, 173, 255),
    (131, 118, 156),
    (255, 119, 168),
    (255, 204, 170),
)
_PALETTE = np.asarray(PICO8_PALETTE, dtype=np.uint8)


def observation_shape(profile: str, stack_size: int) -> tuple[int, int, int]:
    if profile not in OBSERVATION_PROFILES:
        raise ValueError(f"unknown observation profile: {profile!r}")
    if isinstance(stack_size, bool) or not isinstance(stack_size, int):
        raise TypeError("stack_size must be an integer")
    if stack_size < 1:
        raise ValueError("stack_size must be positive")
    return (
        (stack_size, 84, 84)
        if profile == COLLISION_PROFILE
        else (stack_size * 3, 128, 128)
    )


def native_rgb_from_result(result: object) -> np.ndarray:
    """Return owned uint8 RGB CHW without resizing, cropping, or masking."""
    value = (
        result.get("pixels")
        if isinstance(result, Mapping)
        else getattr(result, "pixels", None)
    )
    if value is None:
        raise NativeBatchResultError("native-rgb-v1 requires native pixels")
    indices = np.asarray(value)
    if indices.shape == (1, 128, 128):
        indices = indices[0]
    if indices.shape != (128, 128) or indices.dtype != np.uint8:
        raise NativeBatchResultError("native pixels must be uint8[1,128,128]")
    if np.any(indices > 15):
        raise NativeBatchResultError("native pixels must be PICO-8 palette indices")
    return np.ascontiguousarray(_PALETTE[indices].transpose(2, 0, 1))
