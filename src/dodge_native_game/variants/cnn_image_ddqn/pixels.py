"""Lossless native display pixels and versioned observation profiles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import numpy as np

from .image import NativeBatchResultError

COLLISION_PROFILE: Final = "collision-image-v1"
RGB_PROFILE: Final = "native-rgb-v1"
GRAY_PROFILE: Final = "native-gray-v1"
# Keep the legacy profile ordering stable; new profiles append to the list.
OBSERVATION_PROFILES: Final = (COLLISION_PROFILE, RGB_PROFILE, GRAY_PROFILE)
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


def _integer_luma(red: int, green: int, blue: int) -> int:
    """Return the fixed native grayscale value for one display RGB color."""

    return (299 * red + 587 * green + 114 * blue + 500) // 1000


# The replay path stores palette IDs, so retain one fixed luma value per native
# palette entry.  Entries 3 and 5 intentionally share value 88; inverse
# encoding may choose either equivalent palette ID.
PICO8_LUMA_PALETTE: Final = tuple(
    _integer_luma(red, green, blue) for red, green, blue in PICO8_PALETTE
)
_LUMA_PALETTE = np.asarray(PICO8_LUMA_PALETTE, dtype=np.uint8)


def observation_shape(profile: str, stack_size: int) -> tuple[int, int, int]:
    if profile not in OBSERVATION_PROFILES:
        raise ValueError(f"unknown observation profile: {profile!r}")
    if isinstance(stack_size, bool) or not isinstance(stack_size, int):
        raise TypeError("stack_size must be an integer")
    if stack_size < 1:
        raise ValueError("stack_size must be positive")
    if profile == COLLISION_PROFILE:
        return (stack_size, 84, 84)
    if profile == RGB_PROFILE:
        return (stack_size * 3, 128, 128)
    return (stack_size, 128, 128)


def native_palette_indices_from_result(result: object) -> np.ndarray:
    """Return one validated native framebuffer of PICO-8 palette IDs.

    The native payload is the complete 128x128 display framebuffer.  No
    collision raster, crop, resize, or hidden geometry is accepted here.
    """

    value = (
        result.get("pixels")
        if isinstance(result, Mapping)
        else getattr(result, "pixels", None)
    )
    if value is None:
        raise NativeBatchResultError("native pixel profiles require native pixels")
    indices = np.asarray(value)
    if indices.shape == (1, 128, 128):
        indices = indices[0]
    if indices.shape != (128, 128) or indices.dtype != np.uint8:
        raise NativeBatchResultError("native pixels must be uint8[1,128,128]")
    if np.any(indices > 15):
        raise NativeBatchResultError("native pixels must be PICO-8 palette indices")
    return indices


def native_gray_from_rgb(rgb: object) -> np.ndarray:
    """Convert native RGB CHW pixels to one-channel fixed integer luma."""

    value = np.asarray(rgb)
    if value.shape != (3, 128, 128) or value.dtype != np.uint8:
        raise NativeBatchResultError("native RGB pixels must be uint8[3,128,128]")
    value32 = value.astype(np.uint32, copy=False)
    luma = (
        299 * value32[0]
        + 587 * value32[1]
        + 114 * value32[2]
        + 500
    ) // 1000
    return np.ascontiguousarray(luma[None, ...], dtype=np.uint8)


def native_rgb_from_result(result: object) -> np.ndarray:
    """Return owned uint8 RGB CHW without resizing, cropping, or masking."""
    indices = native_palette_indices_from_result(result)
    return np.ascontiguousarray(_PALETTE[indices].transpose(2, 0, 1))


def native_gray_from_result(result: object) -> np.ndarray:
    """Return owned uint8 grayscale CHW from the complete native framebuffer."""

    return native_gray_from_rgb(native_rgb_from_result(result))


__all__ = [
    "COLLISION_PROFILE",
    "GRAY_PROFILE",
    "OBSERVATION_PROFILES",
    "PICO8_LUMA_PALETTE",
    "PICO8_PALETTE",
    "RGB_PROFILE",
    "native_gray_from_result",
    "native_gray_from_rgb",
    "native_palette_indices_from_result",
    "native_rgb_from_result",
    "observation_shape",
]
