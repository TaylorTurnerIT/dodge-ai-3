"""Validation and conversion for the native collision-image observation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import numpy as np

FRAME_HEIGHT: Final = 84
FRAME_WIDTH: Final = 84
COLLISION_IMAGE_SHAPE: Final = (1, FRAME_HEIGHT, FRAME_WIDTH)


class NativeBatchResultError(RuntimeError):
    """Raised when a native batch result cannot provide the image contract."""


class NativeCollisionImageUnavailable(NativeBatchResultError):
    """Raised when the native integration has not exposed ``collision_image``."""


def _result_field(result: object, name: str) -> object:
    if isinstance(result, Mapping):
        try:
            return result[name]
        except KeyError as error:
            raise NativeCollisionImageUnavailable(
                "native batch result did not expose 'collision_image'; "
                "the native integration must add collision_image to "
                "reset_batch/step_batch results"
            ) from error

    try:
        return getattr(result, name)
    except AttributeError as error:
        raise NativeCollisionImageUnavailable(
            "native batch result did not expose 'collision_image'; "
            "the native integration must add collision_image to "
            "reset_batch/step_batch results"
        ) from error


def _single_lane_image(value: object) -> np.ndarray:
    try:
        image = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise NativeBatchResultError(
            "native collision_image must be a numeric 84x84 array"
        ) from error

    if image.ndim == 2 and image.shape == (FRAME_HEIGHT, FRAME_WIDTH):
        image = image[None, ...]
    elif image.ndim == 3 and image.shape == COLLISION_IMAGE_SHAPE:
        pass
    elif image.ndim == 3 and image.shape == (FRAME_HEIGHT, FRAME_WIDTH, 1):
        image = image.transpose(2, 0, 1)
    elif image.ndim == 4 and image.shape == (1, *COLLISION_IMAGE_SHAPE):
        image = image[0]
    else:
        raise NativeBatchResultError(
            "native collision_image must have one-lane shape "
            f"(84, 84), (1, 84, 84), or (1, 1, 84, 84); got {image.shape}"
        )

    if not np.issubdtype(image.dtype, np.number):
        raise NativeBatchResultError("native collision_image must be numeric")
    return image


def normalize_collision_image(value: object) -> np.ndarray:
    """Return one native image as contiguous float32 ``(1, 84, 84)`` data.

    The native boundary may provide normalized floats or byte-like grayscale
    values. Conversion is deliberately limited to those two representations;
    unexpected ranges are rejected instead of silently clipped.
    """

    image = _single_lane_image(value).astype(np.float32, copy=True)
    if not np.isfinite(image).all():
        raise NativeBatchResultError("native collision_image must be finite")

    minimum = float(image.min())
    maximum = float(image.max())
    if minimum < 0.0 or maximum > 255.0:
        raise NativeBatchResultError(
            "native collision_image values must be in [0, 1] or [0, 255]"
        )
    if maximum > 1.0:
        image /= np.float32(255.0)
    return np.ascontiguousarray(image, dtype=np.float32)


def collision_image_from_result(result: object) -> np.ndarray:
    """Extract and normalize ``collision_image`` from one native batch result."""

    return normalize_collision_image(_result_field(result, "collision_image"))
