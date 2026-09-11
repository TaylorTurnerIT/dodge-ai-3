from __future__ import annotations

import numpy as np

from dodge_native_game import (
    COLLISION_IMAGE_CONFIG,
    COLLISION_IMAGE_OBSERVATION_VERSION,
    COLLISION_IMAGE_SHAPE,
    NativeBatchEnvironment,
)


def test_native_collision_image_is_owned_u8_and_render_free() -> None:
    with NativeBatchEnvironment(
        step_frames=4,
        full_state=False,
        pixels=False,
        board=False,
        collision_image=True,
    ) as environment:
        reset = environment.reset_batch([42])
        step = environment.step_batch([0])

    assert reset.collision_image is not None
    assert step.collision_image is not None
    assert reset.collision_image.shape == (1, *COLLISION_IMAGE_SHAPE)
    assert reset.collision_image.dtype == np.dtype(np.uint8)
    assert step.collision_image.dtype == np.dtype(np.uint8)
    assert int(reset.collision_image.min()) >= 0
    assert int(reset.collision_image.max()) <= 255
    assert COLLISION_IMAGE_OBSERVATION_VERSION == 1
    assert COLLISION_IMAGE_CONFIG == "world128-max-composite-u8-hw-v1"


def test_native_collision_image_trace_is_deterministic() -> None:
    def trace() -> list[np.ndarray]:
        with NativeBatchEnvironment(
            step_frames=4,
            board=False,
            pixels=False,
            collision_image=True,
        ) as environment:
            reset = environment.reset_batch([42])
            images = [reset.collision_image.copy()]
            for action in (0, 8, 3, 6):
                result = environment.step_batch([action])
                images.append(result.collision_image.copy())
        return images

    left = trace()
    right = trace()
    for left_image, right_image in zip(left, right, strict=True):
        np.testing.assert_array_equal(left_image, right_image)
