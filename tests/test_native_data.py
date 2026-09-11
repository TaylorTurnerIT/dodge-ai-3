from __future__ import annotations

import numpy as np

from dodge_native_game import (
    BOARD_SHAPE,
    ML_OBSERVATION_SHAPE,
    PIXEL_SHAPE,
    NativeBatchEnvironment,
)


def test_full_batch_exposes_owned_board_pixels_and_snapshot() -> None:
    with NativeBatchEnvironment(
        step_frames=4,
        full_state=True,
        pixels=True,
        board=True,
    ) as environment:
        reset = environment.reset_batch([42])
        step = environment.step_batch([0])

    assert reset.lane_count == 1
    assert reset.board is not None
    assert reset.board.shape == (1, *BOARD_SHAPE)
    assert reset.pixels is not None
    assert reset.pixels.shape == (1, *PIXEL_SHAPE)
    assert isinstance(reset.snapshot_bytes[0], bytes)
    assert int(step.frames_advanced[0]) == 4
    assert np.isfinite(step.rewards).all()


def test_ml_boundary_exposes_native_vector_without_render_payloads() -> None:
    with NativeBatchEnvironment(ml=True, board=False, pixels=False) as environment:
        reset = environment.reset_ml([42])
        step = environment.step_ml([0])

    assert reset.ml_observation.shape == (1, *ML_OBSERVATION_SHAPE)
    assert reset.player_positions.shape == (1, 2)
    assert step.ml_observation.shape == (1, *ML_OBSERVATION_SHAPE)
    assert reset.frames[0] < step.frames[0]


def test_same_seed_and_action_produce_same_native_hashes() -> None:
    with (
        NativeBatchEnvironment(full_state=True, pixels=True, board=True) as left,
        NativeBatchEnvironment(full_state=True, pixels=True, board=True) as right,
    ):
        left.reset_batch([7])
        right.reset_batch([7])
        left_step = left.step_batch([8])
        right_step = right.step_batch([8])

    np.testing.assert_array_equal(left_step.state_hashes, right_step.state_hashes)
    np.testing.assert_array_equal(left_step.pixel_hashes, right_step.pixel_hashes)
    np.testing.assert_array_equal(left_step.pixels, right_step.pixels)

