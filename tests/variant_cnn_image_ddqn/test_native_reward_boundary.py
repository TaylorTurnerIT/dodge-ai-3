"""Native reward data stays separate from policy pixels and base reward."""

import dodge_native
import numpy as np
import pytest

from dodge_native_game.batch import NativeBatchEnvironment


def test_v71_native_boundary_python_contract():
    field = dodge_native.reward_boundary_costs
    assert field(64, 64, 2, 125, 2, 16) == (0, 0)
    assert field(2, 64, 2, 125, 2, 16) == (1, 0)
    assert field(2, 2, 2, 125, 2, 16) == (1, 1)
    assert field(5, 64, 2, 125, 2, 16) == (0, 0)
    with pytest.raises(ValueError, match="invalid native"):
        field(float("nan"), 64, 2, 125, 2, 16)


def test_v70_pickup_payload_owned_and_base_reward_unchanged():
    env = NativeBatchEnvironment(
        step_frames=4,
        pixels=True,
        full_state=False,
        board=False,
    )
    result = env.reset_batch(np.array([42], dtype=np.uint32), startup=True)
    assert result.powerups_collected is not None
    assert result.powerups_collected.dtype == np.uint32
    assert result.powerups_collected.shape == (1,)
    old = result.powerups_collected.copy()
    assert result.rewards[0] == 0
    nxt = env.step_batch(np.array([0], dtype=np.uint8))
    assert nxt.powerups_collected is not None
    assert not np.shares_memory(result.powerups_collected, nxt.powerups_collected)
    np.testing.assert_array_equal(result.powerups_collected, old)
    assert 0 <= nxt.rewards[0] <= nxt.frames_advanced[0]
    env.close()
