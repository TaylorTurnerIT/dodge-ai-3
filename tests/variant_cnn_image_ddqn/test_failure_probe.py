import pytest

from dodge_native_game.variants.cnn_image_ddqn.failure_probe import summarize_episode
from dodge_native_game.variants.cnn_image_ddqn.reward_profiles import PROFILES


def row(reward=4, terminal=False):
    return dict(
        reward=reward,
        terminated=terminal,
        q=50,
        next_value=1000,
        boundary_cost=0,
        repeat=True,
        q_gap=0.25,
        action=1,
        greedy_action=1,
    )


def test_terminal_target_never_bootstraps_and_return_error_is_signed():
    result = summarize_episode([row(), row(1, True)], 0.99, 3)
    assert result[0]["td_error"] == pytest.approx(50 - 4.99)
    assert result[1]["td_error"] == 49
    assert result[0]["group"] == "terminal_window"
    assert result[0]["return_error"] == result[0]["td_error"]


def test_cap_crossing_targets_excluded_and_no_uncensored_return_claim():
    result = summarize_episode([row(), row(), row()], 0.99, 3)
    assert len(result) == 1
    assert result[0]["return_error"] is None
    assert result[0]["group"] == "ordinary"
    assert result[0]["td_error"] == pytest.approx(
        50 - 4 - 0.99 * 4 - 0.99**2 * 4 - 0.99**3 * 1000
    )


def test_stronger_boundaries_only_scale_geometry_weights():
    for multiplier in (5, 10):
        profile = PROFILES[f"boundary{multiplier}-v1"]
        assert profile[:4] == PROFILES["boundary-v1"][:4]
        assert profile[4] == pytest.approx(0.01 * multiplier)
        assert profile[5] == pytest.approx(0.1 * multiplier)
