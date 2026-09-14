"""Native practice boundary checks; no model training or corpus collection."""
import dodge_native
import numpy as np
import pytest


def arguments():
    return {
        "step_frames": 4,
        "difficulty": 1,
        "permanent_pattern": 0,
        "invulnerable": True,
        "player_start": (64.0, 64.0),
        "commands": [(0, 2.0, 0.0, 2), (0, 1.0, 0.0, 1), (0, 0.0, 0.0, 1)],
        "enemies": [(20.0, 20.0, 4.0, False, [])],
    }


def test_native_practice_replays_executed_actions_and_pixels():
    environment = dodge_native.NativePracticeEnv(**arguments())
    initial = environment.reset(42)
    frames = []
    for index, action in enumerate([2, 2, 1, 0]):
        result = environment.step()
        assert set(result) == {"pixels", "action", "terminated", "finished", "failed"}
        assert result["action"] == action
        assert not result["terminated"] and not result["failed"]
        assert result["finished"] == (index == 3)
        assert result["pixels"].shape == (128, 128)
        assert result["pixels"].dtype == np.uint8
        assert result["pixels"].max() < 16
        frames.append(result["pixels"])
    with pytest.raises(RuntimeError):
        environment.step()
    np.testing.assert_array_equal(environment.reset(42), initial)
    for frame in frames:
        np.testing.assert_array_equal(environment.step()["pixels"], frame)


def test_native_practice_death_ends_before_next_action():
    config = arguments()
    config.update(invulnerable=False, enemies=[(64.0, 64.0, 8.0, False, [])])
    environment = dodge_native.NativePracticeEnv(**config)
    environment.reset(42)
    result = environment.step()
    assert result["terminated"] and not result["finished"]
    assert result["action"] == 2
    with pytest.raises(RuntimeError):
        environment.step()


@pytest.mark.parametrize(
    "override",
    [
        {"step_frames": 1},
        {"player_start": (float("nan"), 64.0)},
        {"commands": [(0, 9.0, 0.0, 1)]},
        {"commands": [(0, 0.0, 0.0, 257)]},
        {"enemies": [(20.0, 20.0, 4.0, True, [])]},
        {"enemies": [(20.0, 20.0, 4.0, False, [(30.0, 20.0, 0)])]},
    ],
)
def test_native_practice_validates_inputs_independently(override):
    with pytest.raises(ValueError):
        dodge_native.NativePracticeEnv(**(arguments() | override))
