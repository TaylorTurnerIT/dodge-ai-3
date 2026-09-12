from copy import deepcopy

import pytest

from dodge_native_game.variants.cnn_image_ddqn.learning_quality import (
    assess_learning,
    select_inner_checkpoint,
)


def test_collapsed_eval_not_hidden_by_training_exploration():
    split = {"action_counts": [0] * 8 + [100], "dead_units_mean": 0.98}
    result = assess_learning(
        {"inner": split, "holdout": split, "training_action_counts": [100] * 9}
    )
    assert result["status"] == "degenerate"
    assert "inner:single-action-policy" in result["reasons"]


def test_missing_and_unflagged_data_are_not_learning_success():
    assert assess_learning({})["status"] == "insufficient"
    split = {"action_counts": [100] * 9, "dead_units_mean": 0.4}
    assert assess_learning({"inner": split, "holdout": split})["status"] == "unproven"
    bad = {"action_counts": [float("nan")] * 9, "dead_units_mean": 0.4}
    assert assess_learning({"inner": bad, "holdout": split})["status"] == "insufficient"


def candidate(step, mean, holdout):
    return {
        "step": step,
        "inner": {
            "seeds": [10001, 10002],
            "episodes": 2,
            "max_steps_per_episode": 4096,
            "mean_survival_frames": mean,
            "censored_share": 0,
        },
        "holdout": {"mean_survival_frames": holdout},
    }


def test_checkpoint_selection_ignores_holdout_and_breaks_ties_early():
    rows = [candidate(200000, 200, 9999), candidate(100000, 300, 1)]
    assert select_inner_checkpoint(rows) == 100000
    rows[0]["holdout"]["mean_survival_frames"] = -99999
    assert select_inner_checkpoint(rows) == 100000
    rows[0]["inner"]["mean_survival_frames"] = 300
    assert select_inner_checkpoint(rows) == 100000


@pytest.mark.parametrize(
    "field,value",
    [
        ("censored_share", 1),
        ("mean_survival_frames", float("nan")),
        ("seeds", [10003, 10004]),
    ],
)
def test_selection_rejects_noncomparable_or_invalid_inner_results(field, value):
    first = candidate(100000, 200, 0)
    second = deepcopy(first)
    second["step"] = 200000
    second["inner"][field] = value
    with pytest.raises(ValueError):
        select_inner_checkpoint([first, second])
