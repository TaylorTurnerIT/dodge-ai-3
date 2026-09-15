from __future__ import annotations

import pytest
import torch
from torch import nn

from dodge_native_game.variants.pixel_repr_ddqn.dynamics import (
    audit_action_conditioning,
    evaluate_dynamics,
    plan_audit_windows,
)


class ActionTraceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("marker", torch.tensor([3.0]))
        self.histories: list[tuple[torch.Tensor, torch.Tensor]] = []

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return pixels.mean(dim=(2, 3, 4), keepdim=False).unsqueeze(-1) * self.scale

    def predict(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        torch.rand((), device=z.device)
        self.histories.append((z.detach().clone(), actions.detach().clone()))
        return z + actions.to(z.dtype).unsqueeze(-1)


def _window(values: list[float], actions: list[int]) -> dict[str, torch.Tensor]:
    pixels = torch.tensor(values, dtype=torch.float32).view(-1, 1, 1, 1)
    return {"pixels": pixels, "actions": torch.tensor(actions, dtype=torch.long)}


def test_teacher_forced_alignment_and_controls_cover_all_windows() -> None:
    model = ActionTraceModel()
    dataset = [
        _window([0.0, 1.0, 3.0, 6.0], [1, 2, 3]),
        _window([2.0, 3.0, 5.0, 8.0], [1, 2, 3]),
    ]

    result = evaluate_dynamics(model, dataset, "cpu")

    assert result == {
        "window_count": 2,
        "prediction_mse": 0.0,
        "persistence_mse": 9.0,
        "wrong_action_mse": 1.0,
        "prediction_vs_persistence_ratio": 0.0,
        "action_sensitivity_mse": 1.0,
        "scope": "one-step teacher-forced",
    }
    assert len(model.histories) == 4
    torch.testing.assert_close(
        model.histories[0][0], torch.tensor([[[0.0], [1.0], [3.0]]])
    )
    torch.testing.assert_close(model.histories[0][1], torch.tensor([[1, 2, 3]]))
    torch.testing.assert_close(model.histories[1][1], torch.tensor([[2, 3, 4]]))


def test_evaluation_is_read_only_preserves_mode_and_rng() -> None:
    model = ActionTraceModel().train()
    before_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    before_rng = torch.random.get_rng_state()

    evaluate_dynamics(model, [_window([0.0, 0.0, 0.0, 0.0], [0, 8, 0])], "cpu")

    assert model.training is True
    assert model.scale.grad is None
    assert torch.equal(before_rng, torch.random.get_rng_state())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before_state[key])


def test_zero_persistence_denominator_reports_null_ratio() -> None:
    result = evaluate_dynamics(
        ActionTraceModel().eval(),
        [_window([4.0, 4.0, 4.0, 4.0], [0, 0, 0])],
        "cpu",
    )

    assert result["persistence_mse"] == 0.0
    assert result["prediction_mse"] == 0.0
    assert result["prediction_vs_persistence_ratio"] is None


def test_plan_audit_windows_spreads_starts_across_episodes() -> None:
    from types import SimpleNamespace

    records = [
        SimpleNamespace(count=128),
        SimpleNamespace(count=2),
        SimpleNamespace(count=9),
    ]
    plan = plan_audit_windows(records, history_size=3)
    assert plan == ((0, 31), (0, 62), (0, 93), (2, 1), (2, 3), (2, 4))
    assert plan_audit_windows(records, history_size=3, fractions=(0.5,)) == (
        (0, 62),
        (2, 3),
    )
    with pytest.raises(ValueError, match="no episode"):
        plan_audit_windows([SimpleNamespace(count=1)], history_size=3)
    with pytest.raises(ValueError, match="fractions"):
        plan_audit_windows(records, history_size=3, fractions=())
    with pytest.raises(ValueError, match="fractions"):
        plan_audit_windows(records, history_size=3, fractions=(1.5,))
    with pytest.raises(TypeError, match="integer"):
        plan_audit_windows(records, history_size=True)


def test_action_audit_reports_interventions_and_splits() -> None:
    model = ActionTraceModel()
    windows = [
        {**_window([0.0, 1.0, 3.0, 6.0], [1, 2, 3]), "episode_id": "e0", "start": 0},
        {**_window([0.0, 0.0, 0.0, 1.0], [0, 0, 0]), "episode_id": "e1", "start": 5},
    ]

    result = audit_action_conditioning(model, windows, "cpu")

    assert result["window_count"] == 2
    assert result["pixel_change_threshold"] == pytest.approx(5.0)
    assert result["prediction_mse"] == pytest.approx(0.5)
    assert result["persistence_mse"] == pytest.approx(5.0)
    assert result["prediction_vs_persistence_ratio"] == pytest.approx(0.1)
    assert result["prediction_win_rate"] == pytest.approx(0.5)
    assert result["recorded_action_best_rate"] == pytest.approx(0.5)
    assert result["wrong_action_mean_mse"] == pytest.approx(13.0625)
    assert result["move_from_current_mse"] == pytest.approx(4.5)
    assert result["action_spread_mse"] == pytest.approx(17.0625)
    assert result["prediction_quantiles"]["p50"] == pytest.approx(0.5)
    assert result["scope"] == "one-step teacher-forced action audit"

    first, second = result["windows"]
    assert (first["episode_id"], first["start"]) == ("e0", 0)
    assert first["prediction_mse"] == pytest.approx(0.0)
    assert first["persistence_mse"] == pytest.approx(9.0)
    assert first["wrong_action_mean_mse"] == pytest.approx(8.625)
    assert first["wrong_action_min_mse"] == pytest.approx(1.0)
    assert first["recorded_action_best"] is True
    assert first["action_changed"] is True
    assert second["prediction_mse"] == pytest.approx(1.0)
    assert second["wrong_action_mean_mse"] == pytest.approx(17.5)
    assert second["wrong_action_min_mse"] == pytest.approx(0.0)
    assert second["recorded_action_best"] is False
    assert second["move_from_current_mse"] == pytest.approx(0.0)
    assert second["action_spread_mse"] == pytest.approx(25.5)
    assert second["action_changed"] is False

    assert result["changing"]["window_count"] == 1
    assert result["changing"]["prediction_win_rate"] == pytest.approx(1.0)
    assert result["static"]["window_count"] == 1
    assert result["static"]["prediction_win_rate"] == pytest.approx(0.0)
    assert result["static"]["recorded_action_best_rate"] == pytest.approx(0.0)


def test_action_audit_is_read_only_and_guards_degenerate_cases() -> None:
    model = ActionTraceModel().train()
    before_rng = torch.random.get_rng_state()

    result = audit_action_conditioning(
        model, [_window([4.0, 4.0, 4.0, 4.0], [0, 0, 0])], "cpu"
    )

    assert model.training is True
    assert model.scale.grad is None
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert result["prediction_vs_persistence_ratio"] is None
    assert result["prediction_win_rate"] == pytest.approx(0.0)
    assert result["recorded_action_best_rate"] == pytest.approx(1.0)
    assert result["changing"]["window_count"] == 0
    assert result["changing"]["prediction_mse"] is None

    with pytest.raises(ValueError, match="action range"):
        audit_action_conditioning(
            ActionTraceModel(), [_window([0.0, 0.0, 0.0, 1.0], [0, 0, 9])], "cpu"
        )
    with pytest.raises(ValueError, match="at least two"):
        audit_action_conditioning(
            ActionTraceModel(), [_window([0.0, 0.0, 0.0, 1.0], [0, 0, 0])], "cpu",
            action_dim=1,
        )
