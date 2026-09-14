from __future__ import annotations

import torch
from torch import nn

from dodge_native_game.variants.pixel_repr_ddqn.dynamics import evaluate_dynamics


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
