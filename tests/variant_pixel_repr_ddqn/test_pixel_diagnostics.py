from __future__ import annotations

import pytest
import torch
from torch import nn

from dodge_native_game.variants.pixel_repr_ddqn.pixel_diagnostics import evaluate_pixels


class IdentityToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("marker", torch.tensor([7.0]))
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        self.calls.append((pixels.detach().clone(), torch.empty(0)))
        return pixels.float().flatten(2) / 255.0 * self.scale

    def predict(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        self.calls[-1] = (self.calls[-1][0], actions.detach().clone())
        torch.rand((), device=z.device)
        return z


class IdentityToyDecoder(nn.Module):
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        torch.rand((), device=latent.device)
        return latent.reshape(-1, 3, 32, 32)


def _frame(value: int, *, moving: tuple[int, int] | None = None) -> torch.Tensor:
    frame = torch.full((3, 32, 32), value, dtype=torch.uint8)
    if moving is not None:
        row, column = moving
        frame[:, row, column] = 255 - value
    return frame


def _window(
    current: torch.Tensor,
    following: torch.Tensor,
    episode_id: str,
) -> dict[str, object]:
    pixels = torch.stack([current, current, current, following])
    return {
        "pixels": pixels,
        "actions": torch.tensor([0, 1, 2]),
        "episode_id": episode_id,
    }


def test_pixel_controls_use_train_mean_and_changed_region_aggregates() -> None:
    train = [_window(_frame(0), _frame(0), "train")]
    validation = [
        _window(_frame(0), _frame(0, moving=(2, 3)), "episode-a"),
        _window(_frame(0), _frame(0), "episode-b"),
    ]

    result = evaluate_pixels(
        IdentityToyModel().eval(),
        IdentityToyDecoder().eval(),
        train,
        validation,
        "cpu",
    )

    assert result["window_count"] == 2
    assert result["total_pixel_count"] == 2 * 32 * 32
    assert result["changed_pixel_count"] == 1
    assert result["changed_pixel_fraction"] == 1 / (2 * 32 * 32)
    assert result["current_reconstruction_mse"] == 0.0
    assert result["current_mean_image_mse"] == 0.0
    assert result["next_persistence_mse"] == 3 / (3 * 32 * 32 * 2)
    assert result["next_prediction_mse"] == result["next_persistence_mse"]
    assert result["next_mean_image_mse"] == result["next_persistence_mse"]
    assert result["changed_pixel_prediction_mse"] == 1.0
    assert result["changed_pixel_persistence_mse"] == 1.0
    assert result["changed_pixel_mean_image_mse"] == 1.0
    assert result["prediction_vs_persistence_ratio"] == 1.0
    assert result["changed_pixel_prediction_vs_persistence_ratio"] == 1.0
    assert result["per_episode"]["episode-a"]["window_count"] == 1
    assert result["per_episode"]["episode-b"]["changed_pixel_count"] == 0


def test_train_mean_does_not_use_validation_windows() -> None:
    train = [_window(_frame(0), _frame(0), "train")]
    validation = [_window(_frame(255), _frame(255), "validation")]

    result = evaluate_pixels(
        IdentityToyModel(), IdentityToyDecoder(), train, validation, "cpu"
    )

    assert result["current_mean_image_mse"] == 1.0
    assert result["next_mean_image_mse"] == 1.0


def test_zero_changed_pixels_report_null_changed_metrics() -> None:
    result = evaluate_pixels(
        IdentityToyModel(),
        IdentityToyDecoder(),
        [_window(_frame(64), _frame(64), "train")],
        [_window(_frame(64), _frame(64), "same")],
        "cpu",
    )

    assert result["changed_pixel_count"] == 0
    assert result["changed_pixel_fraction"] == 0.0
    assert result["changed_pixel_prediction_mse"] is None
    assert result["changed_pixel_persistence_mse"] is None
    assert result["changed_pixel_mean_image_mse"] is None
    assert result["changed_pixel_prediction_vs_persistence_ratio"] is None


def test_evaluation_preserves_modes_rng_and_state() -> None:
    model = IdentityToyModel().train()
    decoder = IdentityToyDecoder().train()
    model_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    decoder_state = {
        key: value.detach().clone() for key, value in decoder.state_dict().items()
    }
    before_rng = torch.random.get_rng_state()

    evaluate_pixels(
        model,
        decoder,
        [_window(_frame(0), _frame(0), "train")],
        [_window(_frame(0), _frame(0, moving=(4, 5)), "validation")],
        "cpu",
    )

    assert model.training is True
    assert decoder.training is True
    assert model.scale.grad is None
    assert torch.equal(before_rng, torch.random.get_rng_state())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, model_state[key], rtol=0, atol=0)
    for key, value in decoder.state_dict().items():
        torch.testing.assert_close(value, decoder_state[key], rtol=0, atol=0)


def test_nonfinite_decoder_fails_and_restores_modes_rng():
    class BrokenDecoder(IdentityToyDecoder):
        def forward(self, latent):
            return super().forward(latent) * float("nan")

    model, decoder = IdentityToyModel().train(), BrokenDecoder().train()
    window = _window(_frame(0), _frame(0, moving=(1, 1)), "test")
    before = torch.get_rng_state()
    with pytest.raises(RuntimeError, match="nonfinite"):
        evaluate_pixels(model, decoder, [window], [window], "cpu")
    assert model.training and decoder.training
    assert torch.equal(before, torch.get_rng_state())


def test_native_resolution_preserves_small_pixel_targets():
    class FlatDecoder(nn.Module):
        def forward(self, latent):
            return latent

    current = torch.zeros(3, 128, 128, dtype=torch.uint8)
    following = current.clone()
    following[:, 40, 60] = 255
    window = _window(current, following, "native")
    model, decoder = IdentityToyModel(), FlatDecoder()
    full = evaluate_pixels(model, decoder, [window], [window], "cpu", output_size=128)
    reduced = evaluate_pixels(model, decoder, [window], [window], "cpu")
    assert full["output_size"] == 128
    assert full["total_pixel_count"] == 128 * 128
    assert full["changed_pixel_count"] == 1
    assert full["changed_pixel_persistence_mse"] == 1.0
    assert reduced["output_size"] == 32
    assert reduced["changed_pixel_persistence_mse"] == 1 / 256
    assert full["current_reconstruction_mse"] == 0.0


def test_changed_current_reconstruction_uses_current_target():
    train = [_window(_frame(0), _frame(0), "train")]
    validation = [_window(_frame(0, moving=(2, 3)), _frame(0), "validation")]
    result = evaluate_pixels(
        IdentityToyModel(), IdentityToyDecoder(), train, validation, "cpu"
    )
    assert result["changed_pixel_current_reconstruction_mse"] == 0.0
    assert result["changed_pixel_current_mean_image_mse"] == 1.0
    assert result["changed_pixel_prediction_mse"] == 1.0
    assert result["changed_pixel_mean_image_mse"] == 0.0
