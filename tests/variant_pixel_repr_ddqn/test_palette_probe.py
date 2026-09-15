from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from dodge_native_game.variants.pixel_repr_ddqn.large_probe import (
    bce_palette_loss,
    ce_palette_loss,
    evaluate_decoder_stream,
    fit_matched_decoders,
)
from dodge_native_game.variants.pixel_repr_ddqn.palette import (
    PaletteCoverageError,
    derive_palette,
    palette_indices,
    palette_one_hot,
    render_palette,
    render_palette_rgb,
    validate_palette_coverage,
)
from dodge_native_game.variants.pixel_repr_ddqn.query_decoder import QueryPixelDecoder


def _pixels() -> np.ndarray:
    return np.asarray(
        [
            [
                [[10, 20], [10, 20]],
                [[30, 30], [40, 40]],
                [[50, 50], [50, 50]],
            ],
            [
                [[20, 10], [20, 10]],
                [[30, 30], [40, 40]],
                [[50, 50], [50, 50]],
            ],
        ],
        dtype=np.uint8,
    )


def test_palette_is_sorted_train_only_and_validation_cannot_extend_it() -> None:
    pixels = _pixels()
    palette = derive_palette(pixels[:1], range(1), batch_size=1)
    np.testing.assert_array_equal(
        palette,
        np.asarray(
            [[10, 30, 50], [10, 40, 50], [20, 30, 50], [20, 40, 50]],
            dtype=np.uint8,
        ),
    )
    with pytest.raises(PaletteCoverageError, match="validation"):
        validate_palette_coverage(
            np.concatenate([pixels[1:], np.full((1, 3, 2, 2), 255, dtype=np.uint8)]),
            range(2),
            palette,
            batch_size=1,
        )


def test_palette_cap_fails_without_leaking_validation_colors() -> None:
    values = np.zeros((1, 3, 1, 257), dtype=np.uint8)
    values[:, 0, 0] = np.arange(257, dtype=np.uint16) % 256
    values[:, 1, 0] = np.arange(257, dtype=np.uint16) // 256
    with pytest.raises(ValueError, match="palette exceeds cap 256"):
        derive_palette(values, range(1), batch_size=17)


def test_palette_targets_losses_and_exact_argmax_rgb_rendering() -> None:
    palette = np.asarray([[0, 10, 20], [30, 40, 50]], dtype=np.uint8)
    pixels = np.asarray(
        [[[[0, 30]], [[10, 40]], [[20, 50]]]], dtype=np.uint8
    )
    targets = torch.from_numpy(palette_indices(pixels, palette))
    assert targets.tolist() == [[[0, 1]]]
    torch.testing.assert_close(
        palette_one_hot(targets, 2),
        torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]]),
    )
    logits = torch.tensor([[[[5.0, -5.0]], [[-5.0, 5.0]]]], requires_grad=True)
    ce = ce_palette_loss(logits, targets)
    bce = bce_palette_loss(logits, targets)
    (ce + bce).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    rendered = render_palette(targets.numpy(), palette, logits=False)
    np.testing.assert_array_equal(rendered, pixels)
    np.testing.assert_array_equal(
        render_palette(logits.detach(), palette), pixels
    )
    torch.testing.assert_close(
        render_palette_rgb(logits.detach(), palette),
        torch.from_numpy(pixels).float() / 255,
    )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_palette_render_rejects_nonfinite_logits(bad_value: float) -> None:
    palette = np.asarray([[0, 10, 20], [30, 40, 50]], dtype=np.uint8)
    with pytest.raises(ValueError, match="nonfinite"):
        render_palette_rgb(
            torch.tensor([[[[bad_value]], [[0.0]]]]),
            palette,
        )


def test_query_decoder_default_rgb_contract_and_palette_logits() -> None:
    default = QueryPixelDecoder(
        latent_dim=4, hidden_dim=8, depth=1, heads=2, output_size=4, patch_size=2
    )
    raw = QueryPixelDecoder(
        latent_dim=4,
        hidden_dim=8,
        depth=1,
        heads=2,
        output_size=4,
        patch_size=2,
        output_channels=5,
        raw_logits=True,
    )
    latent = torch.randn(2, 4)
    assert default(latent).shape == (2, 3, 4, 4)
    assert raw(latent).shape == (2, 5, 4, 4)
    assert torch.equal(raw(latent), raw.forward_logits(latent))


class _PaletteTinyDecoder(nn.Module):
    def __init__(self, latent_dim: int, classes: int = 2) -> None:
        super().__init__()
        self.projection = nn.Linear(latent_dim, classes * 2 * 2)
        self.classes = classes

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.projection(latent).reshape(-1, self.classes, 2, 2)


def test_palette_fit_updates_both_matched_heads_with_same_samples() -> None:
    palette = np.asarray([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    pixels = np.zeros((8, 3, 2, 2), dtype=np.uint8)
    pixels[::2] = 255
    features = torch.randn(8, 4, generator=torch.Generator().manual_seed(5))
    left = _PaletteTinyDecoder(4)
    right = _PaletteTinyDecoder(4)
    right.load_state_dict(left.state_dict())
    result = fit_matched_decoders(
        left,
        right,
        features,
        features,
        pixels,
        milestones=(1, 2),
        batch_size=4,
        loss_kind="palette-ce",
        palette=palette,
        sampler=torch.Generator().manual_seed(903),
    )
    assert result.sampled_indices[0] == tuple(
        torch.randint(8, (4,), generator=torch.Generator().manual_seed(903)).tolist()
    )
    assert all(torch.isfinite(value).all() for value in left.parameters())
    assert all(torch.isfinite(value).all() for value in right.parameters())


class _FixedPaletteDecoder(nn.Module):
    def forward_logits(self, latent: torch.Tensor) -> torch.Tensor:
        logits = torch.tensor(
            [[[[10.0, -10.0], [-10.0, 10.0]], [[-10.0, 10.0], [10.0, -10.0]]]]
        )
        return logits.expand(latent.shape[0], -1, -1, -1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.forward_logits(latent)


def test_palette_evaluation_reports_global_and_changed_confusion() -> None:
    palette = np.asarray([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    pixels = np.zeros((2, 3, 2, 2), dtype=np.uint8)
    pixels[0, :, 0, 1] = 255
    pixels[0, :, 1, 0] = 255
    pixels[1, :, 1, 0] = 255
    features = np.zeros((2, 1), dtype=np.float32)
    changed = np.zeros((2, 2, 2), dtype=bool)
    changed[:, 0, 1] = True
    records = [
        {"episode_id": "train", "frame_index": 1},
        {"episode_id": "validation", "frame_index": 1},
    ]
    result = evaluate_decoder_stream(
        _FixedPaletteDecoder(),
        features,
        pixels,
        changed,
        records,
        {"train": (0, 1), "validation": (1, 2)},
        np.zeros((3, 2, 2), dtype=np.float32),
        device="cpu",
        batch_size=2,
        palette=palette,
    )
    assert result["splits"]["train"]["palette_pixel_accuracy"] == 1.0
    assert result["splits"]["train"]["palette_changed_pixel_accuracy"] == 1.0
    assert result["splits"]["train"]["palette_confusion_matrix"] == [[2, 0], [0, 2]]
