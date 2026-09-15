from __future__ import annotations

import json
from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F

from dodge_native_game.variants.pixel_repr_ddqn.model import (
    INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC,
    INPUT_ENCODING_RGB_NEAREST_SYMMETRIC,
    LeWMConfig,
    LeWorldModel,
)

PALETTE = (
    (29, 43, 83),
    (41, 173, 255),
    (255, 241, 232),
)


def _small_config(**overrides: object) -> LeWMConfig:
    return LeWMConfig.tiny(image_size=4, patch_size=2, **overrides)


def _palette_pixels() -> torch.Tensor:
    classes = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)
    colors = torch.tensor(PALETTE, dtype=torch.uint8)
    frame = colors[classes].permute(2, 0, 1)
    return frame.unsqueeze(0).unsqueeze(0)


def test_input_encoding_config_canonicalizes_palette_and_round_trips() -> None:
    config = _small_config(
        input_encoding=INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC,
        palette_rgb=[list(color) for color in PALETTE],
    )
    assert config.palette_rgb == PALETTE
    assert isinstance(config.palette_rgb, tuple)
    payload = json.loads(json.dumps(asdict(config)))
    assert LeWMConfig(**payload) == config

    with pytest.raises(ValueError, match="input_encoding"):
        _small_config(input_encoding="unknown")
    with pytest.raises(ValueError, match="palette_rgb is required"):
        _small_config(input_encoding=INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC)
    with pytest.raises(ValueError, match="requires K=3"):
        _small_config(
            input_encoding=INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC,
            palette_rgb=PALETTE[:2],
        )
    with pytest.raises(ValueError, match="sorted"):
        _small_config(
            input_encoding=INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC,
            palette_rgb=tuple(reversed(PALETTE)),
        )
    with pytest.raises(ValueError, match="only valid"):
        _small_config(palette_rgb=PALETTE)


def test_rgb_nearest_symmetric_is_exact_and_legacy_stays_byte_compatible() -> None:
    pixels = _palette_pixels()
    rgb_model = LeWorldModel(
        _small_config(input_encoding=INPUT_ENCODING_RGB_NEAREST_SYMMETRIC)
    ).eval()
    with torch.no_grad():
        actual = rgb_model._prepare_pixels(pixels)
    expected = pixels.float().div(255.0).reshape(1, 3, 2, 2)
    expected = F.interpolate(expected, size=(4, 4), mode="nearest")
    expected = expected.mul(2.0).sub(1.0).reshape(1, 1, 3, 4, 4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    legacy_model = LeWorldModel(_small_config()).eval()
    with torch.no_grad():
        legacy_actual = legacy_model._prepare_pixels(pixels)
    legacy_expected = pixels.float().div(255.0).reshape(1, 3, 2, 2)
    legacy_expected = (
        legacy_expected - legacy_model.image_mean
    ) / legacy_model.image_std
    legacy_expected = F.interpolate(
        legacy_expected,
        size=(4, 4),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).reshape(1, 1, 3, 4, 4)
    torch.testing.assert_close(legacy_actual, legacy_expected, rtol=0, atol=0)


def test_palette_onehot_lookup_precedes_nearest_resize_and_rejects_unknowns() -> None:
    pixels = _palette_pixels()
    model = LeWorldModel(
        _small_config(
            input_encoding=INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC,
            palette_rgb=PALETTE,
        )
    ).eval()
    with torch.no_grad():
        actual = model._prepare_pixels(pixels)
        float_actual = model._prepare_pixels(pixels.float().div(255.0))
    classes = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)
    expected = F.one_hot(classes, num_classes=3).permute(2, 0, 1).float()
    expected = F.interpolate(expected.unsqueeze(0), size=(4, 4), mode="nearest")
    expected = expected.mul(2.0).sub(1.0).unsqueeze(0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(float_actual, actual, rtol=0, atol=0)

    unknown = pixels.clone()
    unknown[:, :, :, 0, 0] = torch.tensor([0, 0, 0], dtype=torch.uint8)
    with pytest.raises(ValueError, match="outside configured palette"):
        model._prepare_pixels(unknown)

    non_native = pixels.float().div(255.0)
    non_native[:, :, :, 0, 0] += 0.001
    with pytest.raises(ValueError, match="exact native RGB"):
        model._prepare_pixels(non_native)


def test_old_config_strict_load_and_no_new_buffers() -> None:
    old_payload = asdict(LeWMConfig.tiny())
    old_payload.pop("input_encoding")
    old_payload.pop("palette_rgb")
    old_config = LeWMConfig(**old_payload)
    torch.manual_seed(12)
    source = LeWorldModel(old_config)
    restored = LeWorldModel(old_config)
    restored.load_state_dict(source.state_dict(), strict=True)
    assert "input_encoding" not in restored.state_dict()
    assert "palette_rgb" not in restored.state_dict()
    for name, value in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
