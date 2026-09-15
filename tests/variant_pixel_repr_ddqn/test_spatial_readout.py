from __future__ import annotations

import pytest
import torch

from dodge_native_game.variants.pixel_repr_ddqn.model import (
    LeWMConfig,
    LeWorldModel,
)
from dodge_native_game.variants.pixel_repr_ddqn.spatial_readout import (
    LocalPatchDecoder,
    broadcast_cls,
    pixel_patch_features,
)

PALETTE = (
    (29, 43, 83),
    (41, 173, 255),
    (255, 241, 232),
)


def _patch_pixels() -> torch.Tensor:
    colors = torch.tensor(PALETTE, dtype=torch.uint8)
    ids = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)
    pixels = torch.empty((1, 3, 16, 16), dtype=torch.uint8)
    for row in range(2):
        for column in range(2):
            color = colors[ids[row, column]]
            pixels[:, :, row * 8 : (row + 1) * 8, column * 8 : (column + 1) * 8] = (
                color.view(1, 3, 1, 1)
            )
    return pixels


def test_pixel_patch_features_use_raster_order_and_channel_major_flattening() -> None:
    features = pixel_patch_features(_patch_pixels(), PALETTE)
    assert features.shape == (1, 4, 3 * 8 * 8)
    expected_ids = (0, 1, 2, 0)
    for patch, class_id in enumerate(expected_ids):
        expected = torch.zeros(3, 8, 8)
        expected[class_id] = 1.0
        torch.testing.assert_close(
            features[0, patch], expected.reshape(-1), rtol=0, atol=0
        )


def test_local_decoder_has_fixed_xy_grid_and_cls_control() -> None:
    decoder = LocalPatchDecoder(
        feature_dim=4,
        hidden_dim=8,
        grid_size=2,
        patch_size=2,
    ).eval()
    expected_grid = torch.tensor(
        [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]
    )
    torch.testing.assert_close(decoder.xy_grid, expected_grid, rtol=0, atol=0)
    assert "xy_grid" not in decoder.state_dict()
    assert "xy_grid" not in dict(decoder.named_parameters())

    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.zero_()
        decoder.mlp[0].weight[0, -2] = 1.0
        decoder.mlp[2].weight[0, 0] = 1.0
        decoder.mlp[4].weight[0, 0] = 1.0
        decoder.mlp[4].weight[0, 0] = 1.0
        values = decoder(broadcast_cls(torch.zeros(1, 4), num_patches=4))
    assert values.shape == (1, 3, 4, 4)
    assert not torch.equal(values[:, :, :, :2], values[:, :, :, 2:])


def test_local_decoder_has_no_cross_patch_dependency() -> None:
    torch.manual_seed(4)
    decoder = LocalPatchDecoder(feature_dim=4, hidden_dim=8, grid_size=2, patch_size=2)
    baseline_features = torch.zeros(1, 4, 4)
    changed_features = baseline_features.clone()
    changed_features[:, 1, 0] = 1.0
    with torch.no_grad():
        baseline = decoder(baseline_features)
        changed = decoder(changed_features)
    difference = (changed - baseline).abs()
    assert difference[:, :, :2, :2].sum() == 0
    assert difference[:, :, 2:, :2].sum() == 0
    assert difference[:, :, 2:, 2:].sum() == 0
    assert difference[:, :, :2, 2:].sum() > 0


def test_pixel_patch_features_round_trip_through_unpatchify_for_a_batch() -> None:
    first = _patch_pixels()
    second = first.flip(-1).contiguous()
    pixels = torch.cat((first, second), dim=0)
    features = pixel_patch_features(pixels, PALETTE)
    decoder = LocalPatchDecoder(grid_size=2, patch_size=8).eval()
    with torch.no_grad():
        reconstructed = decoder.unpatchify(features)
    expected = torch.nn.functional.one_hot(
        torch.tensor(
            [
                [[0, 1], [2, 0]],
                [[1, 0], [0, 2]],
            ],
            dtype=torch.long,
        ),
        num_classes=3,
    ).permute(0, 3, 1, 2).repeat_interleave(8, dim=2).repeat_interleave(8, dim=3)
    torch.testing.assert_close(reconstructed, expected.to(dtype=torch.float32))


def test_readout_tokens_preserve_cls_and_return_final_patch_tokens() -> None:
    torch.manual_seed(7)
    model = LeWorldModel(LeWMConfig.tiny()).eval()
    pixels = torch.randint(0, 256, (2, 3, 3, 32, 32), dtype=torch.uint8)
    with torch.no_grad():
        expected_cls = model.encode_cls(pixels)
        cls, patches = model.encode_readout_tokens(pixels)
    assert cls.shape == (2, 3, model.config.embed_dim)
    assert patches.shape == (2, 3, model.config.num_patches, model.config.embed_dim)
    torch.testing.assert_close(cls, expected_cls, rtol=0, atol=0)
    assert torch.isfinite(patches).all()


def test_spatial_readout_finite_and_palette_guards() -> None:
    decoder = LocalPatchDecoder(feature_dim=4, grid_size=2, patch_size=2)
    with pytest.raises(ValueError, match="nonfinite"):
        decoder(torch.full((1, 4, 4), float("nan")))
    with pytest.raises(ValueError, match="nonfinite"):
        broadcast_cls(torch.full((1, 4), float("inf")), num_patches=4)
    unknown = _patch_pixels()
    unknown[:, :, 0, 0] = torch.tensor([0, 0, 0], dtype=torch.uint8)
    with pytest.raises(ValueError, match="outside configured palette"):
        pixel_patch_features(unknown, PALETTE)
