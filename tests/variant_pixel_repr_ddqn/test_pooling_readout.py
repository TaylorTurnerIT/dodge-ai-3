from __future__ import annotations

import hashlib

import pytest
import torch

from dodge_native_game.variants.pixel_repr_ddqn.pooling_readout import (
    ATTENTION,
    CONDITIONS,
    GRID4,
    GRID4_BLOCK_SIZE,
    MAX,
    MEAN,
    PoolingReadout,
    make_pooling_decoders,
    pool_features,
)
from dodge_native_game.variants.pixel_repr_ddqn.spatial_fit import make_spatial_decoders


def _features(batch: int = 2, side: int = 8, dim: int = 3) -> torch.Tensor:
    values = torch.arange(batch * side * side * dim, dtype=torch.float32)
    return values.reshape(batch, side * side, dim) / 17.0


def _state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def test_pooling_modes_preserve_shape_and_expected_order() -> None:
    features = _features(side=8)
    mean = pool_features(features, MEAN, feature_dim=3, grid_size=8)
    maximum = pool_features(features, MAX, feature_dim=3, grid_size=8)
    attention = pool_features(features, ATTENTION, feature_dim=3, grid_size=8)
    grid4 = pool_features(features, GRID4, feature_dim=3, grid_size=8)

    expected_mean = features.mean(dim=1, keepdim=True).expand_as(features)
    expected_max = features.amax(dim=1, keepdim=True).expand_as(features)
    blocks = features.reshape(2, 8, 8, 3).reshape(2, 2, 4, 2, 4, 3)
    expected_grid4 = blocks.mean(dim=(2, 4)).repeat_interleave(
        GRID4_BLOCK_SIZE, dim=1
    ).repeat_interleave(GRID4_BLOCK_SIZE, dim=2).reshape_as(features)

    for actual in (mean, maximum, attention, grid4):
        assert actual.shape == features.shape
        assert torch.isfinite(actual).all()
    torch.testing.assert_close(mean, expected_mean)
    torch.testing.assert_close(maximum, expected_max)
    torch.testing.assert_close(attention, expected_mean)
    torch.testing.assert_close(grid4, expected_grid4)


def test_global_modes_are_permutation_invariant_and_grid4_is_block_ordered() -> None:
    features = _features(batch=1, side=8)
    permutation = torch.arange(64)
    original_zero, original_thirty_six = permutation[0].clone(), permutation[36].clone()
    permutation[0], permutation[36] = original_thirty_six, original_zero
    permuted = features[:, permutation]
    query = torch.tensor([0.5, -0.25, 0.75])
    for mode in (MEAN, MAX, ATTENTION):
        original = pool_features(
            features, mode, query=query, feature_dim=3, grid_size=8
        )
        shuffled = pool_features(
            permuted, mode, query=query, feature_dim=3, grid_size=8
        )
        torch.testing.assert_close(original, shuffled)
    original = pool_features(features, GRID4, feature_dim=3, grid_size=8)
    shuffled = pool_features(permuted, GRID4, feature_dim=3, grid_size=8)
    assert not torch.equal(original, shuffled)


def test_global_modes_change_every_position_but_grid4_stays_local() -> None:
    features = torch.zeros(1, 64, 3)
    changed = features.clone()
    changed[:, 0, 0] = 1.0
    for mode in (MEAN, MAX, ATTENTION):
        difference = pool_features(
            changed, mode, feature_dim=3, grid_size=8
        ) - pool_features(features, mode, feature_dim=3, grid_size=8)
        assert torch.all(difference[:, :, 0] > 0)
    difference = pool_features(
        changed, GRID4, feature_dim=3, grid_size=8
    ) - pool_features(features, GRID4, feature_dim=3, grid_size=8)
    affected = torch.zeros(64, dtype=torch.bool)
    affected.reshape(8, 8)[:4, :4] = True
    assert difference[:, affected, 0].ne(0).all()
    assert difference[:, ~affected, 0].eq(0).all()


def test_attention_query_is_zero_initialized_and_only_attention_learns() -> None:
    decoders = make_pooling_decoders(
        device="cpu",
        seed=904,
        feature_dim=3,
        hidden_dim=8,
        grid_size=8,
        patch_size=1,
        output_channels=1,
    )
    assert set(decoders) == set(CONDITIONS)
    assert all(
        torch.equal(decoder.query, torch.zeros(3)) for decoder in decoders.values()
    )
    assert decoders[ATTENTION].query.requires_grad
    assert all(
        not decoders[mode].query.requires_grad
        for mode in (MEAN, MAX, GRID4)
    )
    features = _features(batch=1, side=8)
    attention_loss = decoders[ATTENTION].pooled_features(features)[:, 0].sum()
    attention_loss.backward()
    assert decoders[ATTENTION].query.grad is not None
    assert torch.linalg.vector_norm(decoders[ATTENTION].query.grad) > 0
    frozen_features = features.detach().requires_grad_()
    mean_loss = decoders[MEAN].pooled_features(frozen_features)[:, 0].sum()
    mean_loss.backward()
    assert decoders[MEAN].query.grad is None


def test_pooling_decoders_have_identical_core_state_and_parameter_counts() -> None:
    decoders = make_pooling_decoders(
        device="cpu",
        seed=904,
        feature_dim=3,
        hidden_dim=8,
        grid_size=8,
        patch_size=1,
        output_channels=1,
    )
    hashes = {_state_hash(decoder) for decoder in decoders.values()}
    core_hashes = {_state_hash(decoder.core) for decoder in decoders.values()}
    parameter_counts = {
        sum(parameter.numel() for parameter in decoder.parameters())
        for decoder in decoders.values()
    }
    core_parameter_counts = {
        sum(parameter.numel() for parameter in decoder.core.parameters())
        for decoder in decoders.values()
    }
    assert len(hashes) == 1
    assert len(core_hashes) == 1
    assert len(parameter_counts) == 1
    assert len(core_parameter_counts) == 1
    assert parameter_counts == {next(iter(core_parameter_counts)) + 3}

    repeated = make_pooling_decoders(
        device="cpu",
        seed=904,
        feature_dim=3,
        hidden_dim=8,
        grid_size=8,
        patch_size=1,
        output_channels=1,
    )
    assert _state_hash(decoders[MEAN]) == _state_hash(repeated[MEAN])


def test_pooling_core_matches_the_spatial_seed904_initialization() -> None:
    pooling = make_pooling_decoders(seed=904)
    spatial = make_spatial_decoders(seed=904)
    reference = spatial["cls"].state_dict()
    for decoder in pooling.values():
        actual = decoder.decoder.state_dict()
        assert tuple(actual) == tuple(reference)
        assert all(torch.equal(actual[name], reference[name]) for name in reference)


def test_pooling_readout_decodes_default_native_shape() -> None:
    decoder = PoolingReadout(ATTENTION).eval()
    values = decoder(torch.randn(2, 256, 192))
    assert values.shape == (2, 3, 128, 128)
    assert torch.isfinite(values).all()


def test_pooling_rejects_nonfinite_and_invalid_grid4_inputs() -> None:
    with pytest.raises(ValueError, match="nonfinite"):
        pool_features(
            torch.full((1, 64, 3), float("nan")),
            MEAN,
            grid_size=8,
            feature_dim=3,
        )
    with pytest.raises(ValueError, match="divisible by four"):
        PoolingReadout(GRID4, feature_dim=3, grid_size=6, patch_size=1)
