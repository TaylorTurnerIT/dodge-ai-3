from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch import nn

from dodge_native_game.variants.pixel_repr_ddqn import spatial_fit


class _Rows:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.accesses: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, key: object) -> np.ndarray:
        self.accesses.append(np.asarray(key).copy())
        return self.values[key]  # type: ignore[index]


class _TinyLocalPatchDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 192,
        output_channels: int = 3,
        raw_logits: bool = True,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(latent_dim, output_channels)
        self.raw_logits = raw_logits

    def forward_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        values = self.projection(tokens).reshape(-1, 16, 16, 3)
        values = values.permute(0, 3, 1, 2)
        return values.repeat_interleave(8, dim=2).repeat_interleave(8, dim=3)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_logits(tokens)


class _NaNLocalPatchDecoder(_TinyLocalPatchDecoder):
    def forward_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return super().forward_logits(tokens) * torch.tensor(float("nan"))


def _palette() -> np.ndarray:
    return np.asarray(
        [[29, 43, 83], [41, 173, 255], [255, 241, 232]], dtype=np.uint8
    )


def _sources(rows: int = 8) -> tuple[_Rows, dict[str, _Rows], np.ndarray]:
    palette = _palette()
    features = np.zeros((rows, 256, 192), dtype=np.float32)
    targets = np.empty((rows, 3, 128, 128), dtype=np.uint8)
    patch_grid = np.arange(256, dtype=np.int64).reshape(16, 16)
    for row in range(rows):
        classes = (patch_grid + row) % 3
        features[row, np.arange(256), classes.reshape(-1)] = 1.0
        rgb = palette[classes]
        image = np.repeat(np.repeat(rgb, 8, axis=0), 8, axis=1)
        targets[row] = image.transpose(2, 0, 1)
    target_rows = _Rows(targets)
    feature_rows = {
        condition: _Rows(features.copy()) for condition in spatial_fit.CONDITIONS
    }
    return target_rows, feature_rows, palette


def _make_decoders(
    monkeypatch: pytest.MonkeyPatch, decoder_type=_TinyLocalPatchDecoder
):
    monkeypatch.setattr(spatial_fit, "LocalPatchDecoder", decoder_type)
    return spatial_fit.make_spatial_decoders(device="cpu", seed=904)


def test_make_spatial_decoders_and_fit_are_matched_and_learn_local_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets, features, palette = _sources()
    decoders = _make_decoders(monkeypatch)
    initial = {
        condition: copy.deepcopy(decoder.state_dict())
        for condition, decoder in decoders.items()
    }
    callbacks: list[tuple[int, dict[str, nn.Module]]] = []
    metrics: list[dict[str, object]] = []
    torch.manual_seed(117)
    before_rng = torch.get_rng_state().clone()

    result = spatial_fit.fit_spatial_decoders(
        decoders,
        features,
        targets,
        palette,
        milestones=(1, 4),
        batch_size=4,
        on_step=metrics.append,
        on_milestone=lambda snapshot, heads: callbacks.append(
            (snapshot.step, dict(heads))
        ),
    )

    assert torch.equal(torch.get_rng_state(), before_rng)
    assert set(decoders) == set(spatial_fit.CONDITIONS)
    assert len(result.metrics) == 4
    assert len(metrics) == 4
    assert set(result.snapshots) == {1, 4}
    assert [step for step, _ in callbacks] == [1, 4]
    assert all(
        all(key in metric for key in ("cls_loss", "patch_loss", "pixels_loss"))
        for metric in metrics
    )
    expected_sampler = torch.Generator(device="cpu").manual_seed(903)
    expected_indices = tuple(
        tuple(torch.randint(8, (4,), generator=expected_sampler).tolist())
        for _ in range(4)
    )
    assert result.sampled_indices == expected_indices
    for condition in spatial_fit.CONDITIONS:
        assert any(
            not torch.equal(initial[condition][key], value)
            for key, value in decoders[condition].state_dict().items()
        )
        assert result.snapshots[4].model[condition]
        assert result.snapshots[4].optimizer[condition]["state"]
        assert targets.accesses
        assert all(np.max(access) < len(targets) for access in targets.accesses)
        assert len(features[condition].accesses) == len(targets.accesses)
        assert all(
            np.array_equal(left, right)
            for left, right in zip(
                features[condition].accesses, targets.accesses, strict=True
            )
        )
    for _, heads in callbacks:
        assert set(heads) == set(spatial_fit.CONDITIONS)


def test_spatial_fit_rejects_invalid_sources_palette_and_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets, features, palette = _sources()
    decoders = _make_decoders(monkeypatch)
    with pytest.raises(ValueError, match=r"exactly \('cls', 'patch', 'pixels'\)"):
        spatial_fit.fit_spatial_decoders(
            {"cls": decoders["cls"]}, features, targets, palette, milestones=(1,)
        )
    wrong_features = dict(features)
    wrong_features["patch"] = _Rows(np.zeros((8, 192), dtype=np.float32))
    with pytest.raises(ValueError, match="patch features must have shape"):
        spatial_fit.fit_spatial_decoders(
            decoders, wrong_features, targets, palette, milestones=(1,), batch_size=4
        )
    wrong_targets = _Rows(targets.values.astype(np.float32))
    with pytest.raises(TypeError, match="targets must have dtype uint8"):
        spatial_fit.fit_spatial_decoders(
            decoders, features, wrong_targets, palette, milestones=(1,), batch_size=4
        )
    with pytest.raises(ValueError, match="exactly 3 colors"):
        spatial_fit.fit_spatial_decoders(
            decoders,
            features,
            targets,
            np.asarray(
                [[0, 0, 0], *palette.tolist()], dtype=np.uint8
            ),
            milestones=(1,),
            batch_size=4,
        )
    with pytest.raises(ValueError, match="capped at 8192"):
        spatial_fit.fit_spatial_decoders(
            decoders, features, targets, palette, milestones=(8193,), batch_size=4
        )


def test_spatial_fit_rejects_nonfinite_loss_and_unmatched_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets, features, palette = _sources()
    nan_decoders = _make_decoders(monkeypatch, _NaNLocalPatchDecoder)
    with pytest.raises(RuntimeError, match="nonfinite spatial decoder loss"):
        spatial_fit.fit_spatial_decoders(
            nan_decoders, features, targets, palette, milestones=(1,), batch_size=4
        )

    decoders = _make_decoders(monkeypatch)
    with torch.no_grad():
        decoders["patch"].projection.bias.add_(1.0)
    with pytest.raises(ValueError, match="share initialization"):
        spatial_fit.fit_spatial_decoders(
            decoders, features, targets, palette, milestones=(1,), batch_size=4
        )


def test_real_local_patch_decoder_completes_one_native_update() -> None:
    targets, features, palette = _sources(rows=1)
    decoders = spatial_fit.make_spatial_decoders(device="cpu", seed=904)
    result = spatial_fit.fit_spatial_decoders(
        decoders,
        features,
        targets,
        palette,
        milestones=(1,),
        batch_size=1,
    )

    assert len(result.metrics) == 1
    assert set(result.snapshots) == {1}
    assert all(
        tuple(decoder_output.shape) == (1, 3, 128, 128)
        for decoder_output in (
            spatial_fit.forward_logits(
                decoders[condition],
                torch.from_numpy(features[condition].values[:1]),
            )
            for condition in spatial_fit.CONDITIONS
        )
    )
