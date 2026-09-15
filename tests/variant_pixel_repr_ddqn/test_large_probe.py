from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from dodge_native_game.variants.pixel_repr_ddqn import large_probe
from dodge_native_game.variants.pixel_repr_ddqn.large_probe import (
    FrameBank,
    balanced_bright_loss,
    evaluate_decoder_stream,
    fit_matched_decoders,
    make_decoder_pair,
    stream_train_mean,
)


class _TinyDecoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(latent_dim, 3 * 4 * 4)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.projection(latent)).reshape(-1, 3, 4, 4)


class _NativeTinyDecoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.latent_dim = latent_dim

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        value = torch.sigmoid(latent[:, :1] + self.bias)
        return value[:, :, None, None].expand(-1, 3, 128, 128)


class _NativePaletteTinyDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        output_channels: int = 3,
        raw_logits: bool = True,
    ) -> None:
        super().__init__()
        self.output_channels = output_channels
        self.raw_logits = raw_logits
        self.projection = nn.Linear(latent_dim, output_channels)

    def forward_logits(self, latent: torch.Tensor) -> torch.Tensor:
        values = self.projection(latent)
        return values[:, :, None, None].expand(-1, -1, 128, 128)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.forward_logits(latent)


class _IdentityDecoder(nn.Module):
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.reshape(-1, 3, 4, 4)


class _Rows:
    """Array-like rows that make the source passed to fitting observable."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.accesses: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, key: object) -> np.ndarray:
        self.accesses.append(np.asarray(key).copy())
        return self.values[key]  # type: ignore[index]


def _records(count: int, split: str, offset: int = 0) -> list[dict[str, object]]:
    return [
        {
            "split": split,
            "index": offset + index,
            "episode_id": f"{split}-episode-{index}",
            "frame_index": index % 4 + 1,
            "recipe_family": f"family-{index}",
        }
        for index in range(count)
    ]


def test_matched_fit_uses_identical_initialization_and_sampling() -> None:
    torch.manual_seed(71)
    cls_decoder, projected_decoder = make_decoder_pair(4, decoder_factory=_TinyDecoder)
    for left, right in zip(
        cls_decoder.state_dict().values(),
        projected_decoder.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)

    cls_values = _Rows(np.random.default_rng(1).normal(size=(8, 4)).astype(np.float32))
    projected_values = _Rows(
        np.random.default_rng(2).normal(size=(8, 4)).astype(np.float32)
    )
    targets = np.random.default_rng(3).integers(
        0, 256, size=(8, 3, 4, 4), dtype=np.uint8
    )
    milestones: list[int] = []
    result = fit_matched_decoders(
        cls_decoder,
        projected_decoder,
        cls_values,
        projected_values,
        targets,
        milestones=(1, 3),
        batch_size=4,
        sampler=torch.Generator().manual_seed(903),
        on_milestone=lambda snapshot, *_: milestones.append(snapshot.step),
    )

    assert milestones == [1, 3]
    assert set(result.snapshots) == {1, 3}
    assert len(result.metrics) == 3
    assert all(
        np.array_equal(left, right)
        for left, right in zip(
            cls_values.accesses, projected_values.accesses, strict=True
        )
    )
    expected = torch.Generator().manual_seed(903)
    expected_indices = [
        tuple(torch.randint(8, (4,), generator=expected).tolist()) for _ in range(3)
    ]
    assert result.sampled_indices == tuple(expected_indices)
    assert result.snapshots[3].optimizer["cls"]["state"]


def test_balanced_bright_is_per_frame_and_weights_gradients_by_class() -> None:
    target = torch.zeros(2, 3, 1, 4)
    target[0, :, 0, 0] = 1.0
    target[1, :, 0, :3] = 1.0
    prediction = target.clone()
    prediction[0, :, 0, 0] = 0.0

    loss = balanced_bright_loss(prediction, target)

    # The first frame misses one bright pixel and the second is exact. Per-frame
    # balancing gives 0.25; pooling its four bright pixels would give 0.125.
    assert loss.item() == pytest.approx(0.25)

    gradient_target = torch.zeros(2, 3, 1, 2)
    gradient_target[:, :, 0, 0] = 1.0
    gradient_prediction = gradient_target.clone()
    gradient_prediction[0, :, 0, 0] = 0.0
    gradient_prediction[1, :, 0, 1] = 1.0
    gradient_prediction.requires_grad_()
    balanced_bright_loss(gradient_prediction, gradient_target).backward()
    pixel_gradient = gradient_prediction.grad.abs().sum(dim=1)
    bright_gradient = pixel_gradient[gradient_target.amin(dim=1) >= 0.8].sum()
    background_gradient = pixel_gradient[gradient_target.amin(dim=1) < 0.8].sum()
    assert bright_gradient == pytest.approx(background_gradient)


@pytest.mark.parametrize("bright", [True, False])
def test_balanced_bright_empty_class_is_finite_and_differentiable(
    bright: bool,
) -> None:
    target = torch.ones(2, 3, 2, 2) if bright else torch.zeros(2, 3, 2, 2)
    prediction = torch.zeros_like(target, requires_grad=True)

    loss = balanced_bright_loss(prediction, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_balanced_bright_uses_target_mask_for_false_white_background() -> None:
    target = torch.tensor([[[[0.0]], [[0.0]], [[1.0]]]])
    prediction = torch.ones_like(target, requires_grad=True)

    loss = balanced_bright_loss(prediction, target)
    loss.backward()

    assert loss.item() == pytest.approx(2.0 / 3.0)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0


def test_explicit_mse_matches_default_fit_exactly() -> None:
    features = torch.randn(8, 4, generator=torch.Generator().manual_seed(11))
    targets = torch.randint(
        0,
        256,
        (8, 3, 4, 4),
        generator=torch.Generator().manual_seed(12),
        dtype=torch.uint8,
    )
    default_cls, default_projected = make_decoder_pair(4, decoder_factory=_TinyDecoder)
    explicit_cls, explicit_projected = make_decoder_pair(
        4, decoder_factory=_TinyDecoder
    )
    default = fit_matched_decoders(
        default_cls,
        default_projected,
        features,
        features,
        targets,
        milestones=(2,),
        batch_size=4,
        sampler=torch.Generator().manual_seed(903),
    )
    explicit = fit_matched_decoders(
        explicit_cls,
        explicit_projected,
        features,
        features,
        targets,
        milestones=(2,),
        batch_size=4,
        loss_kind="mse",
        sampler=torch.Generator().manual_seed(903),
    )
    for left, right in zip(
        default_cls.state_dict().values(),
        explicit_cls.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(
        default_projected.state_dict().values(),
        explicit_projected.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for default_metric, explicit_metric in zip(
        default.metrics, explicit.metrics, strict=True
    ):
        assert default_metric["cls_loss"] == explicit_metric["cls_loss"]
        assert default_metric["projected_loss"] == explicit_metric["projected_loss"]


def test_train_mean_streams_normalized_rows_and_evaluation_uses_bank_masks() -> None:
    pixels = np.zeros((16, 3, 4, 4), dtype=np.uint8)
    pixels[:8] = np.arange(8, dtype=np.uint8)[:, None, None, None]
    mean = stream_train_mean(pixels, range(0, 8), batch_size=3)
    np.testing.assert_allclose(mean, (3.5 / 255.0), rtol=0, atol=1e-7)

    features = pixels.astype(np.float32).reshape(16, -1) / 255.0
    changed = np.zeros((16, 4, 4), dtype=bool)
    changed[8, 1, 2] = True
    records = _records(8, "train") + _records(8, "validation", offset=8)
    identity = _IdentityDecoder()
    result = evaluate_decoder_stream(
        identity,
        features,
        pixels,
        changed,
        records,
        {"train": (0, 8), "validation": (8, 16)},
        np.zeros((3, 4, 4), dtype=np.float32),
        device="cpu",
        batch_size=5,
    )
    assert result["splits"]["train"]["mse"] == pytest.approx(0.0)
    assert result["splits"]["train"]["changed_region_mse"] is None
    assert result["splits"]["validation"]["changed_pixel_count"] == 1
    assert result["splits"]["validation"]["changed_region_mse"] == pytest.approx(0.0)
    assert result["splits"]["train"]["bright_mse"] is None
    assert result["splits"]["train"]["target_bright_share"] == 0.0
    assert result["splits"]["validation"]["changed_bright_count"] == 0

    wrong = evaluate_decoder_stream(
        identity,
        features,
        pixels,
        changed,
        records,
        {"train": (0, 8), "validation": (8, 16)},
        np.zeros((3, 4, 4), dtype=np.float32),
        device="cpu",
        batch_size=5,
        wrong_permutation=True,
    )
    first = next(item for item in wrong["examples"] if item["index"] == 0)
    np.testing.assert_allclose(first["reconstructed"], pixels[4] / 255.0)


def test_evaluation_reports_bright_and_changed_bright_metrics() -> None:
    pixels = np.zeros((4, 3, 4, 4), dtype=np.uint8)
    pixels[:, 2] = 255  # blue background is target-background, not bright
    pixels[:, :, 0, 0] = 255  # one white target pixel per frame
    changed = np.zeros((4, 4, 4), dtype=bool)
    changed[:, 0, 0] = True
    features = np.ones((4, 3 * 4 * 4), dtype=np.float32)
    records = _records(2, "train") + _records(2, "validation", offset=2)

    result = evaluate_decoder_stream(
        _IdentityDecoder(),
        features,
        pixels,
        changed,
        records,
        {"train": (0, 2), "validation": (2, 4)},
        np.zeros((3, 4, 4), dtype=np.float32),
        device="cpu",
        batch_size=3,
    )
    train = result["splits"]["train"]
    assert train["target_bright_share"] == pytest.approx(1 / 16)
    assert train["predicted_bright_share"] == 1.0
    assert train["bright_precision"] == pytest.approx(1 / 16)
    assert train["bright_recall"] == pytest.approx(1.0)
    assert train["bright_iou"] == pytest.approx(1 / 16)
    assert train["changed_bright_count"] == 2
    assert train["changed_bright_mse"] == pytest.approx(0.0)
    assert train["background_mse"] == pytest.approx(2 / 3)


def _fake_frame_bank(train_rows: int = 32, validation_rows: int = 16) -> FrameBank:
    rng = np.random.default_rng(22)
    pixels_train = rng.integers(0, 256, size=(train_rows, 3, 128, 128), dtype=np.uint8)
    pixels_validation = rng.integers(
        0, 256, size=(validation_rows, 3, 128, 128), dtype=np.uint8
    )
    changed_train = np.zeros((train_rows, 128, 128), dtype=bool)
    changed_validation = np.zeros((validation_rows, 128, 128), dtype=bool)
    changed_validation[:, 0, 0] = True
    cls_train = rng.normal(size=(train_rows, 4)).astype(np.float32)
    cls_validation = rng.normal(size=(validation_rows, 4)).astype(np.float32)
    projected_train = rng.normal(size=(train_rows, 4)).astype(np.float32)
    projected_validation = rng.normal(size=(validation_rows, 4)).astype(np.float32)
    train_cls = _Rows(cls_train)
    train_projected = _Rows(projected_train)
    train_pixels = _Rows(pixels_train)
    train = SimpleNamespace(
        cls=train_cls,
        projected=train_projected,
        pixels=train_pixels,
    )
    validation = SimpleNamespace(
        cls=cls_validation,
        projected=projected_validation,
        pixels=pixels_validation,
    )
    records = _records(train_rows, "train") + _records(
        validation_rows, "validation", offset=train_rows
    )
    return FrameBank(
        root=Path("/fake-bank"),
        train=train,  # type: ignore[arg-type]
        validation=validation,  # type: ignore[arg-type]
        pixels=large_probe._Concat((pixels_train, pixels_validation)),
        changed=large_probe._Concat((changed_train, changed_validation)),
        features={
            "cls": large_probe._Concat((cls_train, cls_validation)),
            "projected": large_probe._Concat((projected_train, projected_validation)),
        },
        records=tuple(records),
        split_ranges={
            "train": (0, train_rows),
            "validation": (train_rows, train_rows + validation_rows),
        },
        data_hash="d" * 64,
        frame_index_hash="f" * 64,
    )


def _fake_palette_frame_bank(
    train_rows: int = 32, validation_rows: int = 16
) -> tuple[FrameBank, np.ndarray]:
    rng = np.random.default_rng(29)
    palette = np.asarray(
        [[29, 43, 83], [41, 173, 255], [255, 241, 232]], dtype=np.uint8
    )
    train_ids = rng.integers(0, len(palette), size=(train_rows, 128, 128))
    validation_ids = rng.integers(0, len(palette), size=(validation_rows, 128, 128))
    pixels_train = palette[train_ids].transpose(0, 3, 1, 2).copy()
    pixels_validation = palette[validation_ids].transpose(0, 3, 1, 2).copy()
    changed_train = np.zeros((train_rows, 128, 128), dtype=bool)
    changed_validation = np.zeros((validation_rows, 128, 128), dtype=bool)
    changed_validation[:, 0, 0] = True
    cls_train = rng.normal(size=(train_rows, 4)).astype(np.float32)
    cls_validation = rng.normal(size=(validation_rows, 4)).astype(np.float32)
    projected_train = rng.normal(size=(train_rows, 4)).astype(np.float32)
    projected_validation = rng.normal(size=(validation_rows, 4)).astype(np.float32)
    train_cls = _Rows(cls_train)
    train_projected = _Rows(projected_train)
    train_pixels = _Rows(pixels_train)
    train = SimpleNamespace(
        cls=train_cls,
        projected=train_projected,
        pixels=train_pixels,
    )
    validation = SimpleNamespace(
        cls=cls_validation,
        projected=projected_validation,
        pixels=pixels_validation,
    )
    records = _records(train_rows, "train") + _records(
        validation_rows, "validation", offset=train_rows
    )
    bank = FrameBank(
        root=Path("/fake-palette-bank"),
        train=train,  # type: ignore[arg-type]
        validation=validation,  # type: ignore[arg-type]
        pixels=large_probe._Concat((pixels_train, pixels_validation)),
        changed=large_probe._Concat((changed_train, changed_validation)),
        features={
            "cls": large_probe._Concat((cls_train, cls_validation)),
            "projected": large_probe._Concat((projected_train, projected_validation)),
        },
        records=tuple(records),
        split_ranges={
            "train": (0, train_rows),
            "validation": (train_rows, train_rows + validation_rows),
        },
        data_hash="d" * 64,
        frame_index_hash="f" * 64,
    )
    return bank, palette


@pytest.mark.parametrize(
    ("loss_kind", "expected_experiment", "equal_class_weights"),
    [
        ("mse", large_probe._EXPERIMENT, False),
        (
            "balanced-bright",
            large_probe._BALANCED_BRIGHT_EXPERIMENT,
            True,
        ),
    ],
)
def test_run_study_fits_train_rows_and_publishes_each_milestone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loss_kind: str,
    expected_experiment: str,
    equal_class_weights: bool,
) -> None:
    bank = _fake_frame_bank()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen-world-model")
    frozen = nn.Linear(1, 1)
    before = {key: value.detach().clone() for key, value in frozen.state_dict().items()}

    monkeypatch.setattr(
        large_probe.pretrain,
        "load_model",
        lambda path: (
            frozen,
            {
                "inference_only": True,
                "experiment": "practice-batch32-v1",
                "step": 512,
                "batch_size": 32,
                "profile": "reference",
                "calibration": {},
                "data_hash": "dataset",
            },
        ),
    )
    monkeypatch.setattr(large_probe, "_validate_protocol", lambda *_: "NVIDIA T4")
    monkeypatch.setattr(large_probe, "_ensure_bank", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(large_probe, "open_frame_bank", lambda *_args, **_kwargs: bank)
    monkeypatch.setattr(
        large_probe,
        "make_decoder_pair",
        lambda *_args, **_kwargs: (
            _NativeTinyDecoder(4),
            _NativeTinyDecoder(4),
        ),
    )
    save_calls: list[str] = []
    save_checkpoint = large_probe.pretrain.save_checkpoint

    def observe_checkpoint(path: Path, payload: dict[str, object]) -> None:
        if path.name == "decoder-2.pt":
            save_calls.append(path.parent.name)
            previous = json.loads((path.parent / "visualizations.json").read_text())
            assert previous[0]["step"] == 1
        save_checkpoint(path, payload)

    monkeypatch.setattr(large_probe.pretrain, "save_checkpoint", observe_checkpoint)

    runs = large_probe.run_study(
        checkpoint,
        tmp_path / "dataset",
        tmp_path / "history",
        "tiny-large-probe",
        milestones=(1, 2),
        device="cpu",
        bank_root=tmp_path / "bank",
        loss_kind=loss_kind,
    )

    assert {path.name for path in runs} == {
        "tiny-large-probe-cls",
        "tiny-large-probe-projected",
    }
    assert bank.train.cls.accesses
    assert bank.train.projected.accesses
    assert bank.train.pixels.accesses
    assert all(int(indices.max()) < 32 for indices in bank.train.cls.accesses)
    assert all(int(indices.max()) < 32 for indices in bank.train.projected.accesses)
    assert all(int(indices.max()) < 32 for indices in bank.train.pixels.accesses)
    assert sorted(save_calls) == [
        "tiny-large-probe-cls",
        "tiny-large-probe-projected",
    ]
    for run in runs:
        assert (run / "decoder-1.pt").is_file()
        assert (run / "decoder-2.pt").is_file()
        manifest = json.loads((run / "manifest.json").read_text())
        config = json.loads((run / "config.json").read_text())
        decoder_checkpoint = torch.load(
            run / "decoder-2.pt", map_location="cpu", weights_only=True
        )
        for metadata in (manifest, config, decoder_checkpoint):
            assert metadata["bright_threshold"] == pytest.approx(0.8)
            assert metadata["loss_kind"] == loss_kind
            assert metadata["equal_class_weights"] is equal_class_weights
            assert metadata["loss_normalization"] == (
                "per-frame-then-batch"
                if loss_kind == "balanced-bright"
                else "global-pixel-mean"
            )
            assert metadata["experiment"] == expected_experiment
        assert manifest["experiment"] == expected_experiment
        visualizations = json.loads((run / "visualizations.json").read_text())
        assert len(visualizations) == 16
        assert all(view["metadata"]["current_frame_only"] for view in visualizations)
        rows = [
            json.loads(line)
            for line in (run / "metrics.jsonl").read_text().splitlines()
        ]
        assert {row["step"] for row in rows if row["phase"].endswith("evaluation")} == {
            1,
            2,
        }
    for key, value in frozen.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("loss_kind", "expected_experiment", "expected_normalization"),
    [
        (
            "palette-ce",
            large_probe._PALETTE_CE_EXPERIMENT,
            "unweighted-pixel-mean",
        ),
        (
            "palette-bce",
            large_probe._PALETTE_BCE_EXPERIMENT,
            "unweighted-pixel-class-mean",
        ),
    ],
)
def test_palette_run_study_publishes_train_only_palette_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loss_kind: str,
    expected_experiment: str,
    expected_normalization: str,
) -> None:
    bank, palette = _fake_palette_frame_bank()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen-world-model")
    frozen = nn.Linear(1, 1)
    before = {key: value.detach().clone() for key, value in frozen.state_dict().items()}
    derive_sources: list[object] = []
    validate_sources: list[object] = []
    real_derive = large_probe.derive_palette
    real_validate = large_probe.validate_palette_coverage

    def observe_derive(pixels: object, indices: object, **kwargs: object) -> np.ndarray:
        derive_sources.append(pixels)
        return real_derive(pixels, indices, **kwargs)

    def observe_validate(
        pixels: object, indices: object, palette_value: np.ndarray, **kwargs: object
    ) -> None:
        validate_sources.append(pixels)
        real_validate(pixels, indices, palette_value, **kwargs)

    monkeypatch.setattr(
        large_probe.pretrain,
        "load_model",
        lambda path: (
            frozen,
            {
                "inference_only": True,
                "experiment": "practice-batch32-v1",
                "step": 512,
                "batch_size": 32,
                "profile": "reference",
                "calibration": {},
                "data_hash": "dataset",
            },
        ),
    )
    monkeypatch.setattr(large_probe, "_validate_protocol", lambda *_: "NVIDIA T4")
    monkeypatch.setattr(large_probe, "_ensure_bank", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(large_probe, "open_frame_bank", lambda *_args, **_kwargs: bank)
    monkeypatch.setattr(large_probe, "derive_palette", observe_derive)
    monkeypatch.setattr(large_probe, "validate_palette_coverage", observe_validate)
    real_make_pair = large_probe.make_decoder_pair

    def make_palette_pair(*args: object, **kwargs: object):
        return real_make_pair(
            4,
            decoder_factory=_NativePaletteTinyDecoder,
            **kwargs,
        )

    monkeypatch.setattr(large_probe, "make_decoder_pair", make_palette_pair)

    runs = large_probe.run_study(
        checkpoint,
        tmp_path / "dataset",
        tmp_path / "history",
        f"tiny-{loss_kind}",
        milestones=(1, 2),
        device="cpu",
        bank_root=tmp_path / "bank",
        loss_kind=loss_kind,
    )

    expected_palette = palette.tolist()
    expected_hash = hashlib.sha256(palette.tobytes()).hexdigest()
    expected_metadata = {
        "loss_kind": loss_kind,
        "loss_normalization": expected_normalization,
        "palette_rgb": expected_palette,
        "palette_sha256": expected_hash,
        "palette_source_split": "train",
        "palette_size": len(palette),
        "output_channels": len(palette),
        "raw_logits": True,
        "decoder_input_split": "train",
        "train_only_input": True,
    }
    expected_experiment_metadata = {"experiment": expected_experiment}
    assert derive_sources == [bank.train.pixels]
    assert validate_sources == [bank.validation.pixels]
    assert {path.name for path in runs} == {
        f"tiny-{loss_kind}-cls",
        f"tiny-{loss_kind}-projected",
    }
    comparison = json.loads(
        (
            tmp_path / "history" / f"tiny-{loss_kind}-comparison.json"
        ).read_text()
    )
    for key, value in expected_metadata.items():
        assert comparison[key] == value
    for key, value in expected_experiment_metadata.items():
        assert comparison[key] == value
    for run in runs:
        manifest = json.loads((run / "manifest.json").read_text())
        config = json.loads((run / "config.json").read_text())
        report = json.loads((run / "report.json").read_text())
        decoder_checkpoint = torch.load(
            run / "decoder-2.pt", map_location="cpu", weights_only=True
        )
        evaluation = json.loads((run / "evaluation-2.json").read_text())
        for metadata in (
            manifest,
            config,
            report,
            decoder_checkpoint,
            evaluation,
        ):
            for key, value in expected_metadata.items():
                assert metadata[key] == value
        for metadata in (manifest, config, decoder_checkpoint):
            for key, value in expected_experiment_metadata.items():
                assert metadata[key] == value
        assert (run / "decoder-1.pt").is_file()
        assert (run / "decoder-2.pt").is_file()
        assert (run / "visualizations-1.json").is_file()
        assert (run / "visualizations-2.json").is_file()
        assert len(json.loads((run / "visualizations.json").read_text())) == 16
        assert all(
            "palette_rgb" in view["metadata"]
            for view in json.loads((run / "visualizations.json").read_text())
        )
        rows = [
            json.loads(line)
            for line in (run / "metrics.jsonl").read_text().splitlines()
        ]
        assert {row["step"] for row in rows if row["phase"].endswith("evaluation")} == {
            1,
            2,
        }
        assert evaluation["splits"]["validation"]["palette_class_count"] == 3
    for key, value in frozen.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
