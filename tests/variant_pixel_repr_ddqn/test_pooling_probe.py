from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from dodge_native_game.variants.pixel_repr_ddqn import pooling_probe
from dodge_native_game.variants.pixel_repr_ddqn.pooling_readout import (
    CONDITIONS,
    make_pooling_decoders,
)
from dodge_native_game.variants.pixel_repr_ddqn.probe_bank import _state_digest


def _fixture_module():
    spec = importlib.util.spec_from_file_location(
        "pooling_probe_fixture", Path(__file__).with_name("test_input_probe.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pooling_worker_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "variants/pixel-repr-ddqn/scripts/colab_pooling_worker.py"
    )
    spec = importlib.util.spec_from_file_location("pooling_worker_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pooling_study_rejects_unknown_eval_scope(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="eval_scope"):
        pooling_probe.run_study(
            tmp_path / "dataset",
            tmp_path / "world.pt",
            tmp_path / "history",
            tmp_path / "spatial-banks",
            "pooling-test",
            device="cpu",
            milestones=(1,),
            eval_scope="train-only",
        )


def _eval_scope_fixture(tmp_path: Path, monkeypatch):
    fixture = _fixture_module()
    bank, palette = fixture.fixture_module()._fake_palette_frame_bank()
    bank.train.metadata = {"index_sha256": "f" * 64}
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}")
    payload = _fixture_module()._world_payloads(palette, bank.data_hash)["palette"]
    payload["palette_selected_pixels_sha256"] = pooling_probe._pixel_bytes_sha256(
        bank.train.pixels
    )
    checkpoint = tmp_path / "world.pt"
    torch.save(payload, checkpoint)
    rng = np.random.default_rng(17)
    sidecars = {
        "train": rng.normal(size=(len(bank.train.pixels), 256, 192)).astype(np.float32),
        "validation": rng.normal(
            size=(len(bank.validation.pixels), 256, 192)
        ).astype(np.float32),
    }
    real_file_hash = pooling_probe.file_hash

    def file_hash(path: Path) -> str:
        path = Path(path)
        if path == dataset / "manifest.json":
            return bank.data_hash
        if path == checkpoint:
            return "w" * 64
        if path.name == "metadata.json":
            return "m" * 64
        if path.name == "patches.npy":
            return "p" * 64
        return real_file_hash(path)

    monkeypatch.setattr(pooling_probe, "file_hash", file_hash)
    monkeypatch.setattr(
        pooling_probe.probe, "open_frame_bank", lambda *args, **kwargs: bank
    )
    monkeypatch.setattr(
        pooling_probe, "open_patches", lambda _bank, root: sidecars[Path(root).name]
    )
    return bank, dataset, checkpoint


def test_pooling_study_validation_eval_scope_stays_scoped(
    tmp_path: Path, monkeypatch
) -> None:
    bank, dataset, checkpoint = _eval_scope_fixture(tmp_path, monkeypatch)
    seen: list[tuple[int, int]] = []
    original = pooling_probe.probe.evaluate_decoder_stream

    def spy(decoder, features, pixels, changed, records, ranges, *args, **kwargs):
        seen.append((len(features), len(records)))
        return original(
            decoder, features, pixels, changed, records, ranges, *args, **kwargs
        )

    monkeypatch.setattr(pooling_probe.probe, "evaluate_decoder_stream", spy)
    result = pooling_probe.run_study(
        dataset,
        checkpoint,
        tmp_path / "history",
        tmp_path / "spatial-banks",
        "pooling-scope",
        device="cpu",
        milestones=(1,),
        eval_scope="validation",
    )
    assert result["eval_scope"] == "validation"
    assert seen
    validation_rows = len(bank.validation.pixels)
    assert all(pair == (validation_rows, validation_rows) for pair in seen)
    report = json.loads(
        (tmp_path / "history/pooling-scope-mean/report.json").read_text()
    )
    assert report["eval_scope"] == "validation"
    assert report["step"] == 1


def test_pooling_study_reuses_frozen_banks_and_worker_contract(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _fixture_module()
    bank, palette = fixture.fixture_module()._fake_palette_frame_bank()
    bank.train.metadata = {"index_sha256": "f" * 64}

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}")
    payload = _fixture_module()._world_payloads(
        palette, bank.data_hash
    )["palette"]
    payload["palette_selected_pixels_sha256"] = pooling_probe._pixel_bytes_sha256(
        bank.train.pixels
    )
    checkpoint = tmp_path / "world.pt"
    torch.save(payload, checkpoint)

    rng = np.random.default_rng(17)
    sidecars = {
        "train": rng.normal(size=(len(bank.train.pixels), 256, 192)).astype(
            np.float32
        ),
        "validation": rng.normal(
            size=(len(bank.validation.pixels), 256, 192)
        ).astype(np.float32),
    }
    real_file_hash = pooling_probe.file_hash

    def file_hash(path: Path) -> str:
        path = Path(path)
        if path == dataset / "manifest.json":
            return bank.data_hash
        if path == checkpoint:
            return "w" * 64
        if path.name == "metadata.json":
            return "m" * 64
        if path.name == "patches.npy":
            return "p" * 64
        return real_file_hash(path)

    monkeypatch.setattr(pooling_probe, "file_hash", file_hash)
    monkeypatch.setattr(
        pooling_probe.probe,
        "open_frame_bank",
        lambda *args, **kwargs: bank,
    )
    extraction_called = False

    def fail_extraction(*args, **kwargs):
        nonlocal extraction_called
        extraction_called = True
        raise AssertionError("pooling study must not extract or rebuild features")

    monkeypatch.setattr(
        pooling_probe, "extract_patches", fail_extraction, raising=False
    )
    monkeypatch.setattr(
        pooling_probe,
        "open_patches",
        lambda _bank, root: sidecars[Path(root).name],
    )

    initial = make_pooling_decoders(device="cpu", seed=904)
    core_hash = _state_digest(initial["mean"].core)
    result = pooling_probe.run_study(
        dataset,
        checkpoint,
        tmp_path / "history",
        tmp_path / "spatial-banks",
        "pooling-test",
        device="cpu",
        milestones=(1,),
    )

    assert not extraction_called
    assert result["arms"] == list(CONDITIONS)
    assert result["milestones"] == [1]
    assert result["world_model_updates"] == 0
    assert result["core_initial_state_sha256"] == core_hash
    assert result["initial_state_sha256"] == core_hash
    assert result["core_parameter_count"] == 165056
    assert result["attention_extra_parameter_count"] == 192
    assert result["spatial_bank_read_only"] is True

    history = tmp_path / "history"
    for mode in CONDITIONS:
        run = history / f"pooling-test-{mode}"
        report = json.loads((run / "report.json").read_text())
        assert report["pooling_mode"] == mode
        assert report["step"] == 1
        saved = torch.load(run / "decoder.pt", weights_only=True)
        assert saved["step"] == 1
        assert saved["world_model_sha256"] == "w" * 64

    worker = _pooling_worker_module()
    protocol = {
        "inputs": {
            "world.pt": "w" * 64,
            "dataset/manifest.json": bank.data_hash,
        },
        "frame_index_sha256": bank.frame_index_hash,
        "palette_sha256": result["palette_sha256"],
        "core_initial_state_sha256": core_hash,
        "milestones": [1],
        "arms": list(CONDITIONS),
    }
    worker.validate_result(result, protocol)

