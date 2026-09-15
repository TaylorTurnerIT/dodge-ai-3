from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from dodge_native_game.variants.pixel_repr_ddqn import input_pretrain
from dodge_native_game.variants.pixel_repr_ddqn import input_probe as study
from dodge_native_game.variants.pixel_repr_ddqn.model import (
    INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC,
    INPUT_ENCODING_RGB_NEAREST_SYMMETRIC,
    LeWMConfig,
)


def fixture_module():
    spec = importlib.util.spec_from_file_location(
        "probe_fixture", Path(__file__).with_name("test_large_probe.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _world_payloads(palette, data_hash: str) -> dict[str, dict[str, object]]:
    palette_rgb = palette.tolist()
    palette_sha256 = hashlib.sha256(palette.tobytes()).hexdigest()
    config_base = json.loads(json.dumps(asdict(LeWMConfig.reference())))
    nested_palette = {
        "palette_rgb": palette_rgb,
        "palette_sha256": palette_sha256,
        "palette_source_split": "train",
        "palette_frame_index_sha256": "f" * 64,
        "palette_selected_pixels_sha256": "p" * 64,
        "palette_frame_count": 32,
        "palette_frames_per_episode": 4,
        "palette_sampling_seed": 903,
    }
    shared = {
        "experiment": study.EXPERIMENT,
        "step": 1024,
        "batch_size": 32,
        "seed": 42,
        "initialization_seed": 42,
        "sampling_seed": 43,
        "stochastic_seed": 44,
        "world_model_updates": 1024,
        "sample_trace_steps": 1024,
        "initial_state_sha256": "i" * 64,
        "parameter_count": 123,
        "data_hash": data_hash,
        "palette_source_split": "train",
        "palette_sha256": palette_sha256,
        "palette_frame_index_sha256": "f" * 64,
        "palette_selected_pixels_sha256": "p" * 64,
        "sample_trace_sha256": "s" * 64,
        "stochastic_trace_sha256": "t" * 64,
        "palette_rgb": palette_rgb,
        "palette": nested_palette,
    }
    result = {}
    for arm, encoding in (
        ("rgb", INPUT_ENCODING_RGB_NEAREST_SYMMETRIC),
        ("palette", INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC),
    ):
        config = dict(config_base)
        config.update(
            input_encoding=encoding,
            palette_rgb=palette_rgb if arm == "palette" else None,
        )
        result[arm] = {
            **shared,
            "config": config,
            "input_arm": arm,
            "input_encoding": encoding,
        }
    return result


def test_bank_pair_rejects_target_or_selection_mismatch():
    from copy import deepcopy

    bank = SimpleNamespace(
        data_hash="data",
        frame_index_hash="frames",
        records=(),
        split_ranges={},
        train=SimpleNamespace(metadata={"files": {"pixels": "p", "changed": "c"}}),
        validation=SimpleNamespace(metadata={"files": {"pixels": "p", "changed": "c"}}),
    )
    other = deepcopy(bank)
    study.validate_banks({"rgb": bank, "palette": other})
    other.validation.metadata["files"]["pixels"] = "different"
    with pytest.raises(ValueError, match="pixels"):
        study.validate_banks({"rgb": bank, "palette": other})
    other = deepcopy(bank)
    other.frame_index_hash = "different"
    with pytest.raises(ValueError, match="indices"):
        study.validate_banks({"rgb": bank, "palette": other})


def test_paired_input_probes_freeze_world_and_use_train_cls(tmp_path, monkeypatch):
    fixture = fixture_module()
    bank, palette = fixture._fake_palette_frame_bank()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}")
    payloads = _world_payloads(palette, bank.data_hash)
    bank.train.metadata = {"index_sha256": "f" * 64}
    selected_hash = study._pixel_bytes_sha256(bank.train.pixels)
    for payload in payloads.values():
        payload["palette_selected_pixels_sha256"] = selected_hash
        payload["palette"]["palette_selected_pixels_sha256"] = selected_hash
    checkpoints = {}
    models = []
    for arm in study.ARMS:
        checkpoints[arm] = tmp_path / f"{arm}.pt"
        torch.save(payloads[arm], checkpoints[arm])

    def load(path):
        model = nn.Linear(1, 1)
        models.append((model, {k: v.clone() for k, v in model.state_dict().items()}))
        return model, payloads[path.stem]

    real_file_hash = study.file_hash

    def file_hash(path):
        if Path(path) == dataset / "manifest.json":
            return bank.data_hash
        return real_file_hash(path)

    monkeypatch.setattr(study, "load_model", load)
    monkeypatch.setattr(study, "file_hash", file_hash)
    monkeypatch.setattr(study.probe, "_ensure_bank", lambda *a, **k: None)
    monkeypatch.setattr(study.probe, "open_frame_bank", lambda *a, **k: bank)
    monkeypatch.setattr(study, "validate_banks", lambda banks: None)
    real_pair = study.probe.make_decoder_pair
    monkeypatch.setattr(
        study.probe,
        "make_decoder_pair",
        lambda *a, **k: real_pair(
            4,
            decoder_factory=fixture._NativePaletteTinyDecoder,
            **k,
        ),
    )
    result = study.run_probes(
        dataset,
        checkpoints,
        tmp_path / "history",
        tmp_path / "banks",
        "screen",
        device="cpu",
        milestones=(1, 2),
    )
    assert result["representation"] == "cls"
    assert bank.train.cls.accesses
    assert not bank.train.projected.accesses
    for model, initial in models:
        assert all(torch.equal(model.state_dict()[k], v) for k, v in initial.items())
        assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    states = []
    for arm in study.ARMS:
        run = tmp_path / "history" / f"screen-{arm}-cls"
        report = json.loads((run / "report.json").read_text())
        assert report["input_arm"] == arm and report["representation"] == "cls"
        assert report["palette_source_split"] == "train"
        checkpoint = torch.load(run / "decoder.pt", weights_only=True)
        assert checkpoint["decoder_kind"] == "cls" and checkpoint["step"] == 2
        rows = [
            json.loads(line)
            for line in (run / "metrics.jsonl").read_text().splitlines()
        ]
        assert rows and all(isinstance(row["loss"], float) for row in rows)
        assert report["current_frame_only"] is True
        states.append(checkpoint["sampler"])
    assert torch.equal(*states)


def test_validate_banks_accepts_actual_producer_metadata(tmp_path):
    from dodge_native_game.variants.pixel_repr_ddqn.probe_bank import (
        ProbeBank,
        build_bank,
    )

    spec = importlib.util.spec_from_file_location(
        "bank_fixture", Path(__file__).with_name("test_probe_bank.py")
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    dataset = tmp_path / "dataset"
    fixture._make_dataset(dataset, train_count=1, validation_count=1)
    banks = {}
    for arm in study.ARMS:
        parts = {}
        for split in ("train", "validation"):
            target = tmp_path / arm / split
            build_bank(fixture._FakeModel(), dataset, target, split, device="cpu")
            parts[split] = ProbeBank(target)
        banks[arm] = SimpleNamespace(
            data_hash=parts["train"].metadata["data_hash"],
            frame_index_hash=parts["train"].metadata["index_sha256"],
            records=(),
            split_ranges={},
            **parts,
        )
    study.validate_banks(banks)


def test_validate_world_pair_rejects_swapped_arm_and_shared_protocol_mismatch():
    palette = np.asarray([[29, 43, 83], [41, 173, 255], [255, 241, 232]], dtype="uint8")
    payloads = _world_payloads(palette, "d" * 64)
    study.validate_world_pair(payloads, "d" * 64)

    swapped = copy.deepcopy(payloads)
    swapped["rgb"]["input_arm"] = "palette"
    with pytest.raises(ValueError, match="input_arm"):
        study.validate_world_pair(swapped, "d" * 64)

    wrong_encoding = copy.deepcopy(payloads)
    wrong_encoding["rgb"]["config"]["input_encoding"] = "legacy"
    wrong_encoding["rgb"]["input_encoding"] = "legacy"
    with pytest.raises(ValueError, match="input_encoding"):
        study.validate_world_pair(wrong_encoding, "d" * 64)

    wrong_trace = copy.deepcopy(payloads)
    wrong_trace["palette"]["sample_trace_sha256"] = "x" * 64
    with pytest.raises(ValueError, match="sample_trace_sha256"):
        study.validate_world_pair(wrong_trace, "d" * 64)

    wrong_data = copy.deepcopy(payloads)
    with pytest.raises(ValueError, match="data_hash"):
        study.validate_world_pair(wrong_data, "e" * 64)


def test_validate_world_pair_accepts_actual_checkpoint_payload_producer():
    palette = np.asarray(
        [[29, 43, 83], [41, 173, 255], [255, 241, 232]], dtype=np.uint8
    )
    colors = tuple(tuple(int(channel) for channel in color) for color in palette)
    palette_hash = hashlib.sha256(palette.tobytes()).hexdigest()
    provenance = input_pretrain.PaletteProvenance(
        palette_rgb=colors,
        palette_sha256=palette_hash,
        dataset_manifest_sha256="d" * 64,
        frame_index_sha256="f" * 64,
        selected_pixels_sha256="p" * 64,
        frame_count=32,
        frames_per_episode=4,
        sampling_seed=903,
    )
    payloads = {}
    for arm, encoding in (
        ("rgb", INPUT_ENCODING_RGB_NEAREST_SYMMETRIC),
        ("palette", INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC),
    ):
        config = LeWMConfig.reference(
            input_encoding=encoding,
            palette_rgb=colors if arm == "palette" else None,
        )
        config_payload = input_pretrain._canonical_config(config)
        metadata = input_pretrain._arm_metadata(
            run_id=f"pair-{arm}",
            arm=arm,
            config=config,
            config_payload=config_payload,
            palette=provenance,
            data_hash="d" * 64,
            pair_run_id="pair",
            initial_state_sha256="i" * 64,
            parameter_count=123,
        )
        model = nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        payloads[arm] = input_pretrain._checkpoint_payload(
            model=model,
            optimizer=optimizer,
            config_payload=config_payload,
            metadata=metadata,
            step=1024,
            sampling_rng=torch.Generator().manual_seed(43),
            post_rng=input_pretrain._RNGState(torch.get_rng_state(), None),
            sample_trace_hash="s" * 64,
            stochastic_trace_hash="t" * 64,
        )
    study.validate_world_pair(payloads, "d" * 64)
