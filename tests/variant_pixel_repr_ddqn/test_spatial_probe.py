"""Exercise the actual three-arm artifact producer with a bounded CPU fit."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dodge_native_game.variants.pixel_repr_ddqn import spatial_probe as study


def test_study_real_fit_evaluation_and_artifact_contract(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "fixture", Path(__file__).with_name("test_input_probe.py")
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    bank, palette = fixture.fixture_module()._fake_palette_frame_bank()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}")
    bank.train.metadata = {"index_sha256": "f" * 64}
    payload = fixture._world_payloads(
        palette, study.file_hash(dataset / "manifest.json")
    )["palette"]
    payload["palette_selected_pixels_sha256"] = study._pixel_bytes_sha256(
        bank.train.pixels
    )
    checkpoint = tmp_path / "world.pt"
    torch.save(payload, checkpoint)
    world = nn.Linear(1, 1)
    initial = {k: v.clone() for k, v in world.state_dict().items()}
    monkeypatch.setattr(study, "load_model", lambda p: (world, payload))
    monkeypatch.setattr(study.probe, "_ensure_bank", lambda *a, **k: None)
    monkeypatch.setattr(study.probe, "open_frame_bank", lambda *a, **k: bank)

    # The fake fixture has four-dimensional latents; expand only its features.
    def full_cls(source):
        return np.tile(np.asarray(source[:]), (1, 48)).astype("float32")

    train_cls, val_cls = full_cls(bank.train.cls), full_cls(bank.validation.cls)
    bank.train.cls = train_cls
    bank.validation.cls = val_cls
    bank.features["cls"] = study.probe._Concat([train_cls, val_cls])
    monkeypatch.setattr(
        study,
        "extract_patches",
        lambda m, part, *a, **k: np.repeat(part.cls[:, None], 256, axis=1),
    )
    result = study.run_study(
        dataset,
        checkpoint,
        tmp_path / "history",
        tmp_path / "banks",
        "test-spatial",
        device="cpu",
        milestones=(1,),
    )
    assert result["arms"] == ["cls", "patch", "pixels"]
    assert result["world_model_updates"] == 0
    states = []
    for arm in study.ARMS:
        run = tmp_path / "history" / f"test-spatial-{arm}"
        report = json.loads((run / "report.json").read_text())
        assert report["current_frame_only"]
        assert report["representation"] == arm
        assert report["direct_pixel_control"] == (arm == "pixels")
        assert set(report["splits"]) == {"train", "validation"}
        saved = torch.load(run / "decoder.pt", weights_only=True)
        assert saved["step"] == 1 and saved["world_model_sha256"] == study.file_hash(
            checkpoint
        )
        states.append(saved["sampler"])
        assert list((run / "images/step-1").glob("*-reconstructed.png"))
    assert all(torch.equal(states[0], s) for s in states)
    assert all(torch.equal(world.state_dict()[k], v) for k, v in initial.items())
    assert all(p.grad is None and not p.requires_grad for p in world.parameters())
    # Consume the real producer artifacts with the launcher's validation schema.
    import sys

    scripts = Path(__file__).resolve().parents[2] / "variants/pixel-repr-ddqn/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    launcher_spec = importlib.util.spec_from_file_location(
        "spatial_launcher", scripts / "colab_spatial_study.py"
    )
    launcher = importlib.util.module_from_spec(launcher_spec)
    sys.modules[launcher_spec.name] = launcher
    launcher_spec.loader.exec_module(launcher)
    monkeypatch.setattr(launcher, "MILESTONES", (1,))
    monkeypatch.setattr(
        launcher, "TRAIN_FRAMES", report["splits"]["train"]["frame_count"]
    )
    monkeypatch.setattr(
        launcher, "VALIDATION_FRAMES", report["splits"]["validation"]["frame_count"]
    )
    protocol = launcher._protocol(
        checkpoint_sha256=study.file_hash(checkpoint),
        data_sha256=bank.data_hash,
        run_id="test-spatial",
        source_commit="fixture",
    )
    for arm in study.ARMS:
        launcher._validate_run_artifacts(
            tmp_path / "history" / f"test-spatial-{arm}", arm=arm, protocol=protocol
        )
