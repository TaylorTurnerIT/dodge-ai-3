"""Parent integration checks for artifact isolation and device requirements."""

from pathlib import Path

import pytest
import torch

from dodge_native_game.variants.pixel_repr_ddqn.pretrain import train
from dodge_native_game.variants.pixel_repr_ddqn.run_artifacts import create_run


def test_run_identity_cannot_escape_or_overwrite(tmp_path: Path):
    for invalid in ("../escape", "/absolute", "nested/run", ""):
        with pytest.raises(ValueError):
            create_run(tmp_path, invalid, {})
    run = create_run(tmp_path, "fresh", {"profile": "reference"})
    original = (run / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        create_run(tmp_path, "fresh", {"profile": "different"})
    assert (run / "manifest.json").read_bytes() == original


def test_cuda_requirement_never_falls_back_to_cpu(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Colab T4"):
        train(dataset_root=tmp_path / "missing", history_root=tmp_path)
    assert not (tmp_path / "lewm-mvp").exists()


class _FixtureDataset:
    def __init__(self, *_args, **_kwargs):
        pass

    def __len__(self):
        return 4

    def __getitem__(self, index):
        pixels = torch.arange(4 * 3 * 16 * 16).reshape(4, 3, 16, 16)
        return {
            "pixels": ((pixels + index * 17) % 256).to(torch.uint8),
            "actions": torch.tensor([index, index + 1, index + 2]),
        }


def test_checkpoint_resume_matches_uninterrupted_updates(monkeypatch, tmp_path):
    from dodge_native_game.variants.pixel_repr_ddqn import pretrain

    monkeypatch.setattr(pretrain, "PixelSequenceDataset", _FixtureDataset)
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text("{}")
    common = dict(
        dataset_root=data,
        history_root=tmp_path / "runs",
        profile="tiny",
        batch_size=2,
        device="cpu",
        seed=61,
    )
    full = train(**common, run_id="full", steps=2)
    first = train(**common, run_id="first", steps=1)
    resumed = train(**common, run_id="resumed", steps=1, resume=first / "checkpoint.pt")
    expected = torch.load(full / "checkpoint.pt", weights_only=True)
    actual = torch.load(resumed / "checkpoint.pt", weights_only=True)
    assert actual["step"] == expected["step"] == 2
    for key, value in expected["model"].items():
        torch.testing.assert_close(actual["model"][key], value, rtol=0, atol=0)


def test_probe_preserves_checkpoint_and_uses_validation(monkeypatch, tmp_path):
    import json

    from dodge_native_game.variants.pixel_repr_ddqn import pretrain, probe
    from dodge_native_game.variants.pixel_repr_ddqn.run_artifacts import file_hash

    splits = []

    class RecordingDataset(_FixtureDataset):
        def __init__(self, *_args, split, **_kwargs):
            splits.append(split)

    monkeypatch.setattr(pretrain, "PixelSequenceDataset", _FixtureDataset)
    monkeypatch.setattr(probe, "PixelSequenceDataset", RecordingDataset)
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text("{}")
    run = train(
        dataset_root=data,
        history_root=tmp_path / "runs",
        profile="tiny",
        steps=1,
        batch_size=2,
        device="cpu",
    )
    before = file_hash(run / "checkpoint.pt")
    probe.fit_probe(run, data, steps=1, batch_size=2, device="cpu")
    assert file_hash(run / "checkpoint.pt") == before
    assert splits == ["train", "validation"]
    visualization = json.loads((run / "visualization.json").read_text())
    assert visualization["metadata"]["checkpoint_sha256"] == before
    assert len(visualization["frames"]) == 4
    assert visualization["diagnostic_only"] is True


@pytest.mark.parametrize(
    "override",
    [
        {"steps": 513},
        {"steps": 32},
        {"device": "cpu"},
        {"profile": "tiny"},
        {"batch_size": 4},
        {"seed": 43},
        {"resume": Path("checkpoint.pt")},
    ],
)
def test_practice_envelope_cannot_silently_change(override, tmp_path):
    args = dict(
        dataset_root=tmp_path / "missing",
        history_root=tmp_path,
        experiment="practice-overfit-v1",
        steps=512,
        batch_size=8,
        seed=42,
        profile="reference",
        device="cuda",
    )
    args.update(override)
    with pytest.raises(ValueError, match="practice-overfit-v1 requires"):
        train(**args)
    assert not (tmp_path / "lewm-mvp").exists()


def test_default_mvp_budget_stays_bounded(tmp_path):
    with pytest.raises(ValueError, match="1..32"):
        train(dataset_root=tmp_path, steps=512)


def test_decoder_extension_requires_practice_checkpoint(monkeypatch, tmp_path):
    from dodge_native_game.variants.pixel_repr_ddqn import probe

    (tmp_path / "checkpoint.pt").write_bytes(b"fixture")
    monkeypatch.setattr(probe, "load_model", lambda _: (None, {"experiment": "mvp"}))
    with pytest.raises(ValueError, match="1..32"):
        probe.fit_probe(tmp_path, tmp_path, steps=256, device="cpu")
    monkeypatch.setattr(
        probe, "load_model", lambda _: (None, {"experiment": "practice-overfit-v1"})
    )
    with pytest.raises(ValueError, match="256 updates, batch8, CUDA"):
        probe.fit_probe(tmp_path, tmp_path, steps=256, batch_size=8, device="cpu")
