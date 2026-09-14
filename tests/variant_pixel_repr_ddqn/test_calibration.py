import pytest
import torch

from dodge_native_game.variants.pixel_repr_ddqn import calibration, pretrain
from dodge_native_game.variants.pixel_repr_ddqn.model import LeWMConfig, LeWorldModel
from dodge_native_game.variants.pixel_repr_ddqn.run_artifacts import file_hash


class Dataset:
    def __init__(self, *args, split="train", **kwargs):
        self.split = split

    def __len__(self):
        return 4

    def __getitem__(self, index):
        pixels = torch.arange(4 * 3 * 16 * 16).reshape(4, 3, 16, 16)
        return {
            "pixels": ((pixels + index * 17) % 256).to(torch.uint8),
            "actions": torch.tensor([0, 2, 8]),
        }


def test_calibration_rejects_validation():
    model = LeWorldModel(LeWMConfig.tiny())
    with pytest.raises(ValueError, match="train split"):
        calibration.calibrate_encoder(model, Dataset(split="validation"), "cpu")


def test_export_preserves_weights_parent_and_blocks_resume(monkeypatch, tmp_path):
    monkeypatch.setattr(pretrain, "PixelSequenceDataset", Dataset)
    monkeypatch.setattr(calibration, "PixelSequenceDataset", Dataset)
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text("{}")
    source = pretrain.train(
        dataset_root=data,
        history_root=tmp_path,
        run_id="raw",
        profile="tiny",
        steps=1,
        batch_size=2,
        device="cpu",
    )
    before = file_hash(source / "checkpoint.pt")
    derived = calibration.export_calibrated_run(
        source, data, tmp_path, "derived", device="cpu"
    )
    original = torch.load(source / "checkpoint.pt", weights_only=True)
    payload = torch.load(derived / "checkpoint.pt", weights_only=True)
    assert (
        payload["inference_only"]
        and "optimizer" not in payload
        and "sampling_rng" not in payload
    )
    assert file_hash(source / "checkpoint.pt") == before
    assert payload["calibration"]["parent_checkpoint_sha256"] == before
    assert (
        payload["calibration"]["windows"] == 4
        and payload["calibration"]["frame_occurrences"] == 16
    )
    allowed = payload["calibration"]["changed_keys"]
    for key, value in original["model"].items():
        if key not in allowed:
            assert torch.equal(value, payload["model"][key])
    with pytest.raises(ValueError, match="inference-only"):
        pretrain.train(
            dataset_root=data,
            history_root=tmp_path,
            run_id="bad-resume",
            profile="tiny",
            steps=1,
            batch_size=2,
            device="cpu",
            resume=derived / "checkpoint.pt",
        )
    assert not (tmp_path / "bad-resume").exists()
    model, _ = pretrain.load_model(derived / "checkpoint.pt")
    x = Dataset()[0]["pixels"].unsqueeze(0)
    changed = x.clone()
    changed[:, -1] = 255 - changed[:, -1]
    with torch.no_grad():
        torch.testing.assert_close(
            model.encode(x)[:, :-1], model.encode(changed)[:, :-1], rtol=0, atol=0
        )
    with pytest.raises(FileExistsError):
        calibration.export_calibrated_run(
            source, data, tmp_path, "derived", device="cpu"
        )
