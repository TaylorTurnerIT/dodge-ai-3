from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from dodge_native_game.variants.pixel_repr_ddqn import input_pretrain
from dodge_native_game.variants.pixel_repr_ddqn.model import (
    INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC,
    INPUT_ENCODING_RGB_NEAREST_SYMMETRIC,
    LeWMConfig,
)


def _palette_provenance() -> input_pretrain.PaletteProvenance:
    colors = ((29, 43, 83), (41, 173, 255), (255, 241, 232))
    palette_bytes = bytes(channel for color in colors for channel in color)
    return input_pretrain.PaletteProvenance(
        palette_rgb=colors,
        palette_sha256=hashlib.sha256(palette_bytes).hexdigest(),
        dataset_manifest_sha256="d" * 64,
        frame_index_sha256="f" * 64,
        selected_pixels_sha256="p" * 64,
        frame_count=4,
        frames_per_episode=4,
        sampling_seed=903,
    )


def test_discover_palette_reads_selected_train_frames_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    colors = np.asarray(
        [[29, 43, 83], [41, 173, 255], [255, 241, 232]], dtype=np.uint8
    )
    train_record = SimpleNamespace(count=128, episode_id="train-0", split="train")
    validation_record = SimpleNamespace(
        count=128, episode_id="validation-0", split="validation"
    )
    train_pixels = np.zeros((129, 3, 128, 128), dtype=np.uint8)
    for frame, color in zip((1, 2, 3, 4), (*colors, colors[0]), strict=True):
        train_pixels[frame] = color[:, None, None]
    validation_pixels = np.full((129, 3, 128, 128), 7, dtype=np.uint8)
    opened: list[str] = []

    monkeypatch.setattr(
        input_pretrain,
        "read_dataset_metadata",
        lambda root: SimpleNamespace(
            records={
                "train": (train_record,),
                "validation": (validation_record,),
            }
        ),
    )
    monkeypatch.setattr(
        input_pretrain,
        "_select_frames",
        lambda count, frames_per_episode, rng: np.asarray([1, 2, 3, 4]),
    )

    def read_episode(record, *, verify_hash):
        opened.append(record.split)
        return (
            train_pixels if record.split == "train" else validation_pixels,
            np.zeros(128, dtype=np.int64),
        )

    monkeypatch.setattr(input_pretrain, "_read_episode", read_episode)
    monkeypatch.setattr(input_pretrain, "file_hash", lambda path: "d" * 64)

    provenance = input_pretrain.discover_palette(tmp_path)

    assert opened == ["train"]
    assert provenance.palette_rgb == tuple(
        tuple(int(v) for v in color) for color in colors
    )
    assert provenance.source_split == "train"
    assert provenance.frame_count == 4
    assert provenance.frames_per_episode == 4
    assert provenance.sampling_seed == 903
    assert provenance.palette_sha256 == hashlib.sha256(colors.tobytes()).hexdigest()


class _TinyDataset:
    split = "train"

    def __len__(self) -> int:
        return 5

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "pixels": torch.full((4, 3, 2, 2), index, dtype=torch.uint8),
            "actions": torch.full((3,), index % 9, dtype=torch.int64),
            "episode_id": f"episode-{index}",
            "start": index,
        }


def test_sample_shared_batch_is_reproducible_and_single_materialization() -> None:
    left_rng = torch.Generator(device="cpu").manual_seed(43)
    right_rng = torch.Generator(device="cpu").manual_seed(43)
    left = input_pretrain.sample_shared_batch(_TinyDataset(), 8, left_rng)
    right = input_pretrain.sample_shared_batch(_TinyDataset(), 8, right_rng)

    assert left.indices == right.indices
    assert left.episode_ids == right.episode_ids
    assert left.starts == right.starts
    torch.testing.assert_close(left.pixels, right.pixels, rtol=0, atol=0)
    torch.testing.assert_close(left.actions, right.actions, rtol=0, atol=0)
    assert left.trace_entry(7) == right.trace_entry(7)


def test_sample_shared_batch_rejects_validation_dataset() -> None:
    validation = SimpleNamespace(split="validation")
    with pytest.raises(ValueError, match="train split"):
        input_pretrain.sample_shared_batch(
            validation, 2, torch.Generator(device="cpu").manual_seed(43)
        )


class _StochasticModel(nn.Module):
    def __init__(self, config: LeWMConfig) -> None:
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.tensor(0.25))

    def compute_loss(
        self, pixels: torch.Tensor, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        noise = torch.randn((), device=pixels.device)
        prediction = self.weight * pixels.float().mean() + noise
        loss = prediction.square()
        return {"loss": loss, "pred_loss": loss, "sigreg_loss": loss * 0}


def test_paired_update_restores_per_arm_stochastic_state() -> None:
    config = LeWMConfig.tiny(input_encoding=INPUT_ENCODING_RGB_NEAREST_SYMMETRIC)
    torch.manual_seed(42)
    rgb = _StochasticModel(config)
    torch.manual_seed(42)
    palette = _StochasticModel(config)
    optimizers = {
        "rgb": torch.optim.AdamW(rgb.parameters(), lr=5e-5, weight_decay=1e-3),
        "palette": torch.optim.AdamW(
            palette.parameters(), lr=5e-5, weight_decay=1e-3
        ),
    }
    batch = input_pretrain.SharedBatch(
        indices=(0, 1),
        episode_ids=("episode-0", "episode-1"),
        starts=(0, 1),
        pixels=torch.ones((2, 4, 3, 2, 2), dtype=torch.uint8),
        actions=torch.zeros((2, 3), dtype=torch.int64),
    )
    torch.manual_seed(44)
    before = input_pretrain._capture_rng("cpu")
    metrics, after = input_pretrain._paired_update(
        {"rgb": rgb, "palette": palette},
        optimizers,
        batch,
        stochastic_state=before,
        device="cpu",
    )

    assert metrics["rgb"]["loss"] == pytest.approx(metrics["palette"]["loss"])
    assert input_pretrain._same_rng(after["rgb"], after["palette"])
    torch.testing.assert_close(rgb.weight, palette.weight, rtol=0, atol=0)


def test_build_models_resets_initialization_for_both_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel(nn.Module):
        def __init__(self, config: LeWMConfig) -> None:
            super().__init__()
            self.config = config
            self.weight = nn.Parameter(torch.randn(()))

    monkeypatch.setattr(input_pretrain, "LeWorldModel", FakeModel)
    models = input_pretrain._build_models(_palette_provenance(), device="cpu")

    torch.testing.assert_close(
        models["rgb"].weight, models["palette"].weight, rtol=0, atol=0
    )
    assert models["rgb"].config.input_encoding == (
        INPUT_ENCODING_RGB_NEAREST_SYMMETRIC
    )
    assert models["palette"].config.input_encoding == (
        INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC
    )
    assert models["palette"].config.palette_rgb == _palette_provenance().palette_rgb


def test_checkpoint_payload_round_trips_new_config_fields() -> None:
    config = LeWMConfig.tiny(
        input_encoding=INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC,
        palette_rgb=_palette_provenance().palette_rgb,
    )
    model = _StochasticModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    metadata = input_pretrain._arm_metadata(
        run_id="screen-palette",
        arm="palette",
        config=config,
        config_payload=input_pretrain._canonical_config(config),
        palette=_palette_provenance(),
        data_hash="d" * 64,
        pair_run_id="screen",
        initial_state_sha256="i" * 64,
        parameter_count=11,
    )
    state = input_pretrain._capture_rng("cpu")
    payload = input_pretrain._checkpoint_payload(
        model=model,
        optimizer=optimizer,
        config_payload=metadata["config"],  # type: ignore[arg-type]
        metadata=metadata,
        step=512,
        sampling_rng=torch.Generator(device="cpu").manual_seed(43),
        post_rng=state,
        sample_trace_hash="s" * 64,
        stochastic_trace_hash="t" * 64,
    )

    loaded = LeWMConfig(**payload["config"])  # type: ignore[arg-type]
    assert payload["experiment"] == input_pretrain.EXPERIMENT
    assert payload["step"] == 512
    assert payload["inference_only"] is False
    assert loaded.input_encoding == INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC
    assert loaded.palette_rgb == _palette_provenance().palette_rgb
    assert payload["initial_state_sha256"] == "i" * 64
    assert payload["parameter_count"] == 11


def test_run_protocol_is_fixed_and_cuda_only() -> None:
    with pytest.raises(ValueError, match="exactly 1024"):
        input_pretrain._validate_run_protocol(
            steps=512, batch_size=32, device="cuda", threads=2
        )
    with pytest.raises(ValueError, match="batch32"):
        input_pretrain._validate_run_protocol(
            steps=1024, batch_size=8, device="cuda", threads=2
        )
    with pytest.raises(ValueError, match="CUDA T4"):
        input_pretrain._validate_run_protocol(
            steps=1024, batch_size=32, device="cpu", threads=2
        )


def test_run_pair_writes_checkpoints_accepted_by_input_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from dodge_native_game.variants.pixel_repr_ddqn import input_probe

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "manifest.json").write_bytes(b"{\"fixture\":true}\n")
    data_hash = input_pretrain.file_hash(dataset_root / "manifest.json")
    palette = replace(
        _palette_provenance(), dataset_manifest_sha256=data_hash
    )

    class FakeDataset:
        split = "train"

        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> dict[str, object]:
            return {
                "pixels": torch.full(
                    (3, 3, 8, 8), index % 3, dtype=torch.uint8
                ),
                "actions": torch.zeros((3, 9), dtype=torch.int64),
                "episode_id": f"episode-{index}",
                "start": index,
            }

    class FakeModel(nn.Module):
        def __init__(self, config: LeWMConfig) -> None:
            super().__init__()
            self.config = config
            self.weight = nn.Parameter(torch.tensor(0.25))

        def compute_loss(
            self, pixels: torch.Tensor, actions: torch.Tensor
        ) -> dict[str, torch.Tensor]:
            del actions
            prediction = self.weight * pixels.float().mean() + torch.randn(())
            loss = prediction.square()
            return {"loss": loss, "pred_loss": loss, "sigreg_loss": loss * 0}

    monkeypatch.setattr(input_pretrain, "discover_palette", lambda root: palette)
    monkeypatch.setattr(
        input_pretrain, "LargePixelSequenceDataset", FakeDataset
    )
    monkeypatch.setattr(input_pretrain, "LeWorldModel", FakeModel)
    monkeypatch.setattr(input_pretrain, "append_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(input_pretrain, "write_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        input_pretrain, "_validate_run_protocol", lambda **kwargs: None
    )
    monkeypatch.setattr(input_pretrain.torch.cuda, "manual_seed_all", lambda seed: None)

    result = input_pretrain.run_pair(
        dataset_root,
        tmp_path / "history",
        run_id="fixture-input-pair",
        steps=input_pretrain.STEPS,
        batch_size=input_pretrain.BATCH_SIZE,
        device="cpu",
        threads=1,
    )

    checkpoints = result["checkpoints"]
    payloads = {
        arm: torch.load(path, map_location="cpu", weights_only=True)
        for arm, path in checkpoints.items()  # type: ignore[union-attr]
    }
    input_probe.validate_world_pair(payloads, data_hash)
    pair_manifest = json.loads(
        (tmp_path / "history" / "fixture-input-pair-pair.json").read_text()
    )
    assert pair_manifest["matched_initial_state"] is True
    assert pair_manifest["initial_state_sha256"]["rgb"] == pair_manifest[
        "initial_state_sha256"
    ]["palette"]
    assert pair_manifest["parameter_count"] == {"rgb": 1, "palette": 1}
    for arm, payload in payloads.items():
        assert payload["step"] == 1024
        assert payload["initial_state_sha256"] == pair_manifest[
            "initial_state_sha256"
        ][arm]
        assert payload["parameter_count"] == 1
        checkpoint_512 = (
            tmp_path
            / "history"
            / f"fixture-input-pair-{arm}"
            / "checkpoint-512.pt"
        )
        assert checkpoint_512.exists()


def _continue_fixture(monkeypatch: pytest.MonkeyPatch, palette) -> None:
    from torch import nn

    class FakeDataset:
        split = "train"

        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> dict[str, object]:
            return {
                "pixels": torch.full(
                    (3, 3, 8, 8), index % 3, dtype=torch.uint8
                ),
                "actions": torch.zeros((3, 9), dtype=torch.int64),
                "episode_id": f"episode-{index}",
                "start": index,
            }

    class FakeModel(nn.Module):
        def __init__(self, config: LeWMConfig) -> None:
            super().__init__()
            self.config = config
            self.weight = nn.Parameter(torch.tensor(0.25))

        def compute_loss(
            self, pixels: torch.Tensor, actions: torch.Tensor
        ) -> dict[str, torch.Tensor]:
            del actions
            prediction = self.weight * pixels.float().mean() + torch.randn(())
            loss = prediction.square()
            return {"loss": loss, "pred_loss": loss, "sigreg_loss": loss * 0}

    monkeypatch.setattr(input_pretrain, "discover_palette", lambda root: palette)
    monkeypatch.setattr(
        input_pretrain, "LargePixelSequenceDataset", FakeDataset
    )
    monkeypatch.setattr(input_pretrain, "LeWorldModel", FakeModel)
    monkeypatch.setattr(input_pretrain, "append_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(input_pretrain, "write_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        input_pretrain, "_validate_run_protocol", lambda **kwargs: None
    )
    monkeypatch.setattr(
        input_pretrain, "_validate_continue_protocol", lambda **kwargs: None
    )
    monkeypatch.setattr(input_pretrain.torch.cuda, "manual_seed_all", lambda seed: None)


def test_continue_run_rejects_bad_protocol() -> None:
    with pytest.raises(ValueError, match="at least one extra"):
        input_pretrain._validate_continue_protocol(
            extra_steps=0, batch_size=32, device="cpu", threads=1
        )
    with pytest.raises(ValueError, match="batch32"):
        input_pretrain._validate_continue_protocol(
            extra_steps=8, batch_size=8, device="cpu", threads=1
        )


def test_continue_run_matches_fresh_prefix_plus_suffix(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "manifest.json").write_bytes(b"{\"fixture\":true}\n")
    data_hash = input_pretrain.file_hash(dataset_root / "manifest.json")
    palette = replace(
        _palette_provenance(), dataset_manifest_sha256=data_hash
    )
    _continue_fixture(monkeypatch, palette)
    monkeypatch.setattr(input_pretrain, "CHECKPOINT_STEPS", (4, 8))
    monkeypatch.setattr(input_pretrain, "STEPS", 8)

    fresh = input_pretrain.run_pair(
        dataset_root,
        tmp_path / "history-fresh",
        run_id="fixture-fresh",
        steps=8,
        batch_size=input_pretrain.BATCH_SIZE,
        device="cpu",
        threads=1,
    )
    half = tmp_path / "history-fresh" / "fixture-fresh-palette" / "checkpoint-4.pt"
    assert half.is_file()
    resumed = input_pretrain.continue_run(
        half,
        dataset_root,
        tmp_path / "history-resumed",
        "fixture-resumed",
        extra_steps=4,
        checkpoint_every=4,
        batch_size=input_pretrain.BATCH_SIZE,
        device="cpu",
        threads=1,
    )
    fresh_payload = torch.load(
        fresh["checkpoints"]["palette"], map_location="cpu", weights_only=False
    )
    resumed_payload = torch.load(
        resumed["checkpoint"], map_location="cpu", weights_only=False
    )
    assert resumed_payload["step"] == 8
    assert resumed_payload["sample_trace_sha256"] == (
        fresh_payload["sample_trace_sha256"]
    )
    assert resumed_payload["stochastic_trace_sha256"] == (
        fresh_payload["stochastic_trace_sha256"]
    )
    for key, fresh_value in fresh_payload["model"].items():
        assert torch.equal(fresh_value, resumed_payload["model"][key])
    assert torch.equal(
        fresh_payload["sampling_rng"], resumed_payload["sampling_rng"]
    )
    manifest = json.loads(resumed["manifest"].read_text())
    assert manifest["base_step"] == 4
    assert manifest["steps"] == 8
