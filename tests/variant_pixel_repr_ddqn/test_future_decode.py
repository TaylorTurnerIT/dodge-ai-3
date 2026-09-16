"""Bounded CPU checks for the predicted next-frame decode view (SPEC §AB)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dodge_native_game.variants.pixel_repr_ddqn import future_decode
from dodge_native_game.variants.pixel_repr_ddqn.future_decode import (
    diff_map,
    run_study,
    window_plan,
)
from dodge_native_game.variants.pixel_repr_ddqn.large_dataset import (
    LARGE_DATASET_FORMAT,
)
from dodge_native_game.variants.pixel_repr_ddqn.model import (
    INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC,
)
from dodge_native_game.variants.pixel_repr_ddqn.run_artifacts import file_hash
from dodge_native_game.variants.pixel_repr_ddqn.spatial_readout import (
    LocalPatchDecoder,
)

PALETTE = np.asarray(
    [[29, 43, 83], [41, 173, 255], [255, 241, 232]], dtype=np.uint8
)


def _write_episode(root: Path, split: str, index: int, count: int = 128) -> dict:
    path = root / "episodes" / split / f"episode-{index:06d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    split_offset = 0 if split == "train" else 1
    frames = np.empty((count + 1, 3, 128, 128), dtype=np.uint8)
    for step in range(count + 1):
        color = PALETTE[(index + step + split_offset) % 3]
        frames[step][:] = color[:, None, None]
        frames[step][:, step : step + 2, step : step + 2] = PALETTE[
            (index + step + split_offset + 1) % 3
        ][:, None, None]
    actions = np.asarray([(index + step) % 9 for step in range(count)], dtype=np.int64)
    terminated = np.zeros(count, dtype=np.bool_)
    truncated = np.zeros(count, dtype=np.bool_)
    truncated[-1] = True
    with path.open("wb") as stream:
        np.savez_compressed(
            stream,
            pixels=frames,
            actions=actions,
            terminated=terminated,
            truncated=truncated,
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "episode_id": f"{split}-{index:06d}",
        "split": split,
        "seed": (1000 if split == "train" else 2000) + index,
        "path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "transition_count": count,
        "config_identity": f"recipe-{split}",
    }


def _make_dataset(root: Path) -> None:
    records: dict[str, list[dict]] = {"train": [], "validation": []}
    for split, count in (("train", 2), ("validation", 2)):
        for index in range(count):
            records[split].append(_write_episode(root, split, index))
    manifest = {
        "schema_version": 1,
        "dataset_format": LARGE_DATASET_FORMAT,
        "variant": "pixel-repr-ddqn",
        "status": "complete",
        "ready_marker": "READY",
        "observation": {
            "profile": "native-rgb-v1",
            "shape": [3, 128, 128],
            "dtype": "uint8",
        },
        "action_count": 9,
        "step_frames": 4,
        "history_size_default": 3,
        "decisions_per_episode": 128,
        "episodes": records,
        "splits": {
            split: {
                "episode_count": len(rows),
                "episode_ids": [row["episode_id"] for row in rows],
                "seeds": [row["seed"] for row in rows],
                "transition_count": sum(row["transition_count"] for row in rows),
            }
            for split, rows in records.items()
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "READY").write_text(f"{LARGE_DATASET_FORMAT}\n")


class _StubWorld(torch.nn.Module):
    """Deterministic shape-correct stand-in for the frozen palette LeWM."""

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        batch, time = pixels.shape[:2]
        base = pixels.float().mean(dim=(2, 3, 4)) / 255.0
        return base.unsqueeze(-1).expand(batch, time, 192).contiguous()

    def predict(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return z + 0.01

    def encode_readout_tokens(self, pixels: torch.Tensor):
        cls = self.encode(pixels)
        patches = cls.unsqueeze(2).expand(*cls.shape[:2], 256, 192).contiguous()
        return cls, patches


def _write_world_and_decoder(tmp_path: Path, dataset: Path) -> tuple[Path, Path]:
    payload = {
        "input_arm": "palette",
        "step": 1024,
        "input_encoding": INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC,
        "data_hash": file_hash(dataset / "manifest.json"),
        "config": {"palette_rgb": PALETTE.tolist(), "history_size": 3},
        "model": {},
    }
    checkpoint = tmp_path / "world.pt"
    torch.save(payload, checkpoint)
    world_hash = file_hash(checkpoint)
    decoder = LocalPatchDecoder()
    decoder_path = tmp_path / "current-decoder.pt"
    torch.save(
        {
            "model": decoder.state_dict(),
            "decoder_kind": "local-patch",
            "representation": "patch",
            "world_model_sha256": world_hash,
        },
        decoder_path,
    )
    return checkpoint, decoder_path


def test_window_plan_picks_deterministic_mid_episode_starts() -> None:
    from types import SimpleNamespace

    records = [
        SimpleNamespace(count=128),
        SimpleNamespace(count=2),
        SimpleNamespace(count=9),
    ]
    plan = window_plan(records, history_size=3)
    assert plan == ((0, 62), (2, 2))
    assert window_plan(records, history_size=3, limit=1) == ((0, 62),)
    with pytest.raises(ValueError, match="no episode"):
        window_plan([SimpleNamespace(count=1)], history_size=3)
    with pytest.raises(TypeError, match="integer"):
        window_plan(records, history_size=True)


def test_diff_map_highlights_only_mismatched_pixels() -> None:
    observed = np.zeros((3, 4, 4), dtype=np.uint8)
    decoded = observed.copy()
    decoded[:, 1, 2] = [255, 0, 0]
    result = diff_map(observed, decoded)
    assert result.dtype == np.uint8
    assert result.shape == (3, 4, 4)
    assert (result[:, 1, 2] == 255).all()
    assert result.sum() == 3 * 255
    with pytest.raises(ValueError, match="shape"):
        diff_map(observed, np.zeros((3, 2, 2), dtype=np.uint8))
    with pytest.raises(TypeError, match="uint8"):
        diff_map(observed.astype(np.float32), decoded.astype(np.float32))


def test_palette_metrics_reports_precision_and_unchanged_errors() -> None:
    predicted = np.asarray([[0, 1], [1, 1]])
    target = np.asarray([[0, 1], [1, 0]])
    current = np.asarray([[0, 0], [1, 0]])
    metrics = future_decode._palette_metrics(predicted, target, current)

    assert metrics["changed_pixels"] == 1
    assert metrics["palette_changed_per_color_recall"] == [None, 1.0, None]
    assert metrics["palette_precision"] == [1.0, 2 / 3, None]
    assert metrics["palette_recall"] == [0.5, 1.0, None]
    assert metrics["palette_iou"] == [0.5, 2 / 3, None]
    assert metrics["predicted_occupancy"] == [0.25, 0.75, 0.0]
    assert metrics["unchanged_pixels"] == 3
    assert metrics["unchanged_false_positive_share"] == 1 / 3


def test_future_decode_actual_readout_trains_matched_head_with_cross_matrix(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = tmp_path / "dataset"
    _make_dataset(dataset)
    checkpoint, decoder_path = _write_world_and_decoder(tmp_path, dataset)
    world_hash = file_hash(checkpoint)
    ab_path = tmp_path / "ab-decoder.pt"
    torch.save(
        {
            "model": LocalPatchDecoder().state_dict(),
            "decoder_kind": "local-patch",
            "representation": "predicted",
            "world_model_sha256": world_hash,
            "latent_dim": 192,
        },
        ab_path,
    )

    monkeypatch.setattr(
        future_decode, "load_model", lambda path: (_StubWorld(), {})
    )

    history = tmp_path / "history"
    result = run_study(
        dataset,
        checkpoint,
        decoder_path,
        history,
        "actual-test",
        device="cpu",
        milestones=(1,),
        scene_count=2,
        batch_size=2,
        readout="actual",
        ab_decoder=ab_path,
        ab_source_run="future-test",
        eval_fractions=(0.5,),
    )

    assert result["readout"] == "actual-next-latent"
    assert result["ab_source_run"] == "future-test"
    assert result["ab_decoder_sha256"] == file_hash(ab_path)
    assert file_hash(checkpoint) == world_hash

    run = history / "actual-test"
    report = json.loads((run / "report.json").read_text())
    assert set(report["cross"]) == {
        "primary_on_actual",
        "primary_on_predicted",
        "primary_on_current",
        "reference_on_actual",
        "reference_on_predicted",
        "reference_on_current",
    }
    assert report["broad"]["window_count"] == 2
    assert set(report["broad"]["cells"]) == set(report["cross"])
    saved = torch.load(run / "decoder.pt", weights_only=True)
    assert saved["representation"] == "actual"
    rows = [
        json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()
    ]
    assert rows[0]["loss"] == rows[0]["actual_loss"]
    gallery = (history / "actual-test-comparison.html").read_text()
    assert "Next decoded (actual latent)" in gallery
    assert '"actual"' in gallery

    with pytest.raises(ValueError, match="ab_decoder is required"):
        run_study(
            dataset,
            checkpoint,
            decoder_path,
            history,
            "actual-bad",
            device="cpu",
            milestones=(1,),
            scene_count=2,
            batch_size=2,
            readout="actual",
        )
    with pytest.raises(ValueError, match="readout must be"):
        run_study(
            dataset,
            checkpoint,
            decoder_path,
            history,
            "actual-bad",
            device="cpu",
            milestones=(1,),
            scene_count=2,
            batch_size=2,
            readout="cls",
        )


def test_nearest_palette_classes_picks_closest_color() -> None:
    images = torch.tensor(
        [
            [
                [[29 / 255, 250 / 255], [40 / 255, 30 / 255]],
                [[43 / 255, 240 / 255], [175 / 255, 44 / 255]],
                [[83 / 255, 230 / 255], [250 / 255, 84 / 255]],
            ]
        ],
        dtype=torch.float32,
    )
    classes = future_decode._nearest_palette_classes(images, PALETTE)
    assert classes[0].tolist() == [[0, 2], [1, 0]]


def test_future_decode_study_bounded_cpu(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "dataset"
    _make_dataset(dataset)
    checkpoint, decoder_path = _write_world_and_decoder(tmp_path, dataset)
    world_hash = file_hash(checkpoint)

    monkeypatch.setattr(
        future_decode, "load_model", lambda path: (_StubWorld(), {})
    )

    history = tmp_path / "history"
    result = run_study(
        dataset,
        checkpoint,
        decoder_path,
        history,
        "future-test",
        device="cpu",
        milestones=(1,),
        scene_count=2,
        batch_size=2,
    )

    assert result["run_id"] == "future-test"
    assert result["world_model_updates"] == 0
    assert result["train_windows"] == 2
    assert result["scene_episode_ids"] == ["validation-000000", "validation-000001"]
    assert file_hash(checkpoint) == world_hash

    run = history / "future-test"
    report = json.loads((run / "report.json").read_text())
    assert report["step"] == 1
    assert report["loss_kind"] == "balanced-bright"
    assert report["equal_class_weights"] is True
    assert report["loss_class_weights"] == {"cream": 0.5, "other": 0.5}
    assert report["loss_normalization"] == "per-frame-then-batch"
    assert report["readout"] == "predicted-latent-broadcast"
    for key in (
        "primary",
        "persistence_control",
        "wrong_latent_control",
        "pixel_persistence_baseline",
    ):
        metrics = report[key]
        assert 0.0 <= metrics["class_error"] <= 1.0
        assert len(metrics["palette_changed_per_color_recall"]) == 3
    for key in ("primary", "persistence_control"):
        assert report[key]["mse"] >= 0.0
        assert "changed_region_mse" in report[key]
        assert len(report[key]["palette_precision"]) == 3
        assert len(report[key]["predicted_occupancy"]) == 3
        assert "unchanged_false_positive_share" in report[key]
    assert set(report["cross"]) == {
        "primary_on_actual",
        "primary_on_predicted",
        "primary_on_current",
    }
    assert report["broad"]["window_count"] == 6
    assert set(report["broad"]["cells"]) == set(report["cross"])

    rows = [
        json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1 and rows[0]["step"] == 1
    assert rows[0]["loss"] == rows[0]["predicted_loss"]
    assert rows[0]["loss_kind"] == "balanced-bright"
    assert rows[0]["equal_class_weights"] is True

    saved = torch.load(run / "decoder.pt", weights_only=True)
    assert saved["step"] == 1
    assert saved["world_model_sha256"] == world_hash

    images = sorted((run / "images" / "step-1").glob("*.png"))
    assert len(images) == 12
    names = {path.name for path in images}
    for index in range(2):
        for kind in (
            "current-observed",
            "current-decoded",
            "next-observed",
            "next-decoded",
            "next-persistence",
            "next-diff",
        ):
            assert f"scene-{index:02d}-{kind}.png" in names

    views = json.loads((run / "visualizations.json").read_text())
    assert [view["scene"] for view in views] == [
        "validation-000000",
        "validation-000001",
    ]
    assert views[0]["metadata"]["label"] == (
        "validation-000000 · next-frame prediction"
    )
    assert [frame["label"] for frame in views[0]["frames"]] == [
        "Observed current frame",
        "Decoded current frame (frozen patch readout)",
        "Observed next frame",
        "Decoded predicted next frame",
        "Decoded next frame from persistence control",
        "Pixel difference map (predicted vs observed next)",
    ]
    assert all(
        frame["image"].startswith("data:image/png;base64,")
        for frame in views[0]["frames"]
    )
    gallery = history / "future-test-comparison.html"
    assert gallery.exists() and "__RUN__" not in gallery.read_text()
    comparison = json.loads(
        (history / "future-test-future-comparison.json").read_text()
    )
    assert comparison["gallery"] == "future-test-comparison.html"
