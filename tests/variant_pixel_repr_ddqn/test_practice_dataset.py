from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from dodge_native_game.variants.pixel_repr_ddqn.dataset import (
    DatasetValidationError,
    PixelSequenceDataset,
)
from dodge_native_game.variants.pixel_repr_ddqn.practice import (
    PracticeConfig,
    generate_practice,
)
from dodge_native_game.variants.pixel_repr_ddqn.practice_dataset import (
    import_practice_dataset,
)


class CaptureEnv:
    def __init__(self, **kwargs):
        self.steps = kwargs["commands"][0][3]
        self.index = 0

    def reset(self, seed):
        self.seed = seed
        return np.full((128, 128), seed % 16, dtype=np.uint8)

    def step(self):
        self.index += 1
        return {
            "pixels": np.full(
                (128, 128), (self.seed + self.index) % 16, dtype=np.uint8
            ),
            "action": 2,
            "terminated": False,
            "failed": False,
            "finished": self.index == self.steps,
        }


def capture(path, seed, *, invulnerable=True, count=4):
    config = PracticeConfig.from_mapping(
        {
            "version": 1,
            "name": "test",
            "step_frames": 4,
            "max_decisions": count,
            "difficulty": 1,
            "permanent_pattern": 0,
            "invulnerable": invulnerable,
            "player_start": [32.0, 48.0],
            "player_script": [{"action": "right", "decisions": count}],
            "enemies": [],
        }
    )
    manifest = generate_practice(config, path, seed, native_factory=CaptureEnv)
    # Synthetic artifact provenance, not a claim of native execution.
    manifest["native_module_sha256"] = "a" * 64
    save(path, manifest)
    return path


def save(path, manifest):
    (path / "manifest.json").write_text(json.dumps(manifest))


def rewrite_episode(path, edit):
    with np.load(path / "episode.npz") as source:
        arrays = {key: source[key] for key in source.files}
    edit(arrays)
    np.savez_compressed(path / "episode.npz", **arrays)
    manifest = json.loads((path / "manifest.json").read_text())
    blob = (path / "episode.npz").read_bytes()
    manifest["files"]["episode"].update(
        sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob)
    )
    save(path, manifest)


def test_roundtrip_preserves_bytes_provenance_and_pixel_action_boundary(tmp_path):
    train = capture(tmp_path / "train", 1)
    validation = capture(tmp_path / "val", 2)
    out = tmp_path / "out"
    manifest = import_practice_dataset(out, [train], [validation])
    assert manifest["collection"]["new_native_decisions"] == 0
    for split, source in [("train", train), ("validation", validation)]:
        record = manifest["datasets"][split][0]
        assert (out / record["path"]).read_bytes() == (
            source / "episode.npz"
        ).read_bytes()
        assert (out / record["practice"]["manifest_path"]).read_bytes() == (
            source / "manifest.json"
        ).read_bytes()
        dataset = PixelSequenceDataset(out, split=split)
        assert len(dataset) == 2
        with np.load(source / "episode.npz") as arrays:
            for index, item in enumerate(dataset):
                assert set(item) == {"pixels", "actions", "episode_id", "start"}
                np.testing.assert_array_equal(
                    item["pixels"], arrays["pixels"][index : index + 4]
                )
                np.testing.assert_array_equal(
                    item["actions"], arrays["actions"][index : index + 3]
                )
    with pytest.raises(FileExistsError):
        import_practice_dataset(out, [train], [validation])


@pytest.mark.parametrize(
    "defect",
    [
        "hash",
        "config",
        "trace",
        "boundary",
        "shape",
        "overlap",
        "same_bytes",
        "lethal",
        "short",
        "escape",
        "native",
        "incomplete",
    ],
)
def test_rejects_invalid_capture_without_publishing(tmp_path, defect):
    train = capture(tmp_path / "train", 1)
    validation = capture(
        tmp_path / "val",
        2,
        invulnerable=defect != "lethal",
        count=2 if defect == "short" else 4,
    )
    manifest = json.loads((validation / "manifest.json").read_text())
    if defect == "hash":
        manifest["files"]["episode"]["sha256"] = "0" * 64
    elif defect == "config":
        manifest["config"]["player_start"][0] += 1
    elif defect == "trace":
        manifest["action_trace"][0] = 0
    elif defect == "overlap":
        manifest["seed"] = 1
    elif defect == "same_bytes":
        (validation / "episode.npz").write_bytes((train / "episode.npz").read_bytes())
        manifest["files"]["episode"] = json.loads(
            (train / "manifest.json").read_text()
        )["files"]["episode"]
    elif defect == "escape":
        manifest["files"]["episode"]["path"] = "../train/episode.npz"
    elif defect == "native":
        manifest["native_module_sha256"] = None
    elif defect == "incomplete":
        manifest["status"] = "running"
    save(validation, manifest)
    if defect == "boundary":
        rewrite_episode(validation, lambda a: a["truncated"].__setitem__(0, True))
    if defect == "shape":
        rewrite_episode(validation, lambda a: a.update(pixels=a["pixels"][:, :, :, 1:]))
    with pytest.raises((ValueError, KeyError)):
        import_practice_dataset(tmp_path / "out", [train], [validation])
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out*"))


def test_short_terminal_episode_retained_without_crossing(tmp_path):
    train = capture(tmp_path / "train", 1, invulnerable=False)
    validation = capture(tmp_path / "val", 2, invulnerable=False)
    death = capture(tmp_path / "death", 3, invulnerable=False, count=1)
    rewrite_episode(
        death,
        lambda a: a.update(terminated=np.array([True]), truncated=np.array([False])),
    )
    manifest = json.loads((death / "manifest.json").read_text())
    manifest.update(
        terminated=True,
        truncated=False,
        finished=False,
        success=False,
        failure="terminated",
    )
    save(death, manifest)
    out = tmp_path / "out"
    result = import_practice_dataset(out, [train, death], [validation])
    assert result["datasets"]["train"][1]["terminal"]
    assert len(PixelSequenceDataset(out)) == 2


def test_total_budget_checked_before_publication(tmp_path):
    train = capture(tmp_path / "train", 1, count=129)
    validation = capture(tmp_path / "val", 2, count=128)
    with pytest.raises(DatasetValidationError, match="decision cap"):
        import_practice_dataset(tmp_path / "out", [train], [validation])
    assert not (tmp_path / "out").exists()
