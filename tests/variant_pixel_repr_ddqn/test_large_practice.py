"""Focused planner and collector checks for the large practice corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from dodge_native_game.variants.pixel_repr_ddqn.large_dataset import (
    validate_dataset,
)
from dodge_native_game.variants.pixel_repr_ddqn.large_practice import (
    RECIPE_FAMILIES,
    LargePracticeError,
    collect_large_practice,
    plan_large_practice,
)


def _semantic_config(spec: object) -> str:
    config = spec.config.to_dict()  # type: ignore[attr-defined]
    config["name"] = "<recipe>"
    # Difficulty is frozen at native difficulty 1 in both splits.  Keep this
    # normalization explicit so a future split marker cannot masquerade as a
    # distinct recipe when every authored scene is otherwise identical.
    config.pop("difficulty")
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


@pytest.fixture(scope="module")
def full_plan() -> list[object]:
    return plan_large_practice()


def test_planner_is_deterministic_balanced_and_split_disjoint(
    full_plan: list[object],
) -> None:
    first = full_plan
    second = plan_large_practice()

    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert [item.seed for item in first[:4096]] == list(range(4000, 8096))
    assert [item.seed for item in first[4096:]] == list(range(16000, 16512))
    assert {item.recipe_family for item in first} == set(RECIPE_FAMILIES)
    assert all(
        sum(item.recipe_family == family for item in first[:4096]) == 256
        for family in RECIPE_FAMILIES
    )
    assert all(
        sum(item.recipe_family == family for item in first[4096:]) == 32
        for family in RECIPE_FAMILIES
    )

    train_recipes = {_semantic_config(item) for item in first[:4096]}
    validation_recipes = {_semantic_config(item) for item in first[4096:]}
    assert train_recipes.isdisjoint(validation_recipes)
    assert len(train_recipes) == 4096
    assert len(validation_recipes) == 512


def test_planner_keeps_player_only_controls_empty_and_geometry_native_valid(
    full_plan: list[object],
) -> None:
    plan = full_plan
    for item in plan:
        config = item.config
        assert config.invulnerable is True
        assert config.difficulty == 1
        assert config.step_frames == 4
        assert config.max_decisions == 128
        assert 4.0 <= config.player_start[0] <= 124.0
        assert 4.0 <= config.player_start[1] <= 124.0
        assert len(config.player_script) <= 64
        assert sum(command.decisions for command in config.player_script) == 128
        if item.recipe_family.startswith("player."):
            assert config.enemies == ()
            assert config.permanent_pattern == 0
        for enemy in config.enemies:
            half = enemy.size / 2.0
            assert 2 <= enemy.size <= 16
            assert half <= enemy.start[0] <= 128.0 - half
            assert half <= enemy.start[1] <= 128.0 - half
            if enemy.loop:
                assert enemy.segments[-1].target == enemy.start
            else:
                assert enemy.segments == ()
            for segment in enemy.segments:
                assert half <= segment.target[0] <= 128.0 - half
                assert half <= segment.target[1] <= 128.0 - half
                assert segment.frames >= 1
    assert {item.config.permanent_pattern for item in plan} <= set(range(40))
    assert {item.config.permanent_pattern for item in plan} & set(range(1, 40))


def test_native_collection_has_strict_pixel_action_arrays_and_no_gifs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "large-native"
    manifest = collect_large_practice(
        root,
        train_episodes=32,
        validation_episodes=16,
        workers=1,
    )

    assert manifest["collection"]["native_decisions"] == 48 * 128
    assert manifest["episode_action_count"] == 128
    assert (root / "READY").read_text() == "pixel-repr-ddqn-large-practice-v1\n"
    validate_dataset(root)
    assert not list(root.rglob("*.gif"))
    record = manifest["episodes"]["train"][0]
    with np.load(root / record["path"], allow_pickle=False) as archive:
        assert set(archive.files) == {
            "pixels",
            "actions",
            "terminated",
            "truncated",
        }
        assert archive["pixels"].shape == (129, 3, 128, 128)
        assert archive["pixels"].dtype == np.uint8
        assert archive["actions"].shape == (128,)
        assert archive["actions"].dtype == np.int64
        assert archive["terminated"].dtype == np.bool_
        assert archive["truncated"].dtype == np.bool_
        assert not archive["terminated"][:-1].any()
        assert not archive["truncated"][:-1].any()
        assert bool(archive["terminated"][-1]) != bool(archive["truncated"][-1])
    assert not {
        record["seed"] for record in manifest["episodes"]["train"]
    }.intersection(record["seed"] for record in manifest["episodes"]["validation"])


class _FakePracticeEnv:
    """Fast native-shaped fixture that still reports executed actions."""

    fail_on_instance: int | None = None
    instances = 0

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        type(self).instances += 1
        self.instance = type(self).instances
        self.step_count = 0

    def reset(self, seed: int) -> np.ndarray:
        if type(self).fail_on_instance == self.instance:
            raise RuntimeError("injected collection interruption")
        self.seed = seed
        self.step_count = 0
        return np.full((128, 128), self.seed // 16 % 16, dtype=np.uint8)

    def step(self) -> dict[str, object]:
        self.step_count += 1
        return {
            "pixels": np.full(
                (128, 128),
                (self.seed // 16 + self.seed + self.step_count) % 16,
                dtype=np.uint8,
            ),
            "action": self.step_count % 9,
            "terminated": False,
            "finished": self.step_count == 128,
            "failed": False,
        }

    def close(self) -> None:
        return None


def test_resume_preserves_completed_receipt_and_collects_missing_episode(
    tmp_path: Path,
) -> None:
    _FakePracticeEnv.instances = 0
    _FakePracticeEnv.fail_on_instance = 2
    root = tmp_path / "resume"
    with pytest.raises(RuntimeError, match="interruption"):
        collect_large_practice(
            root,
            train_episodes=1,
            validation_episodes=1,
            native_factory=_FakePracticeEnv,
        )
    receipt = root / "receipts/train/episode-000000.json"
    receipt_bytes = receipt.read_bytes()
    assert not (root / "READY").exists()

    _FakePracticeEnv.instances = 0
    _FakePracticeEnv.fail_on_instance = None
    manifest = collect_large_practice(
        root,
        train_episodes=1,
        validation_episodes=1,
        native_factory=_FakePracticeEnv,
    )
    assert receipt.read_bytes() == receipt_bytes
    assert manifest["collection"]["native_decisions"] == 2 * 128
    validate_dataset(root)


def test_corrupted_episode_is_rejected_before_resume(tmp_path: Path) -> None:
    _FakePracticeEnv.instances = 0
    _FakePracticeEnv.fail_on_instance = None
    root = tmp_path / "corrupt"
    collect_large_practice(
        root,
        train_episodes=1,
        validation_episodes=1,
        native_factory=_FakePracticeEnv,
    )
    episode = root / "episodes/train/episode-000000.npz"
    episode.write_bytes(episode.read_bytes() + b"corrupt")
    (root / "READY").unlink()
    with pytest.raises(LargePracticeError, match="hash mismatch"):
        collect_large_practice(
            root,
            train_episodes=1,
            validation_episodes=1,
            native_factory=_FakePracticeEnv,
        )


def test_collection_hashes_match_receipt(tmp_path: Path) -> None:
    _FakePracticeEnv.instances = 0
    _FakePracticeEnv.fail_on_instance = None
    root = tmp_path / "hashes"
    manifest = collect_large_practice(
        root,
        train_episodes=1,
        validation_episodes=1,
        native_factory=_FakePracticeEnv,
    )
    for record in manifest["episodes"]["train"] + manifest["episodes"]["validation"]:
        path = root / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


@pytest.mark.parametrize("defect", ["source", "native", "dependency", "capture"])
def test_import_rejects_unmatched_capture_provenance(
    tmp_path: Path, defect: str
) -> None:
    from dodge_native_game.variants.pixel_repr_ddqn import large_practice as module

    current = module._collection_provenance(None)
    old = json.loads(json.dumps(current))
    source = tmp_path / "source.py"
    source.write_bytes(Path(module.__file__).read_bytes())
    if defect == "source":
        source.write_text(source.read_text() + "\n# changed\n")
    elif defect == "native":
        old["native_module_sha256"] = "0" * 64
    elif defect == "dependency":
        old["source_hashes"]["pixels"]["sha256"] = "0" * 64
    else:
        source.write_text(
            source.read_text().replace(
                "frames.append(_native_rgb", "frames.insert(0, _native_rgb"
            )
        )
        old["generator_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    (tmp_path / "plan.json").write_text(json.dumps({"plan": {"provenance": old}}))
    with pytest.raises(LargePracticeError):
        module._import_provenance(tmp_path, source, current)
