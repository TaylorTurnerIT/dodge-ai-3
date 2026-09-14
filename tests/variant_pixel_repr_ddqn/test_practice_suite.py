from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dodge_native_game.variants.pixel_repr_ddqn.practice import load_practice
from dodge_native_game.variants.pixel_repr_ddqn.practice_suite import (
    DECISIONS_PER_CONFIG,
    SUITE_SEED,
    TOTAL_DECISIONS,
    generate_suite,
    suite_configs,
)


def test_suite_is_reproducible_bounded_and_action_diverse() -> None:
    first = suite_configs()
    second = suite_configs()
    assert first == second
    assert len(first) == 16
    assert [(split, seed) for split, seed, _ in first] == [
        ("train", seed) for seed in range(500, 512)
    ] + [("validation", seed) for seed in range(1500, 1504)]
    assert sum(config.max_decisions for _, _, config in first) == TOTAL_DECISIONS == 256
    assert len(first) * DECISIONS_PER_CONFIG == TOTAL_DECISIONS
    assert {
        command.action
        for split, _, config in first
        if split == "train"
        for command in config.player_script
    } == set(range(9))
    for _, _, config in first:
        assert config.invulnerable is True
        assert config.difficulty == 1
        assert config.permanent_pattern == 0
        assert len(config.player_script) == 8
        assert all(
            command.kind == "action" and command.decisions == 2
            for command in config.player_script
        )
        assert all(32 <= coordinate <= 96 for coordinate in config.player_start)
        assert len(config.enemies) == 2
        assert config.enemies[0].segments == ()
        moving = config.enemies[1]
        assert moving.loop is True
        assert len(moving.segments) == 2
        assert moving.segments[-1].target == moving.start
        assert all(
            2 <= coordinate <= 126
            for segment in moving.segments
            for coordinate in segment.target
        )


def test_configs_are_immutable_and_no_global_rng_is_needed() -> None:
    before = suite_configs()
    config = before[0][2]
    snapshot = config.to_dict()
    assert config.to_dict() == snapshot
    assert isinstance(config.player_script, tuple)
    assert isinstance(config.enemies, tuple)
    assert SUITE_SEED == 20260914
    assert suite_configs() == before


def test_generate_suite_captures_then_imports_explicit_splits(tmp_path: Path) -> None:
    calls: list[tuple[str, int]] = []

    def fake_generate(config_path: Path, capture: Path, seed: int) -> dict[str, object]:
        config = load_practice(config_path)
        capture.mkdir(parents=True)
        manifest = {
            "config_sha256": config.resolved_sha256(),
            "source_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "seed": seed,
        }
        (capture / "manifest.json").write_text(json.dumps(manifest))
        calls.append((config.name.split("-")[0], seed))
        return manifest

    def fake_importer(
        corpus: Path, train: list[Path], validation: list[Path]
    ) -> dict[str, object]:
        assert len(calls) == 16
        assert [path.name for path in train] == [
            f"{seed:05d}" for seed in range(500, 512)
        ]
        assert [path.name for path in validation] == [
            f"{seed:05d}" for seed in range(1500, 1504)
        ]
        corpus.mkdir()
        (corpus / "manifest.json").write_text(json.dumps({"dataset_id": "mock"}))
        return {"dataset_id": "mock"}

    result = generate_suite(
        tmp_path / "suite", generator=fake_generate, importer=fake_importer
    )
    assert result["seed"] == SUITE_SEED
    assert result["planned_decisions"] == 256
    assert len(result["splits"]["train"]) == 12  # type: ignore[index]
    assert len(result["splits"]["validation"]) == 4  # type: ignore[index]
    assert (tmp_path / "suite" / "manifest.json").is_file()
    assert result["corpus"]["dataset_id"] == "mock"  # type: ignore[index]
