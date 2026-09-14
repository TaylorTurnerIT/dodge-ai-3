"""Deterministic, bounded scripted-practice suite for the pixel variant."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from . import practice
from .practice_dataset import import_practice_dataset

SUITE_SEED = 20260914
TRAIN_SEEDS = tuple(range(500, 512))
VALIDATION_SEEDS = tuple(range(1500, 1504))
DECISIONS_PER_CONFIG, TOTAL_DECISIONS = 16, 256
Config = practice.PracticeConfig


def _point(point: tuple[float, float]) -> str:
    return f"[{point[0]:.1f}, {point[1]:.1f}]"


def _toml(config: Config) -> str:
    lines = [
        "version = 1",
        f"name = {json.dumps(config.name)}",
        "step_frames = 4",
        f"max_decisions = {config.max_decisions}",
        f"difficulty = {config.difficulty}",
        f"permanent_pattern = {config.permanent_pattern}",
        f"invulnerable = {str(config.invulnerable).lower()}",
        f"player_start = {_point(config.player_start)}",
    ]
    for command in config.player_script:
        assert command.action is not None
        lines += [
            "",
            "[[player_script]]",
            f"action = {json.dumps(practice.ACTION_NAMES[command.action])}",
            f"decisions = {command.decisions}",
        ]
    for enemy in config.enemies:
        segments = ", ".join(
            f"{{to = {_point(segment.target)}, frames = {segment.frames}}}"
            for segment in enemy.segments
        )
        lines += [
            "",
            "[[enemies]]",
            f"start = {_point(enemy.start)}",
            f"size = {enemy.size}",
            f"loop = {str(enemy.loop).lower()}",
            f"segments = [{segments}]",
        ]
    return "\n".join(lines) + "\n"


def suite_configs() -> list[tuple[str, int, Config]]:
    """Return the frozen 12-train/4-validation authoring protocol."""
    rng = np.random.default_rng(SUITE_SEED)
    pools = [
        rng.permutation(np.arange(lo, hi + 1, dtype=np.int64))[:16]
        for lo, hi in ((32, 96), (32, 96), (16, 112), (16, 112), (24, 104), (24, 104))
    ]
    result, index = [], 0
    for split, seeds in (("train", TRAIN_SEEDS), ("validation", VALIDATION_SEEDS)):
        for seed in seeds:
            actions = [int(a) for a in rng.permutation(9)[:8]]
            if index == 0:
                actions = list(range(8))
            elif index == 1:
                actions[0] = 8
            start = (float(pools[0][index]), float(pools[1][index]))
            static = (float(pools[2][index]), float(pools[3][index]))
            moving = (float(pools[4][index]), float(pools[5][index]))
            axis, span = int(rng.integers(0, 2)), int(rng.integers(12, 25))
            direction = -1 if index % 2 else 1
            target = list(moving)
            target[axis] = max(8.0, min(120.0, target[axis] + direction * span))
            mover = practice.EnemyScript(
                moving,
                size=4,
                loop=True,
                segments=(
                    practice.EnemySegment(tuple(target), 32),
                    practice.EnemySegment(moving, 32),
                ),
            )
            config = Config(
                name=f"{split}-{seed}",
                max_decisions=DECISIONS_PER_CONFIG,
                difficulty=1,
                permanent_pattern=0,
                invulnerable=True,
                player_start=start,
                player_script=tuple(
                    practice.PlayerCommand("action", 2, action=a) for a in actions
                ),
                enemies=(practice.EnemyScript(static, size=4), mover),
            )
            result.append((split, seed, config))
            index += 1
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_suite(
    output: Path | str,
    *,
    native_factory: Callable[..., object] | None = None,
    generator: Callable[..., dict[str, object]] | None = None,
    importer: Callable[..., dict[str, object]] | None = None,
) -> dict[str, object]:
    """Capture every config, then import explicit train/validation splits."""
    root = Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"practice suite output exists: {root}")
    root.mkdir(parents=True)
    generator = practice.generate_practice if generator is None else generator
    importer = import_practice_dataset if importer is None else importer
    entries: dict[str, list[dict[str, object]]] = {"train": [], "validation": []}
    captures: dict[str, list[Path]] = {"train": [], "validation": []}
    for split, seed, config in suite_configs():
        config_path = root / "configs" / f"{split}-{seed}.toml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(_toml(config), encoding="utf-8")
        capture = root / "captures" / split / f"{seed:05d}"
        kwargs = {} if native_factory is None else {"native_factory": native_factory}
        manifest = generator(config_path, capture, seed, **kwargs)
        captures[split].append(capture)
        entries[split].append(
            {
                "seed": seed,
                "capture": capture.relative_to(root).as_posix(),
                "config_path": config_path.relative_to(root).as_posix(),
                "config": config.to_dict(),
                "config_sha256": manifest["config_sha256"],
                "source_sha256": manifest.get("source_sha256") or _sha256(config_path),
                "manifest_sha256": _sha256(capture / "manifest.json"),
            }
        )
    corpus = root / "corpus"
    corpus_manifest = importer(corpus, captures["train"], captures["validation"])
    suite_manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_format": "pixel-repr-ddqn-practice-suite-v1",
        "seed": SUITE_SEED,
        "decisions_per_config": DECISIONS_PER_CONFIG,
        "planned_decisions": TOTAL_DECISIONS,
        "splits": entries,
        "corpus": {
            "path": "corpus",
            "dataset_id": corpus_manifest.get("dataset_id"),
            "manifest_sha256": _sha256(corpus / "manifest.json"),
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(suite_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return suite_manifest


__all__ = ["generate_suite", "suite_configs"]
