from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from dodge_native_game.variants.pixel_repr_ddqn.practice import (
    ACTION_INDEX,
    PracticeConfigError,
    generate_practice,
    load_practice,
)

ROOT = Path(__file__).parents[2]
SAMPLE = ROOT / "variants/pixel-repr-ddqn/practice/moving-enemy.toml"


def _document(**updates: object) -> str:
    values: dict[str, object] = {
        "version": 1,
        "name": "fixture",
        "step_frames": 4,
        "max_decisions": 4,
        "difficulty": 1,
        "permanent_pattern": 0,
        "invulnerable": True,
        "player_start": [32.0, 48.0],
        "player_script": [{"action": "neutral", "decisions": 4}],
        "enemies": [],
    }
    values.update(updates)
    lines = [
        f"version = {values['version']}",
        f"name = {values['name']!r}",
        f"step_frames = {values['step_frames']}",
        f"max_decisions = {values['max_decisions']}",
        f"difficulty = {values['difficulty']}",
        f"permanent_pattern = {values['permanent_pattern']}",
        f"invulnerable = {str(values['invulnerable']).lower()}",
        f"player_start = {values['player_start']!r}".replace("'", ""),
    ]
    for command in values["player_script"]:  # type: ignore[union-attr]
        lines.append("\n[[player_script]]")
        if "action" in command:
            lines.append(f"action = {command['action']!r}")
            lines.append(f"decisions = {command['decisions']}")
        else:
            lines.append(f"move_to = {command['move_to']!r}".replace("'", ""))
            lines.append(f"max_decisions = {command['max_decisions']}")
    for enemy in values["enemies"]:  # type: ignore[union-attr]
        lines.extend(
            [
                "\n[[enemies]]",
                f"start = {enemy['start']!r}".replace("'", ""),
                f"size = {enemy.get('size', 4)}",
                f"loop = {str(enemy.get('loop', False)).lower()}",
                "segments = []",
            ]
        )
    return "\n".join(lines) + "\n"


class FakePracticeEnv:
    instances: list[FakePracticeEnv] = []
    outcome = "finished"

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.step_count = 0
        self.closed = False
        type(self).instances.append(self)

    def reset(self, seed: int) -> np.ndarray:
        self.seed = seed
        self.step_count = 0
        return np.full((128, 128), 1, dtype=np.uint8)

    def step(self) -> dict[str, object]:
        self.step_count += 1
        stop = self.step_count >= 3
        return {
            "pixels": np.full((128, 128), self.step_count + 1, dtype=np.uint8),
            "action": ACTION_INDEX["right"],
            "terminated": self.outcome == "terminated" and stop,
            "finished": self.outcome == "finished" and stop,
            "failed": self.outcome == "failed" and stop,
        }

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", 2),
        ("step_frames", 8),
        ("max_decisions", 257),
        ("difficulty", 4),
        ("permanent_pattern", 40),
        ("player_start", [3.0, 48.0]),
    ],
)
def test_schema_rejects_boundary_values(
    tmp_path: Path, field: str, value: object
) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(_document(**{field: value}), encoding="utf-8")
    with pytest.raises(PracticeConfigError):
        load_practice(path)


def test_schema_rejects_unknown_keys_and_open_enemy_loops(tmp_path: Path) -> None:
    unknown = tmp_path / "unknown.toml"
    unknown.write_text(_document() + "extra = true\n", encoding="utf-8")
    with pytest.raises(PracticeConfigError):
        load_practice(unknown)

    loop = tmp_path / "loop.toml"
    loop.write_text(
        _document(
            enemies=[
                {
                    "start": [20.0, 20.0],
                    "size": 4,
                    "loop": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(PracticeConfigError, match="end at their start"):
        load_practice(loop)


def test_sample_resolves_both_scripted_actor_types() -> None:
    config = load_practice(SAMPLE)
    assert config.native_commands() == [
        (1, 92.0, 78.0, 48),
        (0, 0, 0, 4),
    ]
    assert config.native_enemies() == [
        (64.0, 32.0, 4, True, [(96.0, 32.0, 32), (64.0, 32.0, 32)]),
        (24.0, 24.0, 4, False, []),
    ]


def test_generator_writes_rgb_episode_and_valid_goal(tmp_path: Path) -> None:
    FakePracticeEnv.instances.clear()
    FakePracticeEnv.outcome = "finished"
    config_path = tmp_path / "practice.toml"
    config_path.write_text(_document(), encoding="utf-8")
    output = tmp_path / "artifact"
    manifest = generate_practice(config_path, output, 7, native_factory=FakePracticeEnv)

    assert manifest["success"] is True
    assert manifest["goal"]["valid"] is True
    assert manifest["action_trace"] == [ACTION_INDEX["right"]] * 3
    assert FakePracticeEnv.instances[0].kwargs["step_frames"] == 4
    assert FakePracticeEnv.instances[0].kwargs["commands"] == [(0, 0, 0, 4)]
    assert FakePracticeEnv.instances[0].closed is True
    assert manifest["truncated"] is True
    assert manifest["goal"]["frame"] == "final.png"
    assert manifest["goal"]["clip"] == "goal.gif"

    with np.load(output / "episode.npz") as episode:
        assert episode["pixels"].shape == (4, 3, 128, 128)
        assert episode["actions"].shape == (3,)
        assert episode["terminated"].tolist() == [False, False, False]
        assert episode["truncated"].tolist() == [False, False, True]
    with np.load(output / "final_clip.npz") as clip:
        assert clip["pixels"].shape == (4, 3, 128, 128)
        assert clip["actions"].shape == (3,)
    assert np.load(output / "initial.npz")["pixels"].shape == (3, 128, 128)
    for record in manifest["files"].values():
        path = output / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    assert (
        manifest["source_sha256"]
        == hashlib.sha256(config_path.read_bytes()).hexdigest()
    )
    assert (
        json.loads((output / "manifest.json").read_text())["config"]
        == manifest["config"]
    )


@pytest.mark.parametrize(
    "outcome, failure", [("failed", "timeout"), ("terminated", "terminated")]
)
def test_failed_or_dead_run_never_marks_goal(
    tmp_path: Path, outcome: str, failure: str
) -> None:
    FakePracticeEnv.instances.clear()
    FakePracticeEnv.outcome = outcome
    path = tmp_path / "practice.toml"
    path.write_text(_document(), encoding="utf-8")
    manifest = generate_practice(
        path, tmp_path / outcome, 0, native_factory=FakePracticeEnv
    )
    assert manifest["success"] is False
    assert manifest["failure"] == failure
    assert manifest["goal"]["valid"] is False
    assert manifest["goal"]["frame"] is None
    assert manifest["goal"]["clip"] is None


def test_max_decisions_is_recorded_as_truncation(tmp_path: Path) -> None:
    FakePracticeEnv.instances.clear()
    FakePracticeEnv.outcome = "never"
    path = tmp_path / "practice.toml"
    path.write_text(
        _document(
            max_decisions=2,
            player_script=[{"action": "neutral", "decisions": 2}],
        ),
        encoding="utf-8",
    )
    manifest = generate_practice(
        path, tmp_path / "timeout", 0, native_factory=FakePracticeEnv
    )
    assert manifest["success"] is False
    assert manifest["failure"] == "timeout"
    with np.load(tmp_path / "timeout" / "episode.npz") as episode:
        assert episode["truncated"].tolist() == [False, True]


def test_direct_config_freezes_inputs_and_rejects_bool_actions():
    from dodge_native_game.variants.pixel_repr_ddqn.practice import (
        PlayerCommand,
        PracticeConfig,
    )

    with pytest.raises(PracticeConfigError):
        PlayerCommand("action", 1, action=True)
    point = [32.0, 48.0]
    commands = [PlayerCommand("action", 1, action=0)]
    config = PracticeConfig(player_start=point, player_script=commands)
    before = config.to_dict()
    point[0] = 90.0
    commands.clear()
    assert config.to_dict() == before
