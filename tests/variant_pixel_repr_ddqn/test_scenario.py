from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from dodge_native_game.variants.pixel_repr_ddqn import collect as collector
from dodge_native_game.variants.pixel_repr_ddqn.collect import collect_dataset
from dodge_native_game.variants.pixel_repr_ddqn.dataset import (
    DatasetValidationError,
    PixelSequenceDataset,
)
from dodge_native_game.variants.pixel_repr_ddqn.native_adapter import (
    PixelNativeAdapter,
)
from dodge_native_game.variants.pixel_repr_ddqn.scenario import (
    ScenarioConfig,
    ScenarioConfigError,
    load_scenario,
    resolved_config_sha256,
    scenario_sha256,
)

SCENARIO_ROOT = Path(__file__).parents[2] / "variants/pixel-repr-ddqn/scenarios"


class _FakeEnvironment:
    def __init__(self) -> None:
        self.seed = 0
        self.step_count = 0
        self.closed = False

    def reset(self, *, seed: int):
        self.seed = seed
        self.step_count = 0
        return self._frame(), {}

    def step(self, action: int):
        self.step_count += 1
        return self._frame(action), float(action), False, False, {}

    def close(self) -> None:
        self.closed = True

    def _frame(self, action: int = 0) -> np.ndarray:
        value = (self.seed + self.step_count + action) % 256
        return np.full((3, 128, 128), value, dtype=np.uint8)


def test_v1_presets_resolve_to_expected_native_rules() -> None:
    expected = {
        "empty": (1, False, False, "none", 0, True),
        "permanent-patterns": (1, True, False, "none", 1, True),
        "normal-easy": (1, False, False, "normal", 0, True),
        "standard": (2, True, True, "all", 0, False),
    }
    for name, values in expected.items():
        config = load_scenario(SCENARIO_ROOT / f"{name}.toml")
        assert (
            config.difficulty,
            config.patterns_enabled,
            config.powerups_enabled,
            config.enemy_mode,
            config.permanent_pattern,
            config.invulnerable,
        ) == values
        assert config.to_dict()["version"] == 1
        assert len(config.resolved_sha256()) == 64


@pytest.mark.parametrize(
    "document",
    [
        "version = 2\n",
        "version = 1\nunknown = true\n",
        "version = 1\ndifficulty = 2.5\n",
        "version = 1\npermanent_pattern = 1\npatterns_enabled = false\n",
        "version = 1\nenemy_mode = \"special\"\n",
    ],
)
def test_scenario_schema_rejects_unsupported_documents(
    tmp_path: Path, document: str
) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(document)
    with pytest.raises(ScenarioConfigError):
        load_scenario(path)


def test_direct_dataclass_construction_remains_strict() -> None:
    with pytest.raises(ScenarioConfigError):
        ScenarioConfig(version=1.0)  # type: ignore[arg-type]
    with pytest.raises(ScenarioConfigError):
        ScenarioConfig(difficulty=2.0)  # type: ignore[arg-type]
    with pytest.raises(ScenarioConfigError):
        ScenarioConfig(enemy_mode=[])  # type: ignore[arg-type]


def test_adapter_uses_injected_native_environment_for_scenario(monkeypatch) -> None:
    import dodge_native_game.batch as batch
    import dodge_native_game.variants.pixel_repr_ddqn.native_adapter as adapter_module

    calls: dict[str, object] = {}

    class FakeNative:
        def __init__(self, **kwargs):
            calls["native"] = self
            calls["native_kwargs"] = kwargs

    class FakeGym:
        def __init__(self, **kwargs):
            calls["gym_kwargs"] = kwargs

    monkeypatch.setattr(batch, "NativeBatchEnvironment", FakeNative)
    monkeypatch.setattr(adapter_module, "CNNImageDDQNEnv", FakeGym)
    config = ScenarioConfig(
        difficulty=1,
        patterns_enabled=False,
        powerups_enabled=False,
        enemy_mode="none",
        invulnerable=True,
    )
    adapter = adapter_module.PixelNativeAdapter(scenario=config)
    assert adapter.scenario == config
    assert calls["native_kwargs"] == {
        "step_frames": 4,
        "full_state": False,
        "pixels": True,
        "board": False,
        "collision_image": False,
        **config.native_kwargs(),
    }
    assert calls["gym_kwargs"] == {
        "stack_size": 1,
        "step_frames": 4,
        "difficulty": 1,
        "patterns": False,
        "powerups": False,
        "observation_profile": "native-rgb-v1",
        "native_environment": calls["native"],
    }


def test_collection_persists_and_validates_scenario_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario_path = SCENARIO_ROOT / "normal-easy.toml"
    monkeypatch.setattr(
        collector,
        "PixelNativeAdapter",
        lambda **_kwargs: PixelNativeAdapter(_FakeEnvironment()),
    )
    root = tmp_path / "dataset"
    manifest = collect_dataset(
        root,
        train_seeds=[0],
        validation_seeds=[1],
        max_steps_per_episode=2,
        seed=7,
        scenario=scenario_path,
    )
    provenance = manifest["scenario"]
    config = load_scenario(scenario_path)
    assert provenance["config"] == config.to_dict()
    assert provenance["sha256"] == resolved_config_sha256(config)
    assert provenance["source_sha256"] == scenario_sha256(scenario_path)
    assert provenance["source"] == scenario_path.as_posix()
    assert PixelSequenceDataset(root, split="train", history_size=1)

    tampered = json.loads((root / "manifest.json").read_text())
    tampered["scenario"]["config"]["difficulty"] = 3
    (root / "manifest.json").write_text(json.dumps(tampered))
    with pytest.raises(DatasetValidationError, match="scenario"):
        PixelSequenceDataset(root)


def test_resolved_hash_is_stable_and_distinct_from_toml_bytes() -> None:
    config = ScenarioConfig.standard()
    assert resolved_config_sha256(config) == hashlib.sha256(
        (
            json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode()
    ).hexdigest()
