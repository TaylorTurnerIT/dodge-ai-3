"""Strict, versioned native-game scenario configuration.

Scenario metadata controls only the native test environment.  It is resolved
before collection and never enters the pixel/action learner boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCENARIO_VERSION = 1
ENEMY_MODES = ("all", "none", "normal")
DIFFICULTY_NAMES = {"easy": 1, "medium": 2, "hard": 3}
DIFFICULTIES = frozenset(DIFFICULTY_NAMES.values())
SCENARIO_KEYS = frozenset(
    {
        "version",
        "name",
        "difficulty",
        "patterns_enabled",
        "powerups_enabled",
        "enemy_mode",
        "permanent_pattern",
        "invulnerable",
    }
)
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class ScenarioConfigError(ValueError):
    """Raised when a scenario TOML document is not a supported v1 config."""


def _strict_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ScenarioConfigError(f"{field} must be a boolean")
    return value


def _strict_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScenarioConfigError(f"{field} must be an integer")
    return value


def _difficulty(value: object) -> int:
    if isinstance(value, str):
        try:
            return DIFFICULTY_NAMES[value]
        except KeyError as error:
            choices = ", ".join(DIFFICULTY_NAMES)
            raise ScenarioConfigError(
                f"difficulty must be one of {choices} or an integer 1..3"
            ) from error
    result = _strict_int(value, "difficulty")
    if result not in DIFFICULTIES:
        raise ScenarioConfigError("difficulty must be one of 1, 2, or 3")
    return result


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """Resolved native configuration for one pixel/action collection run."""

    version: int = SCENARIO_VERSION
    name: str = "standard"
    difficulty: int = 2
    patterns_enabled: bool = True
    powerups_enabled: bool = True
    enemy_mode: str = "all"
    permanent_pattern: int = 0
    invulnerable: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, int)
            or self.version != SCENARIO_VERSION
        ):
            raise ScenarioConfigError(
                f"unsupported scenario version {self.version!r}; expected 1"
            )
        if (
            isinstance(self.name, bool)
            or not isinstance(self.name, str)
            or not _NAME_RE.fullmatch(self.name)
        ):
            raise ScenarioConfigError(
                "name must be 1..64 ASCII letters, digits, '_', '-', or '.'"
            )
        if (
            isinstance(self.difficulty, bool)
            or not isinstance(self.difficulty, int)
            or self.difficulty not in DIFFICULTIES
        ):
            raise ScenarioConfigError("difficulty must be one of 1, 2, or 3")
        if not isinstance(self.enemy_mode, str) or self.enemy_mode not in ENEMY_MODES:
            raise ScenarioConfigError(
                "enemy_mode must be one of 'all', 'none', or 'normal'"
            )
        if not isinstance(self.patterns_enabled, bool):
            raise ScenarioConfigError("patterns_enabled must be a boolean")
        if not isinstance(self.powerups_enabled, bool):
            raise ScenarioConfigError("powerups_enabled must be a boolean")
        if not isinstance(self.invulnerable, bool):
            raise ScenarioConfigError("invulnerable must be a boolean")
        if (
            isinstance(self.permanent_pattern, bool)
            or not isinstance(self.permanent_pattern, int)
        ):
            raise ScenarioConfigError("permanent_pattern must be an integer")
        if not 0 <= self.permanent_pattern <= 39:
            raise ScenarioConfigError("permanent_pattern must be between 0 and 39")
        if self.permanent_pattern and not self.patterns_enabled:
            raise ScenarioConfigError(
                "permanent_pattern requires patterns_enabled=true"
            )

    @classmethod
    def standard(cls) -> ScenarioConfig:
        """Return the ordinary native trajectory configuration."""

        return cls()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ScenarioConfig:
        """Resolve one strict v1 mapping into a validated dataclass."""

        if not isinstance(raw, Mapping):
            raise ScenarioConfigError("scenario TOML must contain a table")
        unknown = sorted(set(raw) - SCENARIO_KEYS)
        if unknown:
            names = ", ".join(repr(key) for key in unknown)
            raise ScenarioConfigError(f"unknown scenario key(s): {names}")
        if "version" not in raw:
            raise ScenarioConfigError("scenario version is required")
        version = _strict_int(raw["version"], "version")
        if version != SCENARIO_VERSION:
            raise ScenarioConfigError(
                f"unsupported scenario version {version!r}; expected 1"
            )

        name = raw.get("name", "custom")
        if isinstance(name, bool) or not isinstance(name, str):
            raise ScenarioConfigError("name must be a string")
        return cls(
            version=version,
            name=name,
            difficulty=_difficulty(raw.get("difficulty", 2)),
            patterns_enabled=_strict_bool(
                raw.get("patterns_enabled", True), "patterns_enabled"
            ),
            powerups_enabled=_strict_bool(
                raw.get("powerups_enabled", True), "powerups_enabled"
            ),
            enemy_mode=raw.get("enemy_mode", "all"),
            permanent_pattern=_strict_int(
                raw.get("permanent_pattern", 0), "permanent_pattern"
            ),
            invulnerable=_strict_bool(
                raw.get("invulnerable", False), "invulnerable"
            ),
        )

    @property
    def difficulty_name(self) -> str:
        """Human-readable difficulty label for reports."""

        return next(
            name for name, value in DIFFICULTY_NAMES.items() if value == self.difficulty
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the fully resolved, JSON/TOML-independent config mapping."""

        return {
            "version": self.version,
            "name": self.name,
            "difficulty": self.difficulty,
            "patterns_enabled": self.patterns_enabled,
            "powerups_enabled": self.powerups_enabled,
            "enemy_mode": self.enemy_mode,
            "permanent_pattern": self.permanent_pattern,
            "invulnerable": self.invulnerable,
        }

    def native_kwargs(self) -> dict[str, Any]:
        """Return only constructor fields understood by NativeBatchEnvironment."""

        return {
            "difficulty": self.difficulty,
            "patterns_enabled": self.patterns_enabled,
            "powerups_enabled": self.powerups_enabled,
            "enemy_mode": self.enemy_mode,
            "permanent_pattern": self.permanent_pattern,
            "invulnerable": self.invulnerable,
        }

    def resolved_sha256(self) -> str:
        """Hash the canonical resolved config used by artifact validation."""

        return hashlib.sha256(_canonical_json(self.to_dict())).hexdigest()


def scenario_sha256(path: Path) -> str:
    """Hash the exact TOML bytes supplied by a caller."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolved_config_sha256(config: ScenarioConfig) -> str:
    """Return the canonical hash for a resolved scenario mapping."""

    if not isinstance(config, ScenarioConfig):
        raise TypeError("config must be a ScenarioConfig")
    return config.resolved_sha256()


def load_scenario(path: Path) -> ScenarioConfig:
    """Parse and resolve a strict version1 scenario TOML file."""

    source = Path(path)
    try:
        raw = tomllib.loads(source.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ScenarioConfigError(f"invalid scenario TOML: {source}") from error
    except (OSError, UnicodeError):
        raise
    return ScenarioConfig.from_mapping(raw)


def scenario_provenance(
    path: Path | None = None,
    config: ScenarioConfig | None = None,
) -> dict[str, Any]:
    """Return source and resolved scenario identity for an artifact manifest."""

    if path is not None:
        source = Path(path)
        resolved = config or load_scenario(source)
        source_hash = scenario_sha256(source)
        source_name = source.as_posix()
    else:
        resolved = config or ScenarioConfig.standard()
        source_hash = None
        source_name = (
            "builtin:standard"
            if resolved == ScenarioConfig.standard()
            else f"inline:{resolved.name}"
        )
    resolved_hash = resolved.resolved_sha256()
    # The resolved hash is the verifiable identity retained by the dataset and
    # run manifests.  The source hash pins the exact TOML bytes when a file was
    # supplied, but cannot be recomputed after only the resolved config remains.
    return {
        "source": source_name,
        "sha256": resolved_hash,
        "source_sha256": source_hash,
        "config": resolved.to_dict(),
    }


def resolve_scenario(
    scenario: Path | str | ScenarioConfig | None,
) -> tuple[ScenarioConfig, dict[str, Any]]:
    """Resolve a path/config/omitted scenario and its artifact provenance."""

    if scenario is None:
        config = ScenarioConfig.standard()
        return config, scenario_provenance(config=config)
    if isinstance(scenario, ScenarioConfig):
        return scenario, scenario_provenance(config=scenario)
    path = Path(scenario)
    config = load_scenario(path)
    return config, scenario_provenance(path, config)


__all__ = [
    "DIFFICULTIES",
    "DIFFICULTY_NAMES",
    "ENEMY_MODES",
    "SCENARIO_KEYS",
    "SCENARIO_VERSION",
    "ScenarioConfig",
    "ScenarioConfigError",
    "load_scenario",
    "resolve_scenario",
    "resolved_config_sha256",
    "scenario_provenance",
    "scenario_sha256",
]
