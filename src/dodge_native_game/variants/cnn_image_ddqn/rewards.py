"""Tunable reward weights, shared by the dashboard and the environments.

Training environments run in separate processes, so a slider in the dashboard
cannot reach them through memory. A file is the source of truth instead: the
dashboard writes it, a text editor can write it just as well, and every
environment re-reads it when an episode starts. Changes therefore take effect
from the next episode in every process, with no message passing.
"""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

CONFIG_PATH = Path("rewards.json")


@dataclass
class RewardConfig:
    survival_per_frame: float = 0.01
    # Survival credit lost at the wall. The interior is flat, so this only
    # discourages hugging an edge; it does not pay the agent to sit still on
    # the centre pixel, which the previous centre-peaked shape did.
    edge_penalty: float = 0.5
    death_penalty: float = 1.0
    score_weight: float = 1.0
    # The cart credits 0.5 whenever an enemy dies to another enemy or to a
    # pattern. The agent does not cause those, so at 0.0 that score is stripped
    # out and only agent-caused score is paid. 1.0 restores the cart's own
    # scoring, at the cost of rewarding luck.
    uncontrolled_score_weight: float = 0.0

    def clamped(self):
        values = asdict(self)
        for spec in SLIDERS:
            values[spec.name] = min(
                spec.maximum, max(spec.minimum, float(values[spec.name]))
            )
        return RewardConfig(**values)


@dataclass(frozen=True)
class SliderSpec:
    name: str
    label: str
    minimum: float
    maximum: float
    step: float
    hint: str


# One definition drives both the dashboard's sliders and the documented range
# for hand-edited files.
SLIDERS = (
    SliderSpec(
        "survival_per_frame",
        "Survival / frame",
        0.0,
        0.05,
        0.001,
        "Paid for every frame survived",
    ),
    SliderSpec(
        "edge_penalty",
        "Edge penalty",
        0.0,
        1.0,
        0.05,
        "Survival credit lost against a wall",
    ),
    SliderSpec(
        "death_penalty",
        "Death penalty",
        0.0,
        5.0,
        0.1,
        "Charged once, on the frame the agent dies",
    ),
    SliderSpec(
        "score_weight",
        "Score weight",
        0.0,
        2.0,
        0.05,
        "Multiplies score the agent caused, such as pickups",
    ),
    SliderSpec(
        "uncontrolled_score_weight",
        "Luck score weight",
        0.0,
        1.0,
        0.05,
        "Score from enemies colliding with each other",
    ),
)

UNCONTROLLED_SCORE_PER_ENEMY = 0.5

# Weights that changed name. Both were "how much position matters", on the same
# 0..1 scale, so an existing file carries its value over rather than silently
# reverting to a default.
RENAMED = {"center_bonus": "edge_penalty"}


def load(path=CONFIG_PATH):
    """Read the config, falling back to defaults for anything missing.

    A malformed or absent file yields defaults rather than raising: training
    must not die because a hand edit left a trailing comma.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return RewardConfig()
    if not isinstance(raw, dict):
        return RewardConfig()
    known = {field.name for field in fields(RewardConfig)}
    values = {RENAMED.get(key, key): value for key, value in raw.items()}
    values = {
        key: value
        for key, value in values.items()
        if key in known and isinstance(value, (int, float))
    }
    return RewardConfig(**values).clamped()


def save(config, path=CONFIG_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
