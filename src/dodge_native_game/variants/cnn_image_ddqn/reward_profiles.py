"""Versioned weights only. Native Rust owns all event and geometry semantics."""

import numpy as np

COMPONENTS = ("survival", "death", "pickups", "enemy_destruction", "edge", "corner")
PROFILES = {
    "survival-v1": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "death-v1": (1.0, 25.0, 0.0, 0.0, 0.0, 0.0),
    "events-v1": (1.0, 0.0, 10.0, 1.0, 0.0, 0.0),
    "boundary-v1": (1.0, 0.0, 0.0, 0.0, 0.01, 0.1),
    "boundary5-v1": (1.0, 0.0, 0.0, 0.0, 0.05, 0.5),
    "boundary10-v1": (1.0, 0.0, 0.0, 0.0, 0.1, 1.0),
    "combined-v1": (1.0, 25.0, 10.0, 1.0, 0.01, 0.1),
}


def contract(profile):
    if profile not in PROFILES:
        raise ValueError(f"unknown native reward profile: {profile}")
    return {
        "profile": profile,
        "weights": dict(zip(COMPONENTS, PROFILES[profile], strict=True)),
        "native_terms_version": 1,
        "geometry": "bounds2:125-edge2-corner16-endpoint-times-frames",
        "enemy_destruction": "native shattered events, including indirect destruction",
    }


def weighted_components(terms, profile):
    values = np.asarray(terms, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("native reward terms missing, nonfinite, or wrong shape")
    return values * PROFILES[profile]
