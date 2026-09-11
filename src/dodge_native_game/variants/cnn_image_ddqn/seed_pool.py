"""Fixed, nested game-seed pools independent of learner initialization."""

from __future__ import annotations

import numpy as np


def training_seed_pool(size: int, pool_seed: int = 1729) -> tuple[int, ...]:
    """Select a shuffled prefix below 10,000, reserving higher seeds for eval.

    The 700-seed pool is a subset of the 5,000-seed pool. Cycling at natural
    episode boundaries makes visits balanced without shortening episodes.
    """
    if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= 10000:
        raise ValueError("training_seed_count must be an integer in [1, 10000]")
    return tuple(
        int(value)
        for value in np.random.default_rng(pool_seed).permutation(10000)[:size]
    )
