from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from dodge_native_game.variants.cnn_image_ddqn.pixel_replay import (
    NativePixelReplayBuffer,
)
from dodge_native_game.variants.cnn_image_ddqn.pixels import (
    GRAY_PROFILE,
    PICO8_LUMA_PALETTE,
)


@pytest.mark.parametrize("capacity", [1, 2, 7])
@pytest.mark.parametrize("stack", [1, 4, 8])
def test_gray_ring_matches_chronological_reference(capacity, stack):
    replay = NativePixelReplayBuffer(
        capacity, stack_size=stack, observation_profile=GRAY_PROFILE, seed=7
    )
    reference = deque(maxlen=capacity)
    observation = np.full((stack, 128, 128), PICO8_LUMA_PALETTE[3], np.uint8)
    for step in range(40):
        # Includes repeated luma across distinct source palette entries 3/5.
        color = (3, 5, 7, 7, 0)[step % 5]
        frame = np.full((1, 128, 128), PICO8_LUMA_PALETTE[color], np.uint8)
        following = np.concatenate((observation[1:], frame))
        done = step % 6 == 5
        replay.add(observation, step, float(step), following, done)
        reference.append((step, observation.copy(), following.copy(), done))
        batch = replay.sample(len(reference))
        by_action = {int(action): i for i, action in enumerate(batch.actions)}
        for action, before, after, terminal in reference:
            row = by_action[action]
            np.testing.assert_array_equal(batch.observations[row], before)
            np.testing.assert_array_equal(batch.next_observations[row], after)
            assert bool(batch.dones[row]) == terminal
        observation = following
        if done:
            observation = np.repeat(frame, stack, axis=0)
