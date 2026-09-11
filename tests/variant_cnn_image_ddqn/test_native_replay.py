from __future__ import annotations

import struct

import numpy as np

from dodge_native_game.variants.cnn_image_ddqn.native_replay import (
    PICO8_PALETTE,
    NativeReplay,
    grayscale_png,
    rgb_png,
)
from dodge_native_game.variants.cnn_image_ddqn.replay import ReplayBuffer


def test_grayscale_png_is_an_84_by_84_png() -> None:
    image = np.zeros((84, 84), dtype=np.uint8)
    image[12, 34] = 255

    payload = grayscale_png(image)

    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert payload[12:16] == b"IHDR"
    assert struct.unpack(">II", payload[16:24]) == (84, 84)
    assert payload.endswith(b"IEND\xaeB`\x82")


def test_rgb_png_maps_pico8_indices_to_truecolor() -> None:
    assert len(PICO8_PALETTE) == 16
    image = np.zeros((128, 128), dtype=np.uint8)
    image[0, 0] = 12

    payload = rgb_png(image)

    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert payload[12:16] == b"IHDR"
    assert struct.unpack(">II", payload[16:24]) == (128, 128)
    # Color type 2 (truecolor): index 12 is PICO-8 light blue.
    assert payload[25:26] == b"\x02"
    import zlib

    at = payload.index(b"IDAT")
    length = struct.unpack(">I", payload[at - 4 : at])[0]
    raw = zlib.decompress(payload[at + 4 : at + 4 + length])
    assert raw[1:4] == bytes((41, 173, 255))
    assert payload.endswith(b"IEND\xaeB`\x82")


def test_v26_training_and_browser_replay_surfaces_stay_separate() -> None:
    assert ReplayBuffer.__module__.endswith(".replay")
    assert NativeReplay.__module__.endswith(".native_replay")
