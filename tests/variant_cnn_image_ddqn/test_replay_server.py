from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

from dodge_native_game.variants.cnn_image_ddqn.native_replay import (
    NativeReplay,
    ReplayFrame,
)
from dodge_native_game.variants.cnn_image_ddqn.replay_server import (
    ReplayServer,
    ReplayStore,
)


def _replay() -> NativeReplay:
    frame = ReplayFrame(
        index=0,
        native_frame=0,
        action=None,
        reward=0.0,
        done=False,
        png=b"\x89PNG\r\n\x1a\n",
        collision_png=b"\x89PNG\r\n\x1a\n",
    )
    return NativeReplay(
        checkpoint=Path("step-32.pt"),
        seed=42,
        step_frames=4,
        created_at="2026-09-10T00:00:00Z",
        frames=(frame,),
    )


def test_replay_server_serves_page_metadata_and_native_frame() -> None:
    store = ReplayStore()
    server = ReplayServer(store=store, host="127.0.0.1", port=0)
    server.start()
    token = store.add(_replay())
    try:
        with urlopen(server.url_for(token)) as response:
            page = response.read()
        with urlopen(f"{server.base_url}/api/replay/{token}") as response:
            metadata = json.loads(response.read())
        with urlopen(f"{server.base_url}/api/replay/{token}/frame/0.png") as response:
            frame = response.read()
        with urlopen(
            f"{server.base_url}/api/replay/{token}/collision/0.png"
        ) as response:
            collision = response.read()
    finally:
        server.close()

    assert b"Dodge Native Replay" in page
    assert metadata["frame_count"] == 1
    assert metadata["frames"][0]["native_frame"] == 0
    assert metadata["frames"][0]["collision_url"].endswith("/collision/0.png")
    assert frame == b"\x89PNG\r\n\x1a\n"
    assert collision == b"\x89PNG\r\n\x1a\n"
