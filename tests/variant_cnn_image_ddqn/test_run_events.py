from __future__ import annotations

import json

import numpy as np

from dodge_native_game.variants.cnn_image_ddqn import CNNImageDDQNEnv
from dodge_native_game.variants.cnn_image_ddqn.run import train_run

DEATH = 1 << 2
SPAWN = 1 << 0


class FlagBatchResult:
    def __init__(self, *, done: bool, flags: int) -> None:
        image = np.full((1, 84, 84), 7, dtype=np.uint8)
        self.collision_image = image
        self.rewards = np.asarray([1.0], dtype=np.float32)
        self.done = np.asarray([done], dtype=np.bool_)
        self.frames = np.asarray([4], dtype=np.uint32)
        self.frames_advanced = np.asarray([4], dtype=np.uint32)
        self.seeds = np.asarray([3], dtype=np.uint32)
        self.event_flags = np.asarray([flags], dtype=np.uint32)
        self.modes = np.asarray([2], dtype=np.uint8)


class FlagNativeBatch:
    def __init__(self) -> None:
        self._steps = 0
        self.closed = False

    def reset_batch(self, seeds: object, *, startup: bool = False) -> FlagBatchResult:
        del seeds, startup
        self._steps = 0
        return FlagBatchResult(done=False, flags=SPAWN)

    def step_batch(self, actions: object) -> FlagBatchResult:
        del actions
        self._steps += 1
        if self._steps == 2:
            return FlagBatchResult(done=True, flags=DEATH | SPAWN)
        return FlagBatchResult(done=False, flags=SPAWN)

    def close(self) -> None:
        self.closed = True


def _factory(**kwargs) -> CNNImageDDQNEnv:
    return CNNImageDDQNEnv(
        stack_size=kwargs.get("stack_size", 1),
        step_frames=kwargs.get("step_frames", 4),
        native_environment=FlagNativeBatch(),
    )


def test_info_exposes_native_event_flags_and_mode() -> None:
    env = _factory()
    try:
        _, info = env.reset(seed=3)
        assert info["native_event_flags"] == SPAWN
        assert info["native_mode"] == 2
    finally:
        env.close()


def test_train_run_records_native_deaths_and_spawns(tmp_path) -> None:
    root = train_run(
        history_root=tmp_path,
        run_id="event-smoke",
        steps=4,
        stack_size=1,
        batch_size=2,
        warmup_steps=2,
        update_every=2,
        log_interval=1,
        eval_episodes=1,
        eval_steps=2,
        device="cpu",
        env_factory=_factory,  # type: ignore[arg-type]
    )
    rows = [
        json.loads(line)
        for line in (root / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows, "expected metric records"
    last = rows[-1]
    assert last["total_deaths"] >= 1
    assert last["total_spawns"] >= 1
    assert last["deaths"] >= 0
    assert "enemies_killed" in last
    assert "total_kills" in last
