from __future__ import annotations

import json

import pytest

from dodge_native_game.variants.cnn_image_ddqn.run import train_run
from dodge_native_game.variants.cnn_image_ddqn.seed_pool import training_seed_pool

from .test_diagnostics import _factory


def test_pools_are_nested_unique_and_reserved_from_evaluation() -> None:
    small = training_seed_pool(700)
    large = training_seed_pool(5000)
    assert small == large[:700]
    assert len(set(large)) == 5000
    assert min(large) >= 0 and max(large) < 10000
    assert large == training_seed_pool(5000)
    with pytest.raises(ValueError):
        training_seed_pool(10001)


def test_training_cycles_pool_and_counts_only_visited_steps(tmp_path) -> None:
    root = train_run(
        history_root=tmp_path,
        run_id="pool",
        steps=6,
        stack_size=1,
        training_seed_count=2,
        evaluation_seed=42,
        batch_size=2,
        warmup_steps=2,
        update_every=2,
        log_interval=4,
        eval_episodes=1,
        eval_steps=2,
        device="cpu",
        env_factory=_factory,
        checkpoint_steps=(2, 4),
    )
    manifest = json.loads((root / "manifest.json").read_text())
    pool = manifest["training_seed_protocol"]["seeds"]
    rows = [
        json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()
    ]
    episodes = [episode for row in rows for episode in row["completed_episodes"]]
    assert [episode["game_seed"] for episode in episodes] == [pool[0], pool[1], pool[0]]
    report = json.loads((root / "report.json").read_text())
    assert report["diagnostics"]["training_seed_steps"] == {
        str(pool[0]): 4,
        str(pool[1]): 2,
    }
    assert report["diagnostics"]["training_seed_pool_coverage"] == 1.0
    assert report["evaluation"]["inner"]["seeds"] == [10042]
    assert (root / "checkpoints/step-2.pt").is_file()
    assert (root / "checkpoints/step-4.pt").is_file()
    assert (root / "checkpoints/step-6.pt").is_file()
