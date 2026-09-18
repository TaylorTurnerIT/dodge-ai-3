from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dodge_native_game.variants.pixel_repr_ddqn import mpc_eval
from dodge_native_game.variants.pixel_repr_ddqn.survival_probe import SurvivalProbe


def _write_episode(root: Path, split: str, index: int, die_at: int | None) -> dict:
    count = 128
    path = root / "episodes" / split / f"episode-{index:06d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.zeros((count + 1, 3, 128, 128), dtype=np.uint8)
    frames[:, 0] = (index * 40 + (80 if split == "validation" else 0)) % 256
    actions = np.zeros(count, dtype=np.int64)
    terminated = np.zeros(count, dtype=np.bool_)
    if die_at is not None:
        terminated[die_at] = True
    truncated = np.zeros(count, dtype=np.bool_)
    truncated[-1] = True
    with path.open("wb") as stream:
        np.savez_compressed(
            stream, pixels=frames, actions=actions,
            terminated=terminated, truncated=truncated,
        )
    return {
        "episode_id": f"{split}-{index:06d}",
        "recipe_family": "test",
        "seed": 22000 + index,
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "decisions": count,
        "terminated": die_at is not None,
    }


def _make_dataset(root: Path) -> None:
    from dodge_native_game.variants.pixel_repr_ddqn.survival_probe import (
        PROBE_FORMAT,
    )

    episodes = {"train": [], "validation": []}
    for split, count in (("train", 2), ("validation", 1)):
        for index in range(count):
            episodes[split].append(
                _write_episode(root, split, index, die_at=127 if index == 0 else None)
            )
    manifest = {
        "format": PROBE_FORMAT,
        "splits": {split: len(rows) for split, rows in episodes.items()},
        "deaths": {
            split: sum(1 for row in rows if row["terminated"])
            for split, rows in episodes.items()
        },
        "episodes": episodes,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n")
    (root / "READY").write_text(PROBE_FORMAT + "\n")


def test_probe_fits_separable_latents() -> None:
    torch.manual_seed(0)
    probe = SurvivalProbe()
    alive = torch.randn(64, 192) - 2.0
    dead = torch.randn(64, 192) + 2.0
    features = torch.cat([alive, dead])
    labels = torch.cat([torch.zeros(64), torch.ones(64)])
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        before = float(criterion(probe(features), labels))
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(probe(features), labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        after = float(criterion(probe(features), labels))
    assert after < before / 4


def test_windows_label_death_within_horizon(tmp_path: Path) -> None:
    from dodge_native_game.variants.pixel_repr_ddqn import survival_probe as sp

    root = tmp_path / "dataset"
    _make_dataset(root)
    pixels, labels, provenance = sp._windows_with_labels(
        root, "train", max_windows=10000, seed=0
    )
    assert len(labels) == 2 * 125
    assert set(np.unique(labels)) == {0.0, 1.0}
    # Episode 0 dies at decision 64: windows ending within 16 steps flag it.
    first = [label for tag, label in zip(provenance, labels, strict=True)
             if tag.startswith("train-000000")]
    assert sum(first) == 16
    assert all(label == 0 for tag, label in zip(provenance, labels, strict=True)
               if tag.startswith("train-000001"))


class _StubModel(torch.nn.Module):
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        batch, time = pixels.shape[:2]
        return torch.zeros(batch, time, 192)

    def predict(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        out = z.clone()
        out[:, :, 0] = actions.float()
        return out


class _StubProbe(torch.nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, 0] - 4.0


class _ScriptedAdapter:
    def __init__(self, die_at: int | None = None) -> None:
        self._die_at = die_at
        self._step = 0
        self.actions: list[int] = []

    def reset(self, seed: int):
        del seed
        self._step = 0
        return np.zeros((3, 128, 128), dtype=np.uint8)

    def step(self, action: int):
        self.actions.append(int(action))
        if self._die_at is not None and self._step == self._die_at:
            return np.zeros((3, 128, 128), dtype=np.uint8), 0.0, True, False
        self._step += 1
        return np.zeros((3, 128, 128), dtype=np.uint8), 0.0, False, False

    def close(self) -> None:
        pass


def test_average_precision_null_when_degenerate() -> None:
    import json

    from dodge_native_game.variants.pixel_repr_ddqn.survival_probe import (
        _average_precision,
    )

    # No positives (or all positives): JSON null, never NaN.
    assert _average_precision(torch.zeros(8), torch.zeros(8)) is None
    assert _average_precision(torch.zeros(8), torch.ones(8)) is None
    perfect = _average_precision(
        torch.tensor([0.1, 0.2, 0.8, 0.9]), torch.tensor([0.0, 0.0, 1.0, 1.0])
    )
    assert perfect is not None and perfect > 0.99
    report = {"val_auprc": None}
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_greedy_action_picks_lowest_predicted_cost() -> None:
    action, costs = mpc_eval._greedy_action(
        _StubModel(), _StubProbe(),
        torch.zeros(1, 3, 3, 128, 128), [0, 0, 0],
        torch.device("cpu"),
    )
    assert costs[0] == pytest.approx(-4.0)
    assert costs[5] == pytest.approx(1.0)
    assert action == 0


def test_run_episode_counts_survived_and_outcome() -> None:
    model = _StubModel()
    device = torch.device("cpu")
    died = mpc_eval.run_episode(
        lambda: _ScriptedAdapter(die_at=10), 0,
        lambda history, past: 0,
        model=model, device=device, max_decisions=128,
    )
    assert died == {"seed": 0, "survived": 10, "outcome": "terminated"}
    lived = mpc_eval.run_episode(
        lambda: _ScriptedAdapter(), 0,
        lambda history, past: 0,
        model=model, device=device, max_decisions=32,
    )
    assert lived == {"seed": 0, "survived": 32, "outcome": "truncated"}


def test_evaluate_rejects_probe_world_mismatch(tmp_path: Path) -> None:
    world = tmp_path / "world.pt"
    torch.save({"config": {}}, world)
    probe_path = tmp_path / "probe.pt"
    torch.save({"model": SurvivalProbe().state_dict(),
                "world_model_sha256": "b" * 64}, probe_path)
    with pytest.raises(ValueError, match="different world checkpoint"):
        mpc_eval.evaluate(
            world, probe_path, [], policies={"mpc": "mpc"}, device="cpu",
        )


def test_plan_mpc_eval_splits_and_seeds() -> None:
    scenarios = mpc_eval.plan_mpc_eval()
    labels = [label for label, _, _ in scenarios]
    assert len(scenarios) == 32
    assert sum(1 for label in labels if label.startswith("ordinary-")) == 16
    assert sum(1 for label in labels if label.startswith("novel-")) == 16
    seeds = [seed for _, _, seed in scenarios]
    assert len(set(seeds)) == 32
    assert min(seeds) >= 24000
    # Outside every training/validation/probe seed range.
    assert max(seeds) <= 32767
    cfgs = [cfg for _, cfg, _ in scenarios]
    assert all(cfg.invulnerable is False for cfg in cfgs)
