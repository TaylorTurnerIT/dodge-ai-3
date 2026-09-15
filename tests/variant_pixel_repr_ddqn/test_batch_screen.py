from pathlib import Path

import pytest
import torch

from dodge_native_game.variants.pixel_repr_ddqn.pretrain import (
    PRACTICE_BATCH32_EXPERIMENT,
    batch_from_dataset,
    save_experiment_checkpoint,
    train,
)


@pytest.mark.parametrize(
    "override",
    [
        {"steps": 511},
        {"batch_size": 8},
        {"seed": 43},
        {"profile": "tiny"},
        {"device": "cpu"},
        {"resume": Path("checkpoint.pt")},
    ],
)
def test_batch32_screen_envelope_cannot_silently_change(override, tmp_path: Path):
    args = {
        "dataset_root": tmp_path / "missing",
        "history_root": tmp_path,
        "run_id": "batch32",
        "experiment": PRACTICE_BATCH32_EXPERIMENT,
        "steps": 512,
        "batch_size": 32,
        "seed": 42,
        "profile": "reference",
        "device": "cuda",
    }
    args.update(override)

    with pytest.raises(ValueError, match="practice-batch32-v1 requires"):
        train(**args)

    assert not (tmp_path / "batch32").exists()


class _TinyDataset:
    def __len__(self):
        return 17

    def __getitem__(self, index):
        value = torch.tensor([index], dtype=torch.int64)
        return {"pixels": value, "actions": value}


def _tiny_payload(step: int, sampling_state: torch.Tensor) -> dict[str, object]:
    return {
        "step": step,
        "model": {"weight": torch.tensor([step], dtype=torch.float32)},
        "optimizer": {"step": torch.tensor(step)},
        "sampling_rng": sampling_state.clone(),
    }


def test_batch32_snapshot_is_immutable_and_keeps_sampling_prefix(tmp_path: Path):
    run = tmp_path / "batch32"
    run.mkdir()
    dataset = _TinyDataset()
    sampling_rng = torch.Generator().manual_seed(43)
    reference_rng = torch.Generator().manual_seed(43)

    for _ in range(128):
        batch_from_dataset(dataset, 32, sampling_rng)
        batch_from_dataset(dataset, 32, reference_rng)
    state_at_128 = sampling_rng.get_state()
    torch.manual_seed(2026)
    global_state = torch.get_rng_state()

    payload = _tiny_payload(128, state_at_128)
    save_experiment_checkpoint(
        run,
        payload,
        step=128,
        experiment=PRACTICE_BATCH32_EXPERIMENT,
    )
    regular_at_128 = (run / "checkpoint.pt").read_bytes()
    retained_at_128 = (run / "checkpoint-128.pt").read_bytes()
    assert retained_at_128 == regular_at_128
    assert torch.equal(sampling_rng.get_state(), state_at_128)
    assert torch.equal(torch.get_rng_state(), global_state)

    expected_next = batch_from_dataset(dataset, 32, reference_rng)
    actual_next = batch_from_dataset(dataset, 32, sampling_rng)
    for actual, expected in zip(actual_next, expected_next, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    retained_hash = retained_at_128
    payload["step"] = 512
    payload["model"]["weight"].fill_(512)  # type: ignore[index]
    payload["optimizer"]["step"] = torch.tensor(512)  # type: ignore[index]
    save_experiment_checkpoint(
        run,
        payload,
        step=512,
        experiment=PRACTICE_BATCH32_EXPERIMENT,
    )

    assert (run / "checkpoint-128.pt").read_bytes() == retained_hash
    midrun = torch.load(run / "checkpoint-128.pt", weights_only=True)
    final = torch.load(run / "checkpoint.pt", weights_only=True)
    assert midrun["step"] == 128
    assert final["step"] == 512
    assert midrun["model"]["weight"].item() == 128
    assert final["model"]["weight"].item() == 512
    assert torch.equal(midrun["sampling_rng"], state_at_128)


def test_matched_exposure_sampling_prefix_is_batch_size_invariant():
    dataset = _TinyDataset()
    batch32_rng = torch.Generator().manual_seed(43)
    batch8_rng = torch.Generator().manual_seed(43)
    batch32_prefix = [
        batch_from_dataset(dataset, 32, batch32_rng)[0].flatten()
        for _ in range(128)
    ]
    batch8_prefix = [
        batch_from_dataset(dataset, 8, batch8_rng)[0].flatten()
        for _ in range(512)
    ]

    torch.testing.assert_close(
        torch.cat(batch32_prefix),
        torch.cat(batch8_prefix),
        rtol=0,
        atol=0,
    )
    assert torch.equal(batch32_rng.get_state(), batch8_rng.get_state())
