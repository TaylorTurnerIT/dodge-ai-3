"""Frozen one-step action-conditioning audit runner (SPEC AC1).

Read-only: the world model is loaded in evaluation mode, never updated, and
verified by hash before and after.  Windows come from existing episodes at
predetermined varied offsets; nothing is collected or fitted.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _fractions(values: Any) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError("fractions must be non-empty values in [0, 1]") from error
    if not result or any(not 0.0 <= value <= 1.0 for value in result):
        raise ValueError("fractions must be non-empty values in [0, 1]")
    return result


def main(argv: Any = None) -> dict[str, Any]:
    import torch

    from dodge_native_game.variants.pixel_repr_ddqn.dynamics import (
        audit_action_conditioning,
        iter_plan_windows,
        plan_audit_windows,
    )
    from dodge_native_game.variants.pixel_repr_ddqn.future_decode import (
        _validate_world_payload,
    )
    from dodge_native_game.variants.pixel_repr_ddqn.large_dataset import (
        make_large_dataset,
        read_dataset_metadata,
    )
    from dodge_native_game.variants.pixel_repr_ddqn.pretrain import load_model
    from dodge_native_game.variants.pixel_repr_ddqn.run_artifacts import (
        atomic_json,
        file_hash,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--split", default="validation", choices=("train", "validation")
    )
    parser.add_argument("--fractions", nargs="+", default=(0.25, 0.5, 0.75))
    parser.add_argument("--action-dim", type=int, default=9)
    args = parser.parse_args(argv)

    dataset_root = Path(args.dataset)
    checkpoint = Path(args.checkpoint)
    output = Path(args.output)
    fractions = _fractions(args.fractions)
    device = torch.device(args.device)

    world_hash = file_hash(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    data_hash = _validate_world_payload(dataset_root, checkpoint, payload)
    history_size = int(payload["config"]["history_size"])
    metadata = read_dataset_metadata(dataset_root)
    records = metadata.records[args.split]
    plan = plan_audit_windows(
        records, history_size=history_size, fractions=fractions
    )
    dataset = make_large_dataset(
        dataset_root, split=args.split, history_size=history_size
    )
    model, _ = load_model(checkpoint)
    model = model.to(device).eval()
    model.requires_grad_(False)
    print(
        f"AUDIT_PHASE {len(plan)} {args.split} windows "
        f"at fractions {list(fractions)}",
        flush=True,
    )
    windows = iter_plan_windows(
        dataset, records, plan, history_size=history_size
    )
    result = audit_action_conditioning(
        model, windows, device, action_dim=args.action_dim
    )
    if file_hash(checkpoint) != world_hash:
        raise RuntimeError("World checkpoint changed during audit")
    report = {
        "experiment": "lewm-dynamics-audit-v1",
        "world_model_sha256": world_hash,
        "data_sha256": data_hash,
        "world_model_updates": 0,
        "split": args.split,
        "history_size": history_size,
        "fractions": list(fractions),
        "action_dim": args.action_dim,
        "device": device.type,
        "torch_version": torch.__version__,
        **result,
    }
    atomic_json(output, report)
    print(
        "AUDIT_COMPLETE windows={} pred={:.6f} copy={:.6f} "
        "ratio={} win_rate={} best_rate={}".format(
            result["window_count"],
            result["prediction_mse"] or float("nan"),
            result["persistence_mse"] or float("nan"),
            result["prediction_vs_persistence_ratio"],
            result["prediction_win_rate"],
            result["recorded_action_best_rate"],
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    main()
