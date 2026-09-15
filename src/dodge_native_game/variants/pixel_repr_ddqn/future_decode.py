"""Teacher-forced next-frame prediction decode view for the frozen palette LeWM.

SPEC §AB.  §Z established that current-frame detail is recoverable from the
frozen encoder's patch tokens.  This study moves one step into the future:
the causal predictor consumes the observed context frames and executed
actions, and a fresh broadcast readout — architecturally identical to the §Z
CLS head — is fit on train-split predicted latents only.  The head is fit
with the balanced-bright weighted MSE
(0.5 cream, 0.5 other, per-frame-then-batch) established by the bright-probe
studies; palette-class metrics and the diff map classify the continuous
output to the nearest palette color.  The world model is never updated;
fitting is verified by checkpoint hash before and after.

The current-frame decode reuses the fitted §Z patch decoder so the gallery
contrasts near-perfect current reconstruction with the predicted next-frame
decode from the same frozen world.  Latent-persistence (current projected
CLS through the same predicted head) and pixel-persistence baselines are
reported alongside the prediction so a trivial copy cannot masquerade as
forecasting.
"""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import large_probe as probe
from .large_dataset import make_large_dataset, read_dataset_metadata
from .model import INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC
from .palette import (
    forward_logits,
    palette_indices,
    render_palette_rgb,
)
from .pretrain import load_model, save_checkpoint
from .probe_bank import _state_digest
from .run_artifacts import append_metric, atomic_json, file_hash
from .spatial_bank import ExpandedFeatures
from .spatial_fit import fit_spatial_decoders, make_spatial_decoders
from .spatial_readout import LocalPatchDecoder

__all__ = [
    "CONDITIONS",
    "EXPERIMENT",
    "MILESTONES",
    "PREDICTED",
    "SCENE_COUNT",
    "diff_map",
    "run_study",
    "window_plan",
]

EXPERIMENT = "lewm-future-decode-v1"
PREDICTED = "predicted"
CONDITIONS = (PREDICTED,)
MILESTONES = (512, 2048)
SCENE_COUNT = 16
BATCH_SIZE = 32
FEATURE_BATCH = 32
DIFF_RGB: tuple[int, int, int] = (255, 255, 255)


def _rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [
        device.index if device.index is not None else torch.cuda.current_device()
    ]


def window_plan(
    records: Sequence[Any],
    *,
    history_size: int,
    limit: int | None = None,
) -> tuple[tuple[int, int], ...]:
    """Pick one deterministic mid-episode window start per episode."""

    if isinstance(history_size, bool) or not isinstance(history_size, int):
        raise TypeError("history_size must be an integer")
    if history_size < 1:
        raise ValueError("history_size must be positive")
    plan: list[tuple[int, int]] = []
    for episode_index, record in enumerate(records):
        count = int(record.count)
        if count < history_size + 1:
            continue
        start = (count - history_size - 1) // 2
        plan.append((episode_index, start))
        if limit is not None and len(plan) >= limit:
            break
    if not plan:
        raise ValueError("no episode can supply a window of the requested history")
    return tuple(plan)


def _dataset_index(
    records: Sequence[Any], episode_index: int, start: int, history_size: int
) -> int:
    base = 0
    for record in records[:episode_index]:
        base += max(0, int(record.count) - history_size + 1)
    return base + start


def diff_map(observed: np.ndarray, decoded: np.ndarray) -> np.ndarray:
    """Render a high-contrast binary map of pixel differences.

    Both inputs are ``(3, H, W)`` uint8 RGB.  Matching pixels render black;
    any channel mismatch renders ``DIFF_RGB`` (white by default).
    """

    observed = np.asarray(observed)
    decoded = np.asarray(decoded)
    if observed.shape != decoded.shape or observed.ndim != 3 or observed.shape[0] != 3:
        raise ValueError("diff inputs must share shape (3, H, W)")
    if observed.dtype != np.uint8 or decoded.dtype != np.uint8:
        raise TypeError("diff inputs must be uint8 RGB")
    mismatch = (observed != decoded).any(axis=0)
    result = np.zeros(observed.shape, dtype=np.uint8)
    result[:, mismatch] = np.asarray(DIFF_RGB, dtype=np.uint8)[:, None]
    return result


def _validate_world_payload(
    dataset_root: Path, checkpoint: Path, payload: Mapping[str, Any]
) -> str:
    data_hash = file_hash(Path(dataset_root) / "manifest.json")
    if (
        payload.get("input_arm") != "palette"
        or payload.get("step") != 1024
        or payload.get("input_encoding")
        != INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC
        or payload.get("data_hash") != data_hash
    ):
        raise ValueError("Future view requires the completed matched palette world")
    return data_hash


def _load_current_decoder(
    path: Path, world_hash: str, device: torch.device
) -> tuple[LocalPatchDecoder, str]:
    decoder_hash = file_hash(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("decoder_kind") != "local-patch"
        or payload.get("representation") != "patch"
        or payload.get("world_model_sha256") != world_hash
    ):
        raise ValueError("Current-frame decoder must be the §Z patch readout")
    decoder = LocalPatchDecoder()
    decoder.load_state_dict(payload["model"], strict=True)
    decoder = decoder.to(device).eval()
    decoder.requires_grad_(False)
    return decoder, decoder_hash


def _extract_split(
    model: torch.nn.Module,
    dataset: Any,
    records: Sequence[Mapping[str, Any]],
    plan: Sequence[tuple[int, int]],
    *,
    history_size: int,
    device: torch.device,
    include_patches: bool,
) -> dict[str, Any]:
    """Encode windows once each and return teacher-forced one-step features."""

    predicted: list[np.ndarray] = []
    persistence: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    currents: list[np.ndarray] = []
    patches: list[np.ndarray] = []
    with (
        torch.random.fork_rng(devices=_rng_devices(device)),
        torch.no_grad(),
    ):
        for cursor in range(0, len(plan), FEATURE_BATCH):
            if cursor and cursor % (FEATURE_BATCH * 16) == 0:
                print(
                    f"FUTURE_FEATURES {cursor}/{len(plan)} windows",
                    flush=True,
                )
            chunk = plan[cursor : cursor + FEATURE_BATCH]
            samples = [
                dataset[_dataset_index(records, episode, start, history_size)]
                for episode, start in chunk
            ]
            pixels = torch.stack([sample["pixels"] for sample in samples]).to(device)
            actions = torch.stack([sample["actions"] for sample in samples]).to(device)
            encoded = model.encode(pixels)
            prediction = model.predict(encoded[:, :-1], actions)[:, -1]
            predicted.append(prediction.detach().cpu().numpy().astype(np.float32))
            persistence.append(
                encoded[:, -2].detach().cpu().numpy().astype(np.float32)
            )
            if include_patches:
                _, tokens = model.encode_readout_tokens(pixels[:, -2:-1])
                patches.append(
                    tokens[:, 0].detach().cpu().numpy().astype(np.float32)
                )
            targets.append(pixels[:, -1].detach().cpu().numpy())
            currents.append(pixels[:, -2].detach().cpu().numpy())
    result: dict[str, Any] = {
        "predicted": np.concatenate(predicted),
        "persistence": np.concatenate(persistence),
        "targets": np.concatenate(targets),
        "currents": np.concatenate(currents),
    }
    if include_patches:
        result["patches"] = np.concatenate(patches)
    return result


def _palette_metrics(
    predicted_classes: np.ndarray,
    target_classes: np.ndarray,
    current_classes: np.ndarray,
) -> dict[str, Any]:
    error = (predicted_classes != target_classes).astype(np.float64)
    changed = target_classes != current_classes
    changed_count = int(changed.sum())
    per_color_recall: list[float | None] = []
    for color in range(3):
        mask = changed & (target_classes == color)
        total = int(mask.sum())
        per_color_recall.append(
            float((predicted_classes[mask] == color).mean()) if total else None
        )
    return {
        "class_error": float(error.mean()),
        "changed_pixels": changed_count,
        "changed_region_class_error": (
            float(error[changed].mean()) if changed_count else None
        ),
        "palette_changed_per_color_recall": per_color_recall,
        "exact_match_share": float(
            (error.reshape(len(error), -1).sum(axis=1) == 0).mean()
        ),
    }


def _nearest_palette_classes(
    images: torch.Tensor, palette: np.ndarray
) -> np.ndarray:
    """Classify continuous ``(B,3,H,W)`` 0-1 images to nearest palette color."""

    colors = torch.as_tensor(palette, device=images.device, dtype=images.dtype).div(
        255.0
    )
    distances = (images.unsqueeze(1) - colors.view(1, 3, 3, 1, 1)).square().sum(dim=2)
    return distances.argmin(dim=1).detach().cpu().numpy()


def _evaluate_scenes(
    decoder: LocalPatchDecoder,
    current_decoder: LocalPatchDecoder,
    scenes: Mapping[str, Any],
    palette: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    """Decode validation scenes through prediction, persistence, and controls."""

    target_classes = palette_indices(scenes["targets"], palette)
    current_classes = palette_indices(scenes["currents"], palette)
    changed = target_classes != current_classes
    target01 = (
        torch.from_numpy(scenes["targets"].astype(np.float32)).to(device).div(255.0)
    )
    rows: dict[str, Any] = {}
    with (
        torch.random.fork_rng(devices=_rng_devices(device)),
        torch.no_grad(),
    ):
        for name, features in (
            (PREDICTED, scenes["predicted"]),
            ("persistence", scenes["persistence"]),
        ):
            output = forward_logits(
                decoder,
                torch.from_numpy(
                    np.asarray(
                        ExpandedFeatures(features)[:], dtype=np.float32
                    )
                ).to(device),
            )
            image01 = output.clamp(0.0, 1.0)
            classes = _nearest_palette_classes(image01, palette)
            metrics = _palette_metrics(classes, target_classes, current_classes)
            squared = (image01 - target01).square().mean(dim=1).detach().cpu().numpy()
            metrics["mse"] = float(squared.mean())
            metrics["changed_region_mse"] = (
                float(squared[changed].mean()) if bool(changed.any()) else None
            )
            rows[name] = {
                "metrics": metrics,
                "decoded": np.transpose(palette[classes], (0, 3, 1, 2)),
                "classes": classes,
            }
        patch_logits = forward_logits(
            current_decoder,
            torch.from_numpy(scenes["patches"].astype(np.float32)).to(device),
        )
        current_decoded = (
            render_palette_rgb(patch_logits, palette).detach().cpu().numpy() * 255.0
        ).round().astype(np.uint8)
    predicted_classes = rows[PREDICTED]["classes"]
    persistence_classes = rows["persistence"]["classes"]
    order = np.arange(len(target_classes))
    wrong = predicted_classes[(order + 1) % len(target_classes)]
    rows["wrong_latent_control"] = {
        "metrics": _palette_metrics(wrong, target_classes, current_classes)
    }
    rows["pixel_persistence_baseline"] = {
        "metrics": _palette_metrics(current_classes, target_classes, current_classes)
    }
    rows["current_decoded"] = current_decoded
    rows["target_classes"] = target_classes
    rows["current_classes"] = current_classes
    rows["predicted_classes"] = predicted_classes
    rows["persistence_classes"] = persistence_classes
    return rows


def _save_png(path: Path, frame: np.ndarray) -> None:
    probe._save_png(path, frame)


def _dashboard_frames(frames: Mapping[str, np.ndarray]) -> list[dict[str, str]]:
    """Render dashboard-schema frames with embedded image data URLs."""

    labels = {
        "current-observed": "Observed current frame",
        "current-decoded": "Decoded current frame (frozen patch readout)",
        "next-observed": "Observed next frame",
        "next-decoded": "Decoded predicted next frame",
        "next-persistence": "Decoded next frame from persistence control",
        "next-diff": "Pixel difference map (predicted vs observed next)",
    }
    return [
        {"label": labels[kind], "image": probe._data_url(frames[kind])}
        for kind in labels
    ]


def _write_scene_images(
    run: Path,
    step: int,
    scene_names: Sequence[str],
    scenes: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    root = run / "images" / f"step-{step}"
    root.mkdir(parents=True, exist_ok=True)
    views: list[dict[str, Any]] = []
    for index, name in enumerate(scene_names):
        prefix = f"scene-{index:02d}"
        current_observed = scenes["currents"][index]
        next_observed = scenes["targets"][index]
        current_decoded = evaluation["current_decoded"][index]
        next_decoded = evaluation[PREDICTED]["decoded"][index]
        persistence = evaluation["persistence"]["decoded"][index]
        diff = diff_map(next_observed, next_decoded)
        frames = {
            "current-observed": current_observed,
            "current-decoded": current_decoded,
            "next-observed": next_observed,
            "next-decoded": next_decoded,
            "next-persistence": persistence,
            "next-diff": diff,
        }
        for kind, frame in frames.items():
            _save_png(root / f"{prefix}-{kind}.png", frame)
        views.append(
            {
                "step": step,
                "scene": name,
                "scene_index": index,
                "diagnostic_only": True,
                "images": {
                    kind: f"images/step-{step}/{prefix}-{kind}.png" for kind in frames
                },
                "frames": _dashboard_frames(frames),
                "metadata": {
                    "label": f"{name} · next-frame prediction",
                    "scene": name,
                    "step": step,
                    "diagnostic_only": True,
                },
            }
        )
    return views


def run_study(
    dataset_root: Path,
    checkpoint: Path,
    current_decoder: Path,
    history_root: Path,
    run_id: str,
    *,
    device: str = "cpu",
    milestones: tuple[int, ...] = MILESTONES,
    scene_count: int = SCENE_COUNT,
    train_window_limit: int | None = None,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Any]:
    """Fit the predicted-latent readout and render the comparison view."""

    dataset_root = Path(dataset_root)
    checkpoint = Path(checkpoint)
    current_decoder = Path(current_decoder)
    history_root = Path(history_root)
    if isinstance(scene_count, bool) or not isinstance(scene_count, int):
        raise TypeError("scene_count must be an integer")
    if scene_count < 2:
        raise ValueError(
            "scene_count must be at least two for the wrong-latent control"
        )

    world_hash = file_hash(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    data_hash = _validate_world_payload(dataset_root, checkpoint, payload)
    target_device = torch.device(device)
    model, _ = load_model(checkpoint)
    model = model.to(target_device).eval()
    model.requires_grad_(False)
    patch_decoder, patch_decoder_hash = _load_current_decoder(
        current_decoder, world_hash, target_device
    )
    palette = np.asarray(payload["config"]["palette_rgb"], dtype=np.uint8)
    history_size = int(payload["config"]["history_size"])

    metadata = read_dataset_metadata(dataset_root)
    train_records = metadata.records["train"]
    validation_records = metadata.records["validation"]
    train_plan = window_plan(
        train_records, history_size=history_size, limit=train_window_limit
    )
    scene_plan = window_plan(
        validation_records, history_size=history_size, limit=scene_count
    )
    scene_names = tuple(
        str(validation_records[episode].episode_id) for episode, _ in scene_plan
    )

    print("FUTURE_PHASE train feature extraction", flush=True)
    train_dataset = make_large_dataset(
        dataset_root, split="train", history_size=history_size
    )
    train_features = _extract_split(
        model,
        train_dataset,
        train_records,
        train_plan,
        history_size=history_size,
        device=target_device,
        include_patches=False,
    )
    del train_dataset
    print("FUTURE_PHASE validation feature extraction", flush=True)
    validation_dataset = make_large_dataset(
        dataset_root, split="validation", history_size=history_size
    )
    scenes = _extract_split(
        model,
        validation_dataset,
        validation_records,
        scene_plan,
        history_size=history_size,
        device=target_device,
        include_patches=True,
    )
    del validation_dataset, model
    gc.collect()

    run = history_root / run_id
    run.mkdir(parents=True, exist_ok=False)
    decoders = make_spatial_decoders(
        device=target_device, seed=904, conditions=CONDITIONS
    )
    study_metadata: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "world_model_sha256": world_hash,
        "data_sha256": data_hash,
        "current_decoder_sha256": patch_decoder_hash,
        "current_decoder_source": "lewm-spatial-study-20260915-v1-patch",
        "milestones": list(milestones),
        "world_model_updates": 0,
        "decoder_seed": 904,
        "initial_state_sha256": _state_digest(decoders[PREDICTED]),
        "parameter_count": sum(
            p.numel() for p in decoders[PREDICTED].parameters()
        ),
        "sampling_seed": 903,
        "batch_size": batch_size,
        "diagnostic_only": True,
        "teacher_forced_one_step": True,
        "loss_kind": "balanced-bright",
        "equal_class_weights": True,
        "loss_class_weights": {"cream": 0.5, "other": 0.5},
        "bright_threshold": 0.8,
        "loss_normalization": "per-frame-then-batch",
        "decoder_architecture": "local-mlp-194-256-256-192",
        "readout": "predicted-latent-broadcast",
        "palette_rgb": palette.tolist(),
        "train_windows": len(train_plan),
        "scene_count": len(scene_plan),
        "scene_episode_ids": list(scene_names),
        "history_size": history_size,
    }
    atomic_json(run / "manifest.json", study_metadata)
    atomic_json(run / "config.json", study_metadata)
    probe._link_or_copy(checkpoint, run / "checkpoint.pt")
    probe._link_or_copy(current_decoder, run / "current-decoder.pt")

    def on_step(row: Mapping[str, Any]) -> None:
        append_metric(
            run / "metrics.jsonl",
            {**row, "loss": row[f"{PREDICTED}_loss"], "readout": PREDICTED},
        )
        if row["step"] % 128 == 0:
            print("FUTURE_DECODER", dict(row), flush=True)

    def on_milestone(snapshot, heads) -> None:
        decoder = heads[PREDICTED]
        path = run / f"decoder-{snapshot.step}.pt"
        save_checkpoint(
            path,
            {
                **study_metadata,
                "model": snapshot.model[PREDICTED],
                "optimizer": snapshot.optimizer[PREDICTED],
                "sampler": snapshot.sampler,
                "torch_rng": snapshot.torch_rng,
                "cuda_rng": snapshot.cuda_rng,
                "step": snapshot.step,
                "decoder_kind": "local-patch",
                "representation": PREDICTED,
                "latent_dim": 192,
            },
        )
        evaluation = _evaluate_scenes(
            decoder, patch_decoder, scenes, palette, target_device
        )
        views = _write_scene_images(
            run, snapshot.step, scene_names, scenes, evaluation
        )
        clean = {
            **study_metadata,
            "step": snapshot.step,
            "predicted": evaluation[PREDICTED]["metrics"],
            "persistence_control": evaluation["persistence"]["metrics"],
            "wrong_latent_control": evaluation["wrong_latent_control"]["metrics"],
            "pixel_persistence_baseline": evaluation[
                "pixel_persistence_baseline"
            ]["metrics"],
            "evaluation_only": True,
        }
        atomic_json(run / f"evaluation-{snapshot.step}.json", clean)
        atomic_json(run / f"visualizations-{snapshot.step}.json", views)
        atomic_json(run / "visualizations.json", views)
        final = snapshot.step == milestones[-1]
        atomic_json(
            run / "status.json",
            {
                "state": "completed" if final else "running",
                "phase": "frozen diagnostic",
                "step": snapshot.step,
            },
        )
        if final:
            probe._link_or_copy(path, run / "decoder.pt")
            atomic_json(run / "evaluation.json", clean)
            atomic_json(run / "report.json", clean)
        print(
            "FUTURE_EVALUATION",
            snapshot.step,
            json.dumps(clean[PREDICTED]),
            flush=True,
        )

    print("FUTURE_PHASE frozen diagnostic fitting", flush=True)
    fit_spatial_decoders(
        decoders,
        {PREDICTED: ExpandedFeatures(train_features["predicted"])},
        train_features["targets"],
        palette,
        milestones=milestones,
        batch_size=batch_size,
        conditions=CONDITIONS,
        loss_kind="balanced-bright",
        on_step=on_step,
        on_milestone=on_milestone,
    )
    if file_hash(checkpoint) != world_hash:
        raise RuntimeError("World checkpoint changed during fitting")

    from .future_gallery import write_gallery

    gallery = write_gallery(history_root, run_id)
    result = {
        **study_metadata,
        "run_id": run_id,
        "gallery": gallery.name,
    }
    atomic_json(history_root / f"{run_id}-future-comparison.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("history/dodge/gymnasium/pixel-repr-ddqn/large-practice-20260914-v2"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "history/dodge/gymnasium/pixel-repr-ddqn/"
            "lewm-spatial-study-20260915-v1-cls/checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--current-decoder",
        type=Path,
        default=Path(
            "history/dodge/gymnasium/pixel-repr-ddqn/"
            "lewm-spatial-study-20260915-v1-patch/decoder.pt"
        ),
    )
    parser.add_argument(
        "--history-root",
        type=Path,
        default=Path("history/dodge/gymnasium/pixel-repr-ddqn"),
    )
    parser.add_argument("--run-id", default="lewm-future-decode-20260915-v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--milestones", type=int, nargs="+", default=list(MILESTONES))
    parser.add_argument("--scenes", type=int, default=SCENE_COUNT)
    parser.add_argument("--train-window-limit", type=int, default=None)
    args = parser.parse_args(argv)
    return run_study(
        args.dataset_root,
        args.checkpoint,
        args.current_decoder,
        args.history_root,
        args.run_id,
        device=args.device,
        milestones=tuple(args.milestones),
        scene_count=args.scenes,
        train_window_limit=args.train_window_limit,
    )


if __name__ == "__main__":
    main()
