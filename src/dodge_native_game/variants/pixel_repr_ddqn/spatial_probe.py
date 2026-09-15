"""Matched local spatial, global CLS, and direct-pixel diagnostic study."""

from __future__ import annotations

import gc
from pathlib import Path

import torch

from . import large_probe as probe
from .input_probe import _pixel_bytes_sha256
from .model import INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC
from .palette import derive_palette, validate_palette_coverage
from .pretrain import load_model, save_checkpoint
from .probe_bank import _state_digest
from .run_artifacts import append_metric, atomic_json, file_hash
from .spatial_bank import ExpandedFeatures, extract_patches
from .spatial_fit import fit_spatial_decoders, make_spatial_decoders

ARMS = ("cls", "patch", "pixels")
EXPERIMENT = "lewm-spatial-readout-v1"


def run_study(
    dataset_root: Path,
    checkpoint: Path,
    history_root: Path,
    bank_root: Path,
    run_id: str,
    *,
    device: str = "cuda",
    milestones: tuple[int, ...] = (512, 2048, 8192),
) -> dict:
    world_hash = file_hash(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        payload.get("input_arm") != "palette"
        or payload.get("step") != 1024
        or payload.get("input_encoding")
        != INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC
        or payload.get("data_hash") != file_hash(dataset_root / "manifest.json")
    ):
        raise ValueError("Spatial study requires the completed matched palette world")
    model, _ = load_model(checkpoint)
    model = model.to(device).eval()
    model.requires_grad_(False)
    print("SPATIAL_PHASE extraction", flush=True)
    probe._ensure_bank(
        model,
        dataset_root,
        bank_root / "standard",
        device=device,
        checkpoint_sha256=world_hash,
    )
    bank = probe.open_frame_bank(
        dataset_root, bank_root / "standard", checkpoint_sha256=world_hash
    )
    if (
        bank.train.metadata["index_sha256"] != payload["palette_frame_index_sha256"]
        or _pixel_bytes_sha256(bank.train.pixels)
        != payload["palette_selected_pixels_sha256"]
    ):
        raise ValueError("Frame selection differs from source palette provenance")
    palette = derive_palette(bank.train.pixels)
    if palette.tolist() != [list(c) for c in payload["config"]["palette_rgb"]]:
        raise ValueError("Palette differs from source world")
    validate_palette_coverage(
        bank.validation.pixels, range(len(bank.validation.pixels)), palette
    )
    patch_banks = {
        split: extract_patches(
            model, getattr(bank, split), bank_root / "spatial" / split, device=device
        )
        for split in ("train", "validation")
    }
    del model, payload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    features = {
        "cls": ExpandedFeatures(bank.features["cls"]),
        "patch": probe._Concat([patch_banks["train"], patch_banks["validation"]]),
        "pixels": ExpandedFeatures(bank.pixels, palette=palette),
    }
    train_features = {
        "cls": ExpandedFeatures(bank.train.cls),
        "patch": patch_banks["train"],
        "pixels": ExpandedFeatures(bank.train.pixels, palette=palette),
    }
    palette_metadata = probe._palette_metadata(palette)
    training_mean = probe.stream_train_mean(bank.pixels, bank.indices("train"))
    decoders = make_spatial_decoders(device=device, seed=904)
    metadata = {
        "experiment": EXPERIMENT,
        "world_model_sha256": world_hash,
        "data_sha256": bank.data_hash,
        "frame_index_sha256": bank.frame_index_hash,
        "milestones": list(milestones),
        "world_model_updates": 0,
        "decoder_seed": 904,
        "initial_state_sha256": _state_digest(decoders["cls"]),
        "parameter_count": sum(p.numel() for p in decoders["cls"].parameters()),
        "sampling_seed": 903,
        "batch_size": 32,
        "diagnostic_only": True,
        "current_frame_only": True,
        "loss_kind": "palette-ce",
        "decoder_architecture": "local-mlp-194-256-256-192",
        **palette_metadata,
    }
    runs = {arm: history_root / f"{run_id}-{arm}" for arm in ARMS}
    for arm, run in runs.items():
        run.mkdir(parents=True, exist_ok=False)
        own = {
            **metadata,
            "representation": arm,
            "direct_pixel_control": arm == "pixels",
        }
        atomic_json(run / "manifest.json", own)
        atomic_json(run / "config.json", own)
        probe._link_or_copy(checkpoint, run / "checkpoint.pt")
    print("SPATIAL_PHASE frozen diagnostic fitting", flush=True)

    def on_step(row):
        for arm in ARMS:
            append_metric(
                runs[arm] / "metrics.jsonl",
                {
                    **row,
                    "loss": row[f"{arm}_loss"],
                    "representation": arm,
                },
            )
        if row["step"] % 128 == 0:
            print("SPATIAL_DECODER", row, flush=True)

    def on_milestone(snapshot, heads):
        for arm in ARMS:
            run, decoder = runs[arm], heads[arm]
            path = run / f"decoder-{snapshot.step}.pt"
            own = {
                **metadata,
                "representation": arm,
                "direct_pixel_control": arm == "pixels",
            }
            save_checkpoint(
                path,
                {
                    **own,
                    "model": snapshot.model[arm],
                    "optimizer": snapshot.optimizer[arm],
                    "sampler": snapshot.sampler,
                    "torch_rng": snapshot.torch_rng,
                    "cuda_rng": snapshot.cuda_rng,
                    "step": snapshot.step,
                    "decoder_kind": "local-patch",
                    "latent_dim": 192,
                },
            )
            kwargs = dict(device=device, palette=palette)
            normal = probe.evaluate_decoder_stream(
                decoder,
                features[arm],
                bank.pixels,
                bank.changed,
                bank.records,
                bank.split_ranges,
                training_mean,
                **kwargs,
            )
            wrong = probe.evaluate_decoder_stream(
                decoder,
                features[arm],
                bank.pixels,
                bank.changed,
                bank.records,
                bank.split_ranges,
                training_mean,
                wrong_permutation=True,
                **kwargs,
            )
            wrong_by_index = {item["index"]: item for item in wrong["examples"]}
            for item in normal["examples"]:
                item["wrong_reconstructed"] = wrong_by_index[item["index"]][
                    "reconstructed"
                ]
            clean = {
                **own,
                "step": snapshot.step,
                "splits": normal["splits"],
                "wrong_latent_control": wrong["splits"],
                "examples": probe._clean_examples(normal["examples"]),
                "evaluation_only": True,
            }
            atomic_json(run / f"evaluation-{snapshot.step}.json", clean)
            views = probe._write_visuals(
                run,
                normal,
                mode=arm,
                step=snapshot.step,
                world_hash=world_hash,
                decoder_hash=file_hash(path),
                palette=palette,
            )
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
                atomic_json(run / "report.json", clean)
            print(
                "SPATIAL_EVALUATION",
                arm,
                snapshot.step,
                normal["splits"]["validation"],
                flush=True,
            )

    fit_spatial_decoders(
        decoders,
        train_features,
        bank.train.pixels,
        palette,
        milestones=milestones,
        batch_size=32,
        on_step=on_step,
        on_milestone=on_milestone,
    )
    if file_hash(checkpoint) != world_hash:
        raise RuntimeError("World checkpoint changed during fitting")
    result = {**metadata, "run_id": run_id, "arms": list(ARMS)}
    atomic_json(history_root / f"{run_id}-spatial-comparison.json", result)
    return result
