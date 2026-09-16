"""Matched pooling readouts over an existing frozen spatial feature bank.

The pooling study consumes the standard frame bank and its published spatial
sidecars as read-only inputs.  It never loads the world model for extraction,
creates a bank, or updates the world checkpoint; only the four decoder heads
are fitted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from . import large_probe as probe
from .input_probe import _pixel_bytes_sha256
from .model import INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC
from .palette import derive_palette, validate_palette_coverage
from .pooling_readout import ATTENTION, CONDITIONS, MEAN, make_pooling_decoders
from .pretrain import save_checkpoint
from .probe_bank import _state_digest
from .run_artifacts import append_metric, atomic_json, file_hash
from .spatial_bank import open_patches
from .spatial_fit import fit_spatial_decoders

__all__ = ["CONDITIONS", "EXPERIMENT", "run_pooling_study", "run_study"]

EXPERIMENT = "lewm-pooling-readout-v1"
MILESTONES = (512, 2048)
BATCH_SIZE = 32


def _validate_world_payload(
    dataset_root: Path, checkpoint: Path, payload: dict[str, Any]
) -> str:
    data_hash = file_hash(Path(dataset_root) / "manifest.json")
    if (
        payload.get("input_arm") != "palette"
        or payload.get("step") != 1024
        or payload.get("input_encoding")
        != INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC
        or payload.get("data_hash") != data_hash
    ):
        raise ValueError("Pooling study requires the completed matched palette world")
    return data_hash


def _open_readonly_inputs(
    dataset_root: Path,
    checkpoint: Path,
    bank_root: Path,
    payload: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any], dict[str, str]]:
    """Open and validate existing standard/spatial banks without writing them."""

    world_hash = file_hash(checkpoint)
    bank = probe.open_frame_bank(
        dataset_root,
        Path(bank_root) / "standard",
        checkpoint_sha256=world_hash,
    )
    if (
        bank.train.metadata["index_sha256"]
        != payload["palette_frame_index_sha256"]
        or _pixel_bytes_sha256(bank.train.pixels)
        != payload["palette_selected_pixels_sha256"]
    ):
        raise ValueError("Frame selection differs from source palette provenance")
    palette = derive_palette(bank.train.pixels)
    if palette.tolist() != [list(color) for color in payload["config"]["palette_rgb"]]:
        raise ValueError("Palette differs from source world")
    validate_palette_coverage(
        bank.validation.pixels,
        range(len(bank.validation.pixels)),
        palette,
    )

    sidecars: dict[str, Any] = {}
    sidecar_hashes: dict[str, str] = {}
    for split in ("train", "validation"):
        root = Path(bank_root) / "spatial" / split
        sidecars[split] = open_patches(getattr(bank, split), root)
        sidecar_hashes[split] = file_hash(root / "patches.npy")
    return bank, palette, sidecars, sidecar_hashes


def _parameter_count(module: torch.nn.Module, *, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def run_pooling_study(
    dataset_root: Path,
    checkpoint: Path,
    history_root: Path,
    bank_root: Path,
    run_id: str,
    *,
    device: str = "cuda",
    milestones: tuple[int, ...] = MILESTONES,
    eval_scope: str = "all",
) -> dict[str, Any]:
    """Fit matched mean/max/attention/grid4 readouts from frozen patch tokens.

    ``eval_scope="validation"`` restricts milestone evaluation streams to
    the validation split.  It exists for unscored smoke verification only;
    scored runs always evaluate the full bank.
    """

    if eval_scope not in ("all", "validation"):
        raise ValueError("eval_scope must be 'all' or 'validation'")

    dataset_root = Path(dataset_root)
    checkpoint = Path(checkpoint)
    bank_root = Path(bank_root)
    world_hash = file_hash(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("world checkpoint payload must be a mapping")
    data_hash = _validate_world_payload(dataset_root, checkpoint, payload)
    bank, palette, sidecars, sidecar_hashes = _open_readonly_inputs(
        dataset_root, checkpoint, bank_root, payload
    )

    # Every mode consumes the same raw spatial sidecar.  Pooling happens in
    # the readout module on the target device, so validation cannot silently
    # substitute pre-pooled or validation-derived features.
    train_features = {mode: sidecars["train"] for mode in CONDITIONS}
    all_features = probe._Concat((sidecars["train"], sidecars["validation"]))
    validation_span = bank.split_ranges["validation"]
    if eval_scope == "validation":
        evaluation_features = {mode: sidecars["validation"] for mode in CONDITIONS}
        val_start, val_stop = validation_span
        eval_pixels = bank.pixels[val_start:val_stop]
        eval_changed = bank.changed[val_start:val_stop]
        eval_records = bank.records[val_start:val_stop]
        eval_ranges = {"train": (0, 0), "validation": (0, val_stop - val_start)}
    else:
        evaluation_features = {mode: all_features for mode in CONDITIONS}
        eval_pixels, eval_changed = bank.pixels, bank.changed
        eval_records, eval_ranges = bank.records, bank.split_ranges
    training_mean = probe.stream_train_mean(bank.pixels, bank.indices("train"))

    decoders = make_pooling_decoders(device=device, seed=904)
    core_hashes = {
        mode: _state_digest(decoder.core) for mode, decoder in decoders.items()
    }
    if len(set(core_hashes.values())) != 1:
        raise RuntimeError("pooling decoder cores did not share initialization")
    initial_hashes = {
        mode: _state_digest(decoder) for mode, decoder in decoders.items()
    }
    if len(set(initial_hashes.values())) != 1:
        raise RuntimeError("pooling decoders did not share initialization")
    attention_query = decoders[ATTENTION].query.detach()
    if bool(torch.count_nonzero(attention_query)):
        raise RuntimeError("attention pooling query must start at zero")
    attention_extra_count = _parameter_count(
        decoders[ATTENTION], trainable_only=True
    ) - _parameter_count(decoders[MEAN], trainable_only=True)
    if attention_extra_count != int(attention_query.numel()):
        raise RuntimeError("attention trainable parameter count is not query-only")

    sidecar_metadata_hashes = {
        split: file_hash(bank_root / "spatial" / split / "metadata.json")
        for split in ("train", "validation")
    }
    palette_metadata = probe._palette_metadata(palette)
    core_parameter_count = _parameter_count(decoders[MEAN].core)
    metadata: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "world_model_sha256": world_hash,
        "data_sha256": data_hash,
        "frame_index_sha256": bank.frame_index_hash,
        "world_model_updates": 0,
        "world_model_fits": False,
        "decoder_seed": 904,
        "sampling_seed": 903,
        "batch_size": BATCH_SIZE,
        "milestones": list(milestones),
        "eval_scope": eval_scope,
        "diagnostic_only": True,
        "current_frame_only": True,
        "loss_kind": "palette-ce",
        "decoder_kind": "pooling-local-patch",
        "decoder_architecture": "pooling-local-mlp-194-256-256-192",
        # ``initial_state_sha256`` is the historical spatial-study field and
        # identifies the shared LocalPatchDecoder core.  Keep the wrapper
        # digest separate because the four pooling wrappers also carry a
        # zero-initialized query parameter.
        "initial_state_sha256": core_hashes[MEAN],
        "wrapper_initial_state_sha256": initial_hashes[MEAN],
        "core_initial_state_sha256": core_hashes[MEAN],
        "parameter_count": core_parameter_count,
        "core_parameter_count": core_parameter_count,
        "attention_extra_parameter_count": attention_extra_count,
        "attention_trainable_parameter_count": attention_extra_count,
        "attention_query_initialization": "zeros",
        "spatial_bank_metadata_sha256": sidecar_metadata_hashes,
        "spatial_bank_patches_sha256": sidecar_hashes,
        "spatial_bank_read_only": True,
        "conditions": list(CONDITIONS),
        **palette_metadata,
    }
    runs: dict[str, Path] = {}
    history_root = Path(history_root)
    for mode in CONDITIONS:
        run = history_root / f"{run_id}-{mode}"
        run.mkdir(parents=True, exist_ok=False)
        runs[mode] = run
        own = {
            **metadata,
            "representation": mode,
            "pooling_mode": mode,
            "attention_extra_parameter_count": (
                attention_extra_count if mode == ATTENTION else 0
            ),
        }
        atomic_json(run / "manifest.json", own)
        atomic_json(run / "config.json", own)
        atomic_json(
            run / "status.json",
            {
                "state": "running",
                "phase": "frozen pooling diagnostic",
                "step": 0,
            },
        )
        probe._link_or_copy(checkpoint, run / "checkpoint.pt")

    def on_step(row: dict[str, Any]) -> None:
        for mode in CONDITIONS:
            append_metric(
                runs[mode] / "metrics.jsonl",
                {
                    **row,
                    "loss": row[f"{mode}_loss"],
                    "representation": mode,
                    "pooling_mode": mode,
                },
            )
        if row["step"] % 128 == 0 or row["step"] == milestones[-1]:
            print("POOLING_DECODER", row, flush=True)

    def on_milestone(snapshot: Any, heads: dict[str, torch.nn.Module]) -> None:
        for mode in CONDITIONS:
            run, decoder = runs[mode], heads[mode]
            own = {
                **metadata,
                "representation": mode,
                "pooling_mode": mode,
                "attention_extra_parameter_count": (
                    attention_extra_count if mode == ATTENTION else 0
                ),
            }
            path = run / f"decoder-{snapshot.step}.pt"
            save_checkpoint(
                path,
                {
                    **own,
                    "model": snapshot.model[mode],
                    "optimizer": snapshot.optimizer[mode],
                    "sampler": snapshot.sampler,
                    "sampling_rng": snapshot.sampler,
                    "torch_rng": snapshot.torch_rng,
                    "cuda_rng": snapshot.cuda_rng,
                    "step": snapshot.step,
                    "latent_dim": 192,
                },
            )
            normal = probe.evaluate_decoder_stream(
                decoder,
                evaluation_features[mode],
                eval_pixels,
                eval_changed,
                eval_records,
                eval_ranges,
                training_mean,
                device=device,
                palette=palette,
            )
            wrong = probe.evaluate_decoder_stream(
                decoder,
                evaluation_features[mode],
                eval_pixels,
                eval_changed,
                eval_records,
                eval_ranges,
                training_mean,
                device=device,
                wrong_permutation=True,
                palette=palette,
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
                mode=mode,
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
                    "phase": "frozen pooling diagnostic",
                    "step": snapshot.step,
                },
            )
            if final:
                probe._link_or_copy(path, run / "decoder.pt")
                atomic_json(run / "report.json", clean)
            print(
                "POOLING_EVALUATION",
                mode,
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
        batch_size=BATCH_SIZE,
        on_step=on_step,
        on_milestone=on_milestone,
        conditions=CONDITIONS,
    )
    if file_hash(checkpoint) != world_hash:
        raise RuntimeError("World checkpoint changed during pooling study")
    for split in ("train", "validation"):
        path = bank_root / "spatial" / split / "patches.npy"
        if file_hash(path) != sidecar_hashes[split]:
            raise RuntimeError("Spatial sidecar changed during pooling study")

    result = {
        **metadata,
        "run_id": run_id,
        "arms": list(CONDITIONS),
        "pooling_modes": list(CONDITIONS),
    }
    atomic_json(history_root / f"{run_id}-pooling-comparison.json", result)
    return result


def run_study(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility alias matching the existing spatial worker entrypoint."""

    return run_pooling_study(*args, **kwargs)
