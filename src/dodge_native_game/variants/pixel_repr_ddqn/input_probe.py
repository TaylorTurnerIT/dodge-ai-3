"""Frozen raw-CLS readouts for the matched RGB/palette world-model screen."""

from __future__ import annotations

import gc
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import large_probe as probe
from .model import (
    INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC,
    INPUT_ENCODING_RGB_NEAREST_SYMMETRIC,
    LeWMConfig,
)
from .palette import derive_palette, validate_palette_coverage
from .pretrain import load_model, save_checkpoint
from .run_artifacts import append_metric, atomic_json, file_hash

ARMS = ("rgb", "palette")
EXPERIMENT = "lewm-input-encoding-v1"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _palette_tuple(value: object, *, name: str) -> tuple[tuple[int, int, int], ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{name} must be a list of RGB triples")
    colors: list[tuple[int, int, int]] = []
    for color in value:
        if not isinstance(color, (tuple, list)) or len(color) != 3:
            raise ValueError(f"{name} must contain RGB triples")
        if any(
            isinstance(channel, bool) or not isinstance(channel, (int, np.integer))
            for channel in color
        ):
            raise ValueError(f"{name} channels must be integers")
        channels = tuple(int(channel) for channel in color)
        if any(channel < 0 or channel > 255 for channel in channels):
            raise ValueError(f"{name} channels must be in [0, 255]")
        colors.append(channels)
    result = tuple(colors)
    if len(result) != 3 or tuple(sorted(result)) != result:
        raise ValueError(f"{name} must contain three sorted colors")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique colors")
    return result


def _require(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"input world checkpoint is missing {key}")
    return payload[key]


def _reference_config_without_input_fields() -> dict[str, object]:
    config = asdict(LeWMConfig.reference())
    config.pop("input_encoding")
    config.pop("palette_rgb")
    return config


def _pixel_bytes_sha256(pixels: object, *, batch_size: int = 64) -> str:
    digest = hashlib.sha256()
    for start in range(0, len(pixels), batch_size):  # type: ignore[arg-type]
        values = np.ascontiguousarray(pixels[start : start + batch_size])  # type: ignore[index]
        if values.dtype != np.uint8:
            raise ValueError("probe bank pixels must be uint8")
        digest.update(values.tobytes())
    return digest.hexdigest()


def validate_world_pair(
    payloads: Mapping[str, Mapping[str, Any]], dataset_hash: str
) -> None:
    """Validate both final world envelopes before loading model weights.

    The paired checkpoints must be the exact reference protocol with only the
    input encoding differing.  Shared train selection, palette provenance,
    optimizer budget, and both trace hashes are checked before any bank build.
    """

    if set(payloads) != set(ARMS):
        raise ValueError(f"input world checkpoints must contain exactly {ARMS}")
    if not isinstance(dataset_hash, str) or len(dataset_hash) != 64:
        raise ValueError("dataset_hash must be a SHA-256 hex digest")
    expected_config = _reference_config_without_input_fields()
    expected_encodings = {
        "rgb": INPUT_ENCODING_RGB_NEAREST_SYMMETRIC,
        "palette": INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC,
    }
    expected_shared = {
        "experiment": EXPERIMENT,
        "step": 1024,
        "batch_size": 32,
        "seed": 42,
        "initialization_seed": 42,
        "sampling_seed": 43,
        "stochastic_seed": 44,
        "world_model_updates": 1024,
        "sample_trace_steps": 1024,
        "data_hash": dataset_hash,
        "palette_source_split": "train",
    }
    required_shared = (
        *expected_shared,
        "palette_sha256",
        "palette_frame_index_sha256",
        "palette_selected_pixels_sha256",
        "sample_trace_sha256",
        "stochastic_trace_sha256",
        "initial_state_sha256",
        "parameter_count",
    )
    shared_values: dict[str, Any] = {}
    palette: tuple[tuple[int, int, int], ...] | None = None
    nested_palette_reference: Mapping[str, Any] | None = None
    for arm in ARMS:
        payload = payloads[arm]
        if not isinstance(payload, Mapping):
            raise ValueError(f"{arm} world checkpoint payload must be a mapping")
        for key, expected in expected_shared.items():
            value = _require(payload, key)
            if value != expected:
                raise ValueError(f"{arm} world checkpoint {key} is invalid")
        for key in required_shared:
            value = _require(payload, key)
            if key in shared_values and value != shared_values[key]:
                raise ValueError(f"paired world checkpoints disagree on {key}")
            shared_values.setdefault(key, value)
        if _require(payload, "input_arm") != arm:
            raise ValueError(f"world checkpoint input_arm does not match {arm}")
        encoding = _require(payload, "input_encoding")
        if encoding != expected_encodings[arm]:
            raise ValueError(f"{arm} world checkpoint input_encoding is invalid")
        config = _require(payload, "config")
        if not isinstance(config, Mapping):
            raise ValueError(f"{arm} world checkpoint config must be a mapping")
        config_without_input = dict(config)
        if config_without_input.pop("input_encoding", None) != encoding:
            raise ValueError(f"{arm} world checkpoint config encoding is invalid")
        config_palette = config_without_input.pop("palette_rgb", None)
        if arm == "rgb":
            if config_palette is not None:
                raise ValueError(
                    "rgb world checkpoint config unexpectedly has a palette"
                )
            config_palette_tuple = None
        else:
            if config_palette is None:
                raise ValueError("palette world checkpoint config is missing a palette")
            config_palette_tuple = _palette_tuple(config_palette, name="config palette")
        if _canonical(config_without_input) != _canonical(expected_config):
            raise ValueError(
                f"{arm} world checkpoint config is not reference-compatible"
            )
        payload_palette = _palette_tuple(
            _require(payload, "palette_rgb"), name="palette_rgb"
        )
        if arm == "palette" and config_palette_tuple != payload_palette:
            raise ValueError("palette world checkpoint config differs from palette_rgb")
        if palette is None:
            palette = payload_palette
        elif payload_palette != palette:
            raise ValueError("paired world checkpoints disagree on palette_rgb")
        nested_palette = _require(payload, "palette")
        if not isinstance(nested_palette, Mapping):
            raise ValueError(f"{arm} world checkpoint palette provenance is invalid")
        if nested_palette_reference is None:
            nested_palette_reference = nested_palette
        elif _canonical(nested_palette) != _canonical(nested_palette_reference):
            raise ValueError("paired world checkpoints disagree on palette provenance")
        nested_colors = _palette_tuple(
            _require(nested_palette, "palette_rgb"), name="palette.palette_rgb"
        )
        if nested_colors != payload_palette:
            raise ValueError(f"{arm} world checkpoint nested palette differs")
        for key in (
            "palette_sha256",
            "palette_source_split",
            "palette_frame_index_sha256",
            "palette_selected_pixels_sha256",
        ):
            if _require(nested_palette, key) != _require(payload, key):
                raise ValueError(f"{arm} world checkpoint palette {key} differs")
    assert palette is not None
    palette_bytes = bytes(channel for color in palette for channel in color)
    palette_hash = hashlib.sha256(palette_bytes).hexdigest()
    if shared_values["palette_sha256"] != palette_hash:
        raise ValueError("input world checkpoint palette_sha256 is invalid")


def validate_banks(banks) -> None:
    """Require identical pixels, changes, selection, and split provenance."""
    first, second = (banks[arm] for arm in ARMS)
    if (
        first.data_hash != second.data_hash
        or first.frame_index_hash != second.frame_index_hash
    ):
        raise ValueError("input comparison bank dataset or frame indices differ")
    if first.records != second.records or first.split_ranges != second.split_ranges:
        raise ValueError("input comparison bank records or splits differ")
    for split in ("train", "validation"):
        a, b = getattr(first, split), getattr(second, split)
        for key in ("pixels", "changed"):
            if a.metadata["files"][key] != b.metadata["files"][key]:
                raise ValueError(f"input comparison {split} {key} differs")


def run_probes(
    dataset_root: Path,
    checkpoints: dict[str, Path],
    history_root: Path,
    bank_root: Path,
    run_id: str,
    *,
    device: str = "cuda",
    milestones: tuple[int, ...] = (512, 2048, 8192),
) -> dict:
    """Extract once per world model, then fit two matched diagnostic heads."""
    dataset_hash = file_hash(Path(dataset_root) / "manifest.json")
    hashes = {arm: file_hash(checkpoints[arm]) for arm in ARMS}
    payloads = {
        arm: torch.load(checkpoints[arm], map_location="cpu", weights_only=True)
        for arm in ARMS
    }
    validate_world_pair(payloads, dataset_hash)
    banks, configs = {}, {}
    for arm in ARMS:
        checkpoint = checkpoints[arm]
        configs[arm] = payloads[arm]["config"]
        model, _ = load_model(checkpoint)
        model = model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        probe._ensure_bank(
            model,
            dataset_root,
            bank_root / arm,
            device=device,
            checkpoint_sha256=hashes[arm],
        )
        banks[arm] = probe.open_frame_bank(
            dataset_root,
            bank_root / arm,
            checkpoint_sha256=hashes[arm],
        )
        if file_hash(checkpoint) != hashes[arm]:
            raise RuntimeError("world checkpoint changed during extraction")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    validate_banks(banks)
    bank = banks["rgb"]
    palette_payload = payloads["palette"]
    if (
        bank.train.metadata.get("index_sha256")
        != palette_payload["palette_frame_index_sha256"]
    ):
        raise ValueError("palette checkpoint and train bank frame selection differ")
    if (
        _pixel_bytes_sha256(bank.train.pixels)
        != palette_payload["palette_selected_pixels_sha256"]
    ):
        raise ValueError("palette checkpoint and train bank pixels differ")
    palette = derive_palette(bank.train.pixels)
    if len(palette) != 3 or palette.tolist() != [
        list(c) for c in configs["palette"]["palette_rgb"]
    ]:
        raise ValueError("diagnostic palette differs from world input palette")
    validate_palette_coverage(
        bank.validation.pixels, range(len(bank.validation.pixels)), palette
    )
    palette_metadata = probe._palette_metadata(palette)
    training_mean = probe.stream_train_mean(bank.pixels, bank.indices("train"))
    runs = {arm: history_root / f"{run_id}-{arm}-cls" for arm in ARMS}
    metadata = {}
    for arm, run in runs.items():
        run.mkdir(parents=True, exist_ok=False)
        metadata[arm] = {
            "experiment": EXPERIMENT,
            "input_arm": arm,
            "input_encoding": configs[arm]["input_encoding"],
            "representation": "cls",
            "loss_kind": "palette-ce",
            "world_model_sha256": hashes[arm],
            "data_sha256": bank.data_hash,
            "frame_index_sha256": bank.frame_index_hash,
            "milestones": list(milestones),
            "world_model_updates": 0,
            "decoder_seed": 904,
            "sampling_seed": 903,
            "diagnostic_only": True,
            "current_frame_only": True,
            **palette_metadata,
        }
        atomic_json(run / "manifest.json", metadata[arm])
        atomic_json(run / "config.json", metadata[arm])
        probe._link_or_copy(checkpoints[arm], run / "checkpoint.pt")
    first, second = probe.make_decoder_pair(
        192, device=device, output_channels=3, raw_logits=True
    )

    def on_step(row):
        for arm, key in zip(ARMS, ("cls", "projected"), strict=True):
            append_metric(
                runs[arm] / "metrics.jsonl",
                {
                    **row,
                    "input_arm": arm,
                    "internal_fit_slot": key,
                    "loss": row[f"{key}_loss"],
                    "representation": "cls",
                },
            )
        if row["step"] % 128 == 0:
            print("INPUT_DECODER", row, flush=True)

    def on_milestone(snapshot, rgb_decoder, palette_decoder):
        for arm, slot, decoder in zip(
            ARMS, ("cls", "projected"), (rgb_decoder, palette_decoder), strict=True
        ):
            run, own = runs[arm], banks[arm]
            path = run / f"decoder-{snapshot.step}.pt"
            save_checkpoint(
                path,
                {
                    **metadata[arm],
                    "model": snapshot.model[slot],
                    "optimizer": snapshot.optimizer[slot],
                    "sampler": snapshot.sampler,
                    "torch_rng": snapshot.torch_rng,
                    "cuda_rng": snapshot.cuda_rng,
                    "step": snapshot.step,
                    "decoder_kind": "cls",
                    "latent_dim": 192,
                },
            )
            normal = probe.evaluate_decoder_stream(
                decoder,
                own.features["cls"],
                own.pixels,
                own.changed,
                own.records,
                own.split_ranges,
                training_mean,
                device=device,
                palette=palette,
            )
            wrong = probe.evaluate_decoder_stream(
                decoder,
                own.features["cls"],
                own.pixels,
                own.changed,
                own.records,
                own.split_ranges,
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
                **metadata[arm],
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
                mode="cls",
                step=snapshot.step,
                world_hash=hashes[arm],
                decoder_hash=file_hash(path),
                palette=palette,
            )
            atomic_json(run / f"visualizations-{snapshot.step}.json", views)
            atomic_json(run / "visualizations.json", views)
            atomic_json(
                run / "status.json",
                {
                    "state": "completed"
                    if snapshot.step == milestones[-1]
                    else "running",
                    "phase": "frozen diagnostic",
                    "step": snapshot.step,
                },
            )
            if snapshot.step == milestones[-1]:
                probe._link_or_copy(path, run / "decoder.pt")
                atomic_json(run / "report.json", clean)
            print(
                "INPUT_EVALUATION",
                arm,
                snapshot.step,
                normal["splits"]["validation"],
                flush=True,
            )

    probe.fit_matched_decoders(
        first,
        second,
        banks["rgb"].train.cls,
        banks["palette"].train.cls,
        bank.train.pixels,
        milestones=milestones,
        batch_size=32,
        loss_kind="palette-ce",
        palette=palette,
        on_step=on_step,
        on_milestone=on_milestone,
    )
    for arm in ARMS:
        if file_hash(checkpoints[arm]) != hashes[arm]:
            raise RuntimeError("world checkpoint changed during diagnostic fitting")
    result = {
        "experiment": EXPERIMENT,
        "run_id": run_id,
        "arms": list(ARMS),
        "world_model_sha256": hashes,
        "data_sha256": bank.data_hash,
        "frame_index_sha256": bank.frame_index_hash,
        "milestones": list(milestones),
        "representation": "cls",
        "loss_kind": "palette-ce",
        **palette_metadata,
    }
    atomic_json(history_root / f"{run_id}-input-comparison.json", result)
    return result
