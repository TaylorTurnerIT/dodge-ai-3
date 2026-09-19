"""Run the AD5 Track-B survival probe and gated 1-step MPC eval on one T4.

Two modes run as separate fresh processes from the remote driver:

``smoke`` hash-verifies the frozen checkpoint and mortal probe set, encodes
one batch on CUDA, fits a tiny unscored probe to scratch outputs (exercising
the report/artifact path including the null-AUPRC branch), and rolls two
short headless episodes proving the native adapter works on this device.

``scored`` re-verifies the immutable inputs, fits the full protocol probe on
CUDA, applies the prespecified AUPRC gate, and runs the 32-scenario MPC
comparison only on a pass.  A gate failure still packages the probe evidence
with an mpc-skipped record: the negative result is the finding, never a
silent skip.  Nothing here updates the world model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path
from typing import Any

CODE_ROOT = Path("/content/lewm-survival-code")
INPUT_ROOT = Path("/content/lewm-survival-inputs")
WORK_ROOT = Path("/content/lewm-survival-work")
RESULTS_ROOT = Path("/content/lewm-survival-results")

PROTOCOL_NAME = "survival_protocol.json"
SMOKE_MARKER = "SURVIVAL_SMOKE_COMPLETE"
SCORED_MARKER = "SURVIVAL_DRIVER_COMPLETE"


SMOKE_WINDOW_SEED = 907


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def probe_gate_decision(report: dict[str, Any], minimum: float) -> dict[str, Any]:
    """Apply the prespecified AUPRC gate to a finished probe report.

    Returns a JSON-serializable verdict; a null AUPRC (degenerate validation
    slice) or a below-minimum score fails closed with the reason recorded.
    """

    auprc = report.get("val_auprc")
    if not isinstance(auprc, (int, float)):
        return {"passed": False, "reason": "val_auprc is null; no ranking signal"}
    if not auprc >= minimum:
        return {
            "passed": False,
            "reason": f"val_auprc {auprc:.4f} below prespecified {minimum:.4f}",
        }
    return {"passed": True, "reason": f"val_auprc {auprc:.4f} meets {minimum:.4f}"}


def _read_protocol() -> dict[str, Any]:
    try:
        value = json.loads((CODE_ROOT / PROTOCOL_NAME).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("survival protocol is not valid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError("survival protocol must contain a JSON object")
    return value


def _verify_inputs(protocol: dict[str, Any]) -> dict[str, str]:
    """Check every frozen input byte against the protocol before any fit."""

    expected = protocol["inputs"]
    checkpoint = INPUT_ROOT / "checkpoint.pt"
    manifest = INPUT_ROOT / "probe-set" / "manifest.json"
    actual = {str(checkpoint): digest(checkpoint), str(manifest): digest(manifest)}
    if actual[str(checkpoint)] != expected["checkpoint.pt"]:
        raise ValueError("world checkpoint failed verification")
    if actual[str(manifest)] != expected["probe-set/manifest.json"]:
        raise ValueError("mortal probe manifest failed verification")
    return actual


def _smoke_batch(probe_set: Path):
    """Load two real probe windows for the smoke encode.

    Synthetic zeros are invalid input for the palette-arm encoder, which
    rejects any RGB outside its three colors; real recorded frames are
    palette-covered by construction.
    """

    import torch

    from dodge_native_game.variants.pixel_repr_ddqn.survival_probe import (
        _windows_with_labels,
    )

    windows, _, _ = _windows_with_labels(
        probe_set, "train", max_windows=2, seed=SMOKE_WINDOW_SEED
    )
    return torch.from_numpy(windows)


def _cuda_device() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device on this worker")
    return "cuda"


def run_smoke(protocol: dict[str, Any]) -> None:
    import torch

    from dodge_native_game.variants.pixel_repr_ddqn.mpc_eval import run_episode
    from dodge_native_game.variants.pixel_repr_ddqn.native_adapter import (
        PixelNativeAdapter,
    )
    from dodge_native_game.variants.pixel_repr_ddqn.pretrain import load_model
    from dodge_native_game.variants.pixel_repr_ddqn.survival_probe import fit_probe

    _verify_inputs(protocol)
    device = _cuda_device()
    checkpoint = INPUT_ROOT / "checkpoint.pt"
    model, _ = load_model(checkpoint)
    model = model.to(device).eval()
    with torch.no_grad():
        batch = _smoke_batch(INPUT_ROOT / "probe-set").to(device)
        encoded = model.encode(batch)
    assert tuple(encoded.shape) == (2, 4, 192), (
        f"unexpected encode shape {encoded.shape}"
    )
    print(f"SMOKE_ENCODE_OK shape={tuple(encoded.shape)}", flush=True)
    scratch = WORK_ROOT / "smoke"
    report = fit_probe(
        INPUT_ROOT / "probe-set",
        checkpoint,
        scratch / "probe.pt",
        device=device,
        steps=10,
        batch_size=32,
        seed=905,
        max_windows=200,
    )
    assert (scratch / "probe.json").is_file()
    print(f"SMOKE_PROBE_OK val_auprc={report['val_auprc']}", flush=True)
    for policy_name in ("neutral", "random"):
        if policy_name == "neutral":
            policy = lambda history, past: (0, None)  # noqa: E731
        else:
            policy = lambda history, past: (  # noqa: E731
                int(torch.randint(9, (1,)).item()),
                None,
            )
        result = run_episode(
            PixelNativeAdapter,
            24000,
            policy,
            model=model,
            device=torch.device(device),
            max_decisions=4,
            trace_dir=scratch / "trace" / policy_name,
        )
        assert result["survived"] <= 4
        assert (scratch / "trace" / policy_name / "steps.jsonl").is_file()
        assert (scratch / "trace" / policy_name / "frame_000.png").is_file()
    print("SMOKE_EPISODES_OK", flush=True)
    from dodge_native_game.variants.pixel_repr_ddqn.mpc_eval import _plan_action
    from dodge_native_game.variants.pixel_repr_ddqn.survival_probe import (
        SurvivalProbe,
    )

    probe = SurvivalProbe().to(device).eval()
    with torch.no_grad():
        planned = _plan_action(
            model, probe, batch[:1], [0, 0, 0], 2, torch.device(device)
        )
    assert planned[0] in range(9) and len(planned[1]) == 9
    print("SMOKE_PLAN_OK", flush=True)
    print(SMOKE_MARKER, flush=True)


def run_scored(protocol: dict[str, Any], run_id: str, source_hash: str) -> None:
    import torch
    from transformers import __version__ as transformers_version

    from dodge_native_game.variants.pixel_repr_ddqn.mpc_eval import (
        evaluate,
        plan_mpc_eval,
    )
    from dodge_native_game.variants.pixel_repr_ddqn.survival_probe import fit_probe

    _verify_inputs(protocol)
    device = _cuda_device()
    checkpoint = INPUT_ROOT / "checkpoint.pt"
    probe_root = INPUT_ROOT / "probe-set"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    probe_cfg = protocol["probe"]
    report = fit_probe(
        probe_root,
        checkpoint,
        RESULTS_ROOT / "probe.pt",
        device=device,
        steps=int(probe_cfg["steps"]),
        batch_size=int(probe_cfg["batch_size"]),
        seed=int(probe_cfg["seed"]),
        max_windows=int(probe_cfg["max_windows"]),
    )
    verdict = probe_gate_decision(report, float(protocol["gate"]["min_val_auprc"]))
    print(f"PROBE_GATE passed={verdict['passed']} {verdict['reason']}", flush=True)
    mpc_cfg = protocol["mpc"]
    if verdict["passed"]:
        mpc_report = evaluate(
            checkpoint,
            RESULTS_ROOT / "probe.pt",
            plan_mpc_eval(
                cohorts=tuple(int(c) for c in mpc_cfg["scenario_cohorts"])
            ),
            policies=dict(mpc_cfg["policies"]),
            trace_dir=RESULTS_ROOT / "trace",
            device=device,
            history_size=int(mpc_cfg["history_size"]),
            max_decisions=int(mpc_cfg["max_decisions"]),
        )
        (RESULTS_ROOT / "mpc-report.json").write_text(
            json.dumps(mpc_report, indent=1) + "\n"
        )
        print(
            f"MPC_COMPLETE {json.dumps(mpc_report['summary'])}", flush=True
        )
    else:
        (RESULTS_ROOT / "mpc-skipped.json").write_text(
            json.dumps({"verdict": verdict, "gate": protocol["gate"]}, indent=1)
            + "\n"
        )
    environment = {
        "run_id": run_id,
        "experiment": protocol["experiment"],
        "source_sha256": source_hash,
        "protocol": protocol,
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "cuda_device": torch.cuda.get_device_name(0),
        "world_model_sha256": report["world_model_sha256"],
        "probe_set_sha256": report["probe_manifest_sha256"],
        "gate_verdict": verdict,
    }
    (RESULTS_ROOT / "environment.json").write_text(
        json.dumps(environment, indent=1) + "\n"
    )
    archive = Path("/content/lewm-survival-results.tar.gz")
    with tarfile.open(archive, "w:gz") as output:
        output.add(RESULTS_ROOT, arcname="results")
    sha_path = Path("/content/lewm-survival-results.sha256")
    sha_path.write_text(digest(archive) + "\n")
    print(SCORED_MARKER, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke", "scored"))
    args = parser.parse_args()
    protocol = _read_protocol()
    run_id = os.environ["LEWM_RUN_ID"]
    source_hash = os.environ["LEWM_SOURCE_HASH"]
    if args.mode == "smoke":
        run_smoke(protocol)
    else:
        run_scored(protocol, run_id, source_hash)


if __name__ == "__main__":
    main()
