"""Run the P9 DDQN screen on frozen LeWM latents on one T4.

Two modes run as separate fresh processes from the remote driver:

``smoke`` hash-verifies the frozen checkpoint, encodes one batch on CUDA,
collects two tiny epsilon-greedy episodes through the shared live-input
boundary, runs two TD updates, round-trips a checkpoint, and rolls one
greedy episode.  ``scored`` re-verifies the immutable inputs and runs the
frozen P9 protocol: epsilon-greedy collection to 200k decisions, one TD
update per decision after warmup, greedy evaluation on the 64 screen
scenarios every 20k decisions, best-by-eval plus final checkpoints.
Nothing here updates the world model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tarfile
from pathlib import Path
from typing import Any

CODE_ROOT = Path("/content/lewm-ddqn-code")
INPUT_ROOT = Path("/content/lewm-ddqn-inputs")
WORK_ROOT = Path("/content/lewm-ddqn-work")
RESULTS_ROOT = Path("/content/lewm-ddqn-results")

PROTOCOL_NAME = "ddqn_protocol.json"
SMOKE_MARKER = "DDQN_SMOKE_COMPLETE"
SCORED_MARKER = "DDQN_DRIVER_COMPLETE"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _read_protocol() -> dict[str, Any]:
    try:
        value = json.loads((CODE_ROOT / PROTOCOL_NAME).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("ddqn protocol is not valid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError("ddqn protocol must contain a JSON object")
    return value


def _verify_inputs(protocol: dict[str, Any]) -> dict[str, str]:
    """Check the frozen checkpoint bytes against the protocol."""

    expected = protocol["inputs"]
    checkpoint = INPUT_ROOT / "checkpoint.pt"
    actual = digest(checkpoint)
    if actual != expected["checkpoint.pt"]:
        raise ValueError("world checkpoint failed verification")
    return {"checkpoint.pt": actual}


def _cuda_device() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("P9 screen requires CUDA; no silent CPU fallback")
    return "cuda"


def epsilon_at(decisions: int, cfg: dict[str, Any]) -> float:
    start = float(cfg["epsilon_start"])
    final = float(cfg["epsilon_final"])
    span = int(cfg["epsilon_decay_decisions"])
    if decisions >= span:
        return final
    return start + (final - start) * (decisions / span)


def run_smoke(protocol: dict[str, Any]) -> None:
    import numpy
    import torch

    from dodge_native_game.variants.pixel_repr_ddqn.ddqn import (
        DDQNAgent,
        ReplayBuffer,
        collect_episode,
    )
    from dodge_native_game.variants.pixel_repr_ddqn.mpc_eval import (
        plan_mpc_eval,
    )
    from dodge_native_game.variants.pixel_repr_ddqn.native_adapter import (
        PixelNativeAdapter,
    )
    from dodge_native_game.variants.pixel_repr_ddqn.pretrain import load_model

    _verify_inputs(protocol)
    device = _cuda_device()
    checkpoint = INPUT_ROOT / "checkpoint.pt"
    model, _ = load_model(checkpoint)
    model = model.to(device).eval()
    with torch.no_grad():
        adapter = PixelNativeAdapter(
            scenario=plan_mpc_eval()[0][1]
        )
        try:
            frame = adapter.reset(seed=24000)
        finally:
            adapter.close()
        batch = (
            torch.from_numpy(numpy.asarray(frame, dtype="uint8"))
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(1, 4, 1, 1, 1)
            .to(device)
        )
        encoded = model.encode(batch)
    assert tuple(encoded.shape) == (1, 4, 192), (
        f"unexpected encode shape {encoded.shape}"
    )
    print(f"SMOKE_ENCODE_OK shape={tuple(encoded.shape)}", flush=True)
    agent = DDQNAgent(device=device)
    buffer = ReplayBuffer(capacity=64)
    rng = random.Random(7)
    scenarios = plan_mpc_eval(cohorts=(0, 1))
    for _label, cfg, seed in (scenarios[0], scenarios[32]):
        report = collect_episode(
            lambda cfg=cfg: PixelNativeAdapter(scenario=cfg),
            seed, agent, model, rng, epsilon=1.0,
            device=torch.device(device), max_decisions=4,
        )
        for transition in report["transitions"]:
            buffer.push(*transition)
    assert len(buffer) == 8, len(buffer)
    batch_tensors = buffer.sample(4, rng)
    loss = agent.update(batch_tensors)
    assert loss == loss and abs(loss) != float("inf")
    print(f"SMOKE_UPDATE_OK loss={loss:.4f}", flush=True)
    scratch = WORK_ROOT / "smoke"
    scratch.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), scratch / "agent.pt")
    clone = DDQNAgent(device=device)
    clone.load_state_dict(torch.load(scratch / "agent.pt", weights_only=True))
    assert clone.updates == agent.updates == 1
    label, cfg, seed = scenarios[1]
    report = collect_episode(
        lambda: PixelNativeAdapter(scenario=cfg),
        seed, agent, model, rng, epsilon=0.0,
        device=torch.device(device), max_decisions=4,
    )
    assert report["survived"] <= 4
    print("SMOKE_EPISODES_OK", flush=True)
    print(SMOKE_MARKER, flush=True)


def run_scored(protocol: dict[str, Any], run_id: str, source_hash: str) -> None:
    import torch
    from transformers import __version__ as transformers_version

    from dodge_native_game.variants.pixel_repr_ddqn.ddqn import (
        DDQNAgent,
        ReplayBuffer,
        collect_episode,
    )
    from dodge_native_game.variants.pixel_repr_ddqn.mpc_eval import (
        plan_mpc_eval,
    )
    from dodge_native_game.variants.pixel_repr_ddqn.native_adapter import (
        PixelNativeAdapter,
    )
    from dodge_native_game.variants.pixel_repr_ddqn.pretrain import load_model

    _verify_inputs(protocol)
    device = _cuda_device()
    train = protocol["training"]
    checkpoint = INPUT_ROOT / "checkpoint.pt"
    model, _ = load_model(checkpoint)
    model = model.to(device).eval()
    torch.manual_seed(int(train["seed"]))
    rng_act = random.Random(int(train["seed"]))
    rng_batch = random.Random(1000 + int(train["seed"]))
    agent = DDQNAgent(
        gamma=float(train["gamma"]),
        learning_rate=float(train["learning_rate"]),
        sync_interval=int(train["sync_interval"]),
        device=device,
    )
    buffer = ReplayBuffer(capacity=int(train["replay_capacity"]))
    scenarios = plan_mpc_eval(
        cohorts=tuple(int(c) for c in train["scenario_cohorts"])
    )
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    ckpt_dir = RESULTS_ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    budget = int(train["decisions"])
    warmup = int(train["warmup_decisions"])
    eval_every = int(train["eval_every_decisions"])
    batch_size = int(train["batch_size"])
    max_episode = int(train["max_decisions"])
    metrics_path = RESULTS_ROOT / "metrics.jsonl"
    eval_path = RESULTS_ROOT / "eval.json"
    evaluations: list[dict[str, Any]] = []
    decisions = 0
    updates = 0
    loss_window: list[float] = []
    best_mean: float | None = None
    episode_index = 0

    def save(tag: str) -> None:
        torch.save(
            {
                "agent": agent.state_dict(),
                "decisions": decisions,
                "updates": updates,
                "rng_act": rng_act.getstate(),
                "rng_batch": rng_batch.getstate(),
                "torch_rng": torch.get_rng_state(),
            },
            ckpt_dir / f"agent-{tag}.pt",
        )

    def evaluate() -> dict[str, Any]:
        survived: dict[str, float] = {}
        for label, cfg, seed in scenarios:
            report = collect_episode(
                lambda cfg=cfg: PixelNativeAdapter(scenario=cfg),
                seed, agent, model, rng_act, epsilon=0.0,
                device=torch.device(device), max_decisions=max_episode,
            )
            survived[label] = float(report["survived"])
        mean = sum(survived.values()) / len(survived)
        return {"decisions": decisions, "mean": mean, "survived": survived}

    while decisions < budget:
        label, cfg, seed = scenarios[episode_index % len(scenarios)]
        episode_index += 1
        report = collect_episode(
            lambda cfg=cfg: PixelNativeAdapter(scenario=cfg),
            seed, agent, model, rng_act,
            epsilon=epsilon_at(decisions, train),
            device=torch.device(device), max_decisions=max_episode,
        )
        for transition in report["transitions"]:
            buffer.push(*transition)
            decisions += 1
            if decisions > warmup and len(buffer) >= batch_size:
                loss = agent.update(buffer.sample(batch_size, rng_batch))
                updates += 1
                loss_window.append(loss)
                if len(loss_window) > 1000:
                    del loss_window[:-1000]
                if decisions % 1000 == 0:
                    with metrics_path.open("a", encoding="utf-8") as stream:
                        stream.write(
                            json.dumps(
                                {
                                    "decisions": decisions,
                                    "updates": updates,
                                    "epsilon": epsilon_at(decisions, train),
                                    "replay_size": len(buffer),
                                    "mean_loss_1k": sum(loss_window)
                                    / len(loss_window),
                                }
                            )
                            + "\n"
                        )
            if decisions >= budget:
                break
        if decisions // eval_every > len(evaluations):
            row = evaluate()
            evaluations.append(row)
            print(
                f"EVAL decisions={decisions} mean={row['mean']:.1f}",
                flush=True,
            )
            if best_mean is None or row["mean"] > best_mean:
                best_mean = row["mean"]
                save("best")
    if not evaluations or evaluations[-1]["decisions"] != decisions:
        row = evaluate()
        evaluations.append(row)
        print(
            f"EVAL decisions={decisions} mean={row['mean']:.1f}", flush=True
        )
        if best_mean is None or row["mean"] > best_mean:
            best_mean = row["mean"]
            save("best")
    save("final")
    eval_path.write_text(json.dumps(evaluations, indent=1) + "\n")
    environment = {
        "run_id": run_id,
        "experiment": protocol["experiment"],
        "source_sha256": source_hash,
        "protocol": protocol,
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "cuda_device": torch.cuda.get_device_name(0),
        "world_model_sha256": protocol["inputs"]["checkpoint.pt"],
        "decisions": decisions,
        "updates": updates,
        "best_eval_mean": best_mean,
    }
    (RESULTS_ROOT / "environment.json").write_text(
        json.dumps(environment, indent=1) + "\n"
    )
    archive = Path("/content/lewm-ddqn-results.tar.gz")
    with tarfile.open(archive, "w:gz") as output:
        output.add(RESULTS_ROOT, arcname="results")
    digest_value = digest(archive)
    Path("/content/lewm-ddqn-results.sha256").write_text(digest_value + "\n")
    print(SCORED_MARKER, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "scored"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-hash", default="")
    args = parser.parse_args()
    protocol = _read_protocol()
    if args.mode == "smoke":
        run_smoke(protocol)
    else:
        run_scored(protocol, args.run_id, args.source_hash)


if __name__ == "__main__":
    main()
