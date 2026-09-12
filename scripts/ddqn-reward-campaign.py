"""Release 200k native-RGB reward screens only after a same-T4 telemetry gate."""

import argparse
import json
import runpy
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("/content/t4-history"))
    parser.add_argument("--treatments", nargs="+", required=True)
    parser.add_argument("--gate-worker")
    parser.add_argument("--log-interval", type=int, default=1000)
    args = parser.parse_args()
    args.history.mkdir(parents=True, exist_ok=True)
    if args.gate_worker:
        import torch

        from dodge_native_game.variants.cnn_image_ddqn.run import train_run

        torch.set_num_threads(1)
        assert "T4" in torch.cuda.get_device_name(0)
        started = time.perf_counter()
        root = train_run(
            history_root=args.history,
            run_id=args.gate_worker,
            steps=10000,
            seed=42,
            training_seed_count=700,
            evaluation_seed=512,
            observation_profile="native-rgb-v1",
            stack_size=4,
            device="cuda",
            replay_capacity=100000,
            warmup_steps=1000,
            batch_size=32,
            learning_rate=1e-4,
            update_every=4,
            target_sync_interval=10000,
            epsilon_decay_steps=500000,
            log_interval=args.log_interval,
            eval_episodes=1,
            eval_steps=4,
            reward_profile="combined-v1",
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        rows = [
            json.loads(line)
            for line in (root / "metrics.jsonl").read_text().splitlines()
        ]
        (args.history / f"{args.gate_worker}.json").write_text(
            json.dumps(
                {
                    "training_loop_seconds": 10000 / rows[-1]["throughput"],
                    "all_in_seconds": elapsed,
                    "optimizer_steps": rows[-1]["optimizer_step"],
                    "gpu": torch.cuda.get_device_name(0),
                    "log_interval": args.log_interval,
                }
            )
        )
        return
    scripts = Path(__file__).resolve().parent
    pairs = []
    for repeat in range(3):
        pair = {}
        order = ("minimal", "routine") if repeat % 2 == 0 else ("routine", "minimal")
        for mode in order:
            name = f"reward-gate-{repeat}-{mode}"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--history",
                    str(args.history),
                    "--treatments",
                    *args.treatments,
                    "--gate-worker",
                    name,
                    "--log-interval",
                    "1000" if mode == "routine" else "1000000",
                ],
                check=True,
            )
            pair[mode] = json.loads((args.history / f"{name}.json").read_text())
        pairs.append(pair)
    gate = runpy.run_path(str(scripts / "ddqn-rgb-campaign.py"))["gate_summary"](pairs)
    (args.history / "reward_telemetry_gate.json").write_text(json.dumps(gate, indent=2))
    print("REWARD_TELEMETRY_GATE", json.dumps(gate), flush=True)
    if not gate["passed"]:
        raise RuntimeError(
            "reward telemetry gate failed; training screens not released"
        )
    subprocess.run(
        [
            sys.executable,
            str(scripts / "ddqn-t4-screen.py"),
            "--history",
            str(args.history),
            "--treatments",
            *args.treatments,
        ],
        check=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Only the supervisor publishes failure after its worker has exited.
        if "--gate-worker" not in sys.argv:
            history_index = (
                sys.argv.index("--history") + 1 if "--history" in sys.argv else None
            )
            history = (
                Path(sys.argv[history_index])
                if history_index
                else Path("/content/t4-history")
            )
            if history.exists():
                temporary = history / ".campaign-failure.tar.gz"
                with tarfile.open(temporary, "w:gz") as archive:
                    for path in history.iterdir():
                        if path != temporary:
                            archive.add(path, arcname=path.name)
                temporary.rename(history / "campaign-failure.tar.gz")
        raise
