#!/usr/bin/env python3
"""Generate one bounded native replay into a disk cache for the web dashboard.

Writes frame-XXXX.png + collision-XXXX.png plus meta.json with progress.
Run with the project venv python so torch and dodge_native resolve.
"""
import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))


def write_meta(out_dir, payload):
    tmp_fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix=".json")
    with os.fdopen(tmp_fd, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp_path, os.path.join(out_dir, "meta.json"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--epsilon", type=float, default=0.0)
    args = parser.parse_args()

    from dodge_native_game.variants.cnn_image_ddqn.native_replay import (
        generate_native_replay,
    )

    seeds = [int(part) for part in args.seeds.split(",") if part.strip() != ""]
    os.makedirs(args.out_dir, exist_ok=True)
    checkpoint_mtime = os.stat(args.checkpoint).st_mtime
    write_meta(args.out_dir, {"status": "generating", "done": 0, "total": len(seeds)})
    summaries = []
    for position, seed in enumerate(seeds):
        seed_dir = os.path.join(args.out_dir, f"seed-{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        replay = generate_native_replay(
            args.checkpoint, seed=seed, steps=args.steps, device="cpu",
            epsilon=args.epsilon,
        )
        total = len(replay.frames)
        for frame in replay.frames:
            frame_path = os.path.join(seed_dir, f"frame-{frame.index:04d}.png")
            with open(frame_path, "wb") as fh:
                fh.write(frame.png)
            with open(
                os.path.join(seed_dir, f"collision-{frame.index:04d}.png"), "wb"
            ) as fh:
                fh.write(frame.collision_png)
        summaries.append(
            {
                "seed": seed,
                "frames": total,
                "checkpoint": os.path.basename(args.checkpoint),
                "checkpoint_mtime": checkpoint_mtime,
                "created_at": replay.created_at,
            }
        )
        write_meta(
            args.out_dir,
            {"status": "generating", "done": position + 1, "total": len(seeds)},
        )
    summaries.sort(key=lambda item: item["frames"], reverse=True)
    write_meta(
        args.out_dir,
        {
            "status": "ready",
            "replays": summaries,
            "epsilon": args.epsilon,
            "checkpoint": os.path.basename(args.checkpoint),
            "checkpoint_mtime": checkpoint_mtime,
        },
    )
    print(f"cached {len(summaries)} replays in {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
