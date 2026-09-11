"""Small installed-package smoke check for the native data boundary."""

from __future__ import annotations

import argparse
import json

from .batch import NativeBatchEnvironment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    with NativeBatchEnvironment(
        step_frames=4,
        full_state=True,
        pixels=True,
        board=True,
    ) as environment:
        reset = environment.reset_batch([arguments.seed])
        step = environment.step_batch([0])
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "lane_count": reset.lane_count,
                    "seed": int(reset.seeds[0]),
                    "reset_frame": int(reset.frames[0]),
                    "step_frame": int(step.frames[0]),
                    "frames_advanced": int(step.frames_advanced[0]),
                    "board_shape": (
                        list(reset.board.shape) if reset.board is not None else None
                    ),
                    "pixel_shape": (
                        list(reset.pixels.shape) if reset.pixels is not None else None
                    ),
                    "snapshot_bytes": len(reset.snapshot_bytes[0] or b""),
                }
            )
        )


if __name__ == "__main__":
    main()
