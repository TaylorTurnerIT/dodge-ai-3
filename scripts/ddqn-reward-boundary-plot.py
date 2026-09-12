"""Plot the actual native reward geometry, not a Python reimplementation."""

from pathlib import Path

import dodge_native
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    coordinates = np.linspace(2, 125, 247)
    costs = np.array(
        [
            [
                dodge_native.reward_boundary_costs(x, y, 2, 125, 2, 16)
                for x in coordinates
            ]
            for y in coordinates
        ]
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    fields = [
        costs[:, :, 0],
        costs[:, :, 1],
        0.01 * costs[:, :, 0] + 0.1 * costs[:, :, 1],
    ]
    titles = [
        "Edge cost (2-pixel band)",
        "Corner cost (16-pixel overlap)",
        "Illustrative penalty/frame:\n0.01 edge + 0.1 corner",
    ]
    for ax, field, title in zip(axes, fields, titles, strict=True):
        plot = ax.imshow(field, extent=(2, 125, 125, 2), cmap="magma", vmin=0)
        ax.set(title=title, xlabel="Native player-center x", ylabel="y")
        fig.colorbar(plot, ax=ax, shrink=0.7)
    fig.suptitle(
        "Native reward geometry — illustrative weights, not active training rewards"
    )
    destination = Path("analysis/DDQN_REWARD_BOUNDARY.png")
    fig.savefig(destination, dpi=160)
    print(destination)


if __name__ == "__main__":
    main()
