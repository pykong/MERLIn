from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.analyzer.utils.coloring import generate_color_mapping


def plot_reward_distribution(data: pd.DataFrame, tail: int, out_file: Path) -> None:
    """
    Plot the reward distributions of the experiments as stacked histograms.

    Args:
        data (pd.DataFrame): The frame holding the experimental data.
    """
    variants = data["variant"].unique()

    tail_df = data.groupby(["variant", "run"]).tail(tail)

    # create color map
    color_map = generate_color_mapping(variants)  # type:ignore

    # Calculate number of rows for the 2-column layout
    nrows = len(variants) // 2
    figsize = (15, nrows * 6)
    fig, axes = plt.subplots(nrows, 2, sharex=True, figsize=figsize)

    # Flatten the axes array for easier iteration
    flat_axes = axes.ravel()

    for ax, variant in zip(flat_axes, variants):
        subset = tail_df[tail_df["variant"] == variant]
        ax.hist(
            subset["reward"],
            bins=43,
            range=(-21, 22),
            alpha=0.7,
            color=color_map[variant],
            edgecolor="black",
        )
        ax.set_title(variant)
        ax.set_xlabel("Reward")
        ax.set_ylabel("Frequency")

    # if the number of experiments is odd, hide the last unused subplot
    if len(variants) % 2 == 1:
        flat_axes[-1].axis("off")

    # Set the main title for the entire figure
    fig.suptitle("Reward distributions for DQN Architectures", fontsize=24, y=0.99)

    plt.tight_layout()
    plt.savefig(out_file)
