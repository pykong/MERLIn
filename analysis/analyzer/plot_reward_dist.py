from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_reward_distribution(data: pd.DataFrame, tail: int, out_file: Path) -> None:
    """
    Plot the reward distribution of each experiment as stacked histograms.

    Args:
        data (pd.DataFrame): The frame holding the experimental data.
    """
    variants = data["variant_id"].unique()

    tail_df = data.groupby(["variant_id", "run_id"]).tail(tail)

    colors = plt.cm.viridis_r(np.linspace(0, 1, len(variants)))  # type:ignore

    # Calculate number of rows for the 2-column layout
    nrows = -(-len(variants) // 2)  # equivalent to math.ceil(len(variants) / 2)
    figsize = (15, nrows * 5)
    fig, axes = plt.subplots(nrows, 2, sharex=True, figsize=figsize)

    # Flatten the axes array for easier iteration
    flat_axes = axes.ravel()

    for ax, variant, color in zip(flat_axes, variants, colors):
        subset = tail_df[tail_df["variant_id"] == variant]
        ax.hist(
            subset["reward"],
            bins=43,
            range=(-21, 22),
            alpha=0.7,
            color=color,
            edgecolor="black",
        )
        ax.set_title(f"Reward Distribution for {variant}")
        ax.set_ylabel("Frequency")

    # If the number of experiments is odd, hide the last unused subplot
    if len(variants) % 2 == 1:
        flat_axes[-1].axis("off")

    plt.xlabel("Reward")
    plt.tight_layout()
    plt.savefig(out_file)
