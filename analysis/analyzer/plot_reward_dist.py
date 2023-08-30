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

    colors = plt.cm.viridis_r(np.linspace(0, 1, len(variants)))

    figsize = (10, len(variants) * 5)
    _, axes = plt.subplots(len(variants), 1, sharex=True, figsize=figsize)

    for ax, experiment, color in zip(axes, variants, colors):
        subset = tail_df[tail_df["variant_id"] == experiment]
        ax.hist(
            subset["reward"],
            bins=43,
            range=(-21, 22),
            alpha=0.7,
            color=color,
            edgecolor="black",
        )
        ax.set_title(f"Reward Distribution for {experiment}")
        ax.set_ylabel("Frequency")

    plt.xlabel("Reward")
    plt.tight_layout()
    plt.savefig(out_file)
