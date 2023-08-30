from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_reward_distribution(data: pd.DataFrame, out_file: Path) -> None:
    """
    Plot the reward distribution of each experiment as stacked histograms.

    Args:
        data (pd.DataFrame): The frame holding the experimental data.
    """
    variants = data["variant_id"].unique()

    figsize = (10, len(variants) * 5)
    _, axes = plt.subplots(len(variants), 1, sharex=True, figsize=figsize)

    for ax, variant in zip(axes, variants):
        subset = data[data["variant_id"] == variant]
        ax.hist(
            subset["reward"],
            bins=43,
            range=(-21, 22),
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )
        ax.set_title(f"Reward Distribution for {variant}")
        ax.set_ylabel("Frequency")

    plt.xlabel("Reward")
    plt.tight_layout()
    plt.savefig(out_file)
