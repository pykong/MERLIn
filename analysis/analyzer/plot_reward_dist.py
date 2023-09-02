from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.analyzer.utils.coloring import generate_color_mapping


def plot_reward_distribution(data: pd.DataFrame, tail: int, out_file: Path) -> None:
    """
    Plot the reward distribution of each experiment as violin plots.

    Args:
        data (pd.DataFrame): The frame holding the experimental data.
        tail (int): The number of episodes from the end to consider.
        out_file (Path): The file path to save the figure.
    """
    # get the last X episodes
    tail_df = data.groupby(["variant", "run"]).tail(tail)

    # define colors for each variant
    variants = tail_df["variant"].unique()

    # create color map
    color_map = generate_color_mapping(variants)  # type:ignore

    # set up the figure and axes
    plt.figure(figsize=(12, 8))
    sns.violinplot(
        x="variant", y="reward", data=tail_df, palette=color_map, inner="quartile"
    )

    # set title and labels
    plt.title("Reward distributions for DQN Architecture")
    plt.xlabel("Experiment")
    plt.ylabel("Reward")
    plt.xticks(rotation=45)
    plt.ylim(-21, 21)
    plt.axhline(0, color="grey", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    # save the figure
    plt.savefig(out_file)
