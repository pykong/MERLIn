from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.analyzer.utils.coloring import generate_color_mapping

FIG_SIZE: Final[tuple[int, int]] = (12, 7)
plt.rcParams.update({"font.size": 17})


def plot_reward_distribution(data: pd.DataFrame, tail: int, out_dir: Path) -> None:
    """
    Plot the reward distribution of each experiment as violin plots.

    Args:
        data (pd.DataFrame): The frame holding the experimental data.
        tail (int): The number of episodes from the end to consider.
        out_dir (Path): The dir path to save the figure to.
    """
    # get the last X episodes
    tail_df = data.groupby(["variant", "run"]).tail(tail)

    # define colors for each variant
    variants = tail_df["variant"].unique()

    # create color map
    color_map = generate_color_mapping(variants)  # type:ignore

    # exclude 'random_walker' from the dataset
    tail_df = tail_df[tail_df["variant"] != "random_walker"]

    # set up the figure and axes
    plt.figure(figsize=FIG_SIZE)
    sns.violinplot(
        x="variant",
        y="reward",
        data=tail_df,
        palette=color_map,
        inner="quartile",
        width=1,
    )

    # set title and labels
    plt.title("Reward Distribution of DQN Architectures", fontsize=22)
    plt.xlabel("architecture", fontweight="bold")
    plt.ylabel("reward", fontweight="bold")
    plt.xticks(rotation=45)
    plt.ylim(-21, 21)
    plt.axhline(0, color="grey", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    # save the figure
    plt.savefig(out_dir / "reward_dist.svg")
