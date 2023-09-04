from copy import deepcopy
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from analysis.analyzer.utils.coloring import generate_color_mapping

EPSILON_COLOR: Final[str] = "#D95F02"
FIG_SIZE: Final[tuple[int, int]] = (12, 7)


def plot_reward(df: pd.DataFrame, plot_file: Path, smooth: int | None = None) -> None:
    df = deepcopy(df)
    if smooth:
        df.reset_index(drop=True, inplace=True)
        df["reward"] = (
            df["reward"].rolling(smooth, center=True, min_periods=smooth).mean()
        )
        df["reward"].fillna(method="bfill", inplace=True)
        df["reward"].fillna(method="ffill", inplace=True)

    # create a figure and a first axis for the reward
    _, ax1 = plt.subplots(figsize=FIG_SIZE)

    # create color map
    variants = df["variant"].unique()
    color_map = generate_color_mapping(variants)  # type:ignore

    # plot the mean reward and confidence intervals on the first y-axis
    sns.lineplot(
        data=df,
        x="episode",
        y="reward",
        hue="variant",
        ax=ax1,
        palette=color_map,
        linewidth=2,
    )

    # add a horizontal line at y=0
    ax1.axhline(0, color="grey", linestyle="--", linewidth=0.5)
    ax1.set_xlabel("episode", fontweight="bold")
    ax1.set_ylabel("reward", fontweight="bold")

    # create a second y-axis for epsilon, sharing the x-axis with the first one
    ax2 = ax1.twinx()

    # plot epsilon on the second y-axis, using the epsilon values of
    epsilons = df.drop_duplicates(subset=["episode", "epsilon"])
    sns.lineplot(
        data=epsilons,
        x="episode",
        y="epsilon",
        color=EPSILON_COLOR,
        ax=ax2,
        legend=False,  # type:ignore
        linewidth=3,
        errorbar=None,
    )
    ax2.set_ylabel("epsilon", fontweight="bold")

    # get the handles and labels for all lines
    handles_ax1, labels_ax1 = ax1.get_legend_handles_labels()

    # manually add the Epsilon label (workaround)
    epsilon_line = Line2D([0], [0], color=EPSILON_COLOR, lw=3)
    handles_ax1.append(epsilon_line)
    labels_ax1.append("epsilon")

    # create a new legend with all lines
    ax1.legend(
        handles=handles_ax1,
        labels=labels_ax1,
        prop={"size": 15},
        loc="lower left",
        bbox_to_anchor=(0, 0.1),
    )

    # create plot
    plt.title("Reward and Epsilon over Episodes", fontsize=20)
    if smooth:
        plt.suptitle(f"(Reward smoothed with window size {smooth})", fontsize=14)
    plt.savefig(plot_file)
