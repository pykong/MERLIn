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
        df["reward"] = df["reward"].rolling(smooth).mean()

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
    ax1.set_ylabel("reward")

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
        linewidth=2,
    )
    ax2.set_ylabel("epsilon")

    # get the handles and labels for all lines
    handles_ax1, labels_ax1 = ax1.get_legend_handles_labels()

    # manually add the Epsilon label (workaround)
    epsilon_line = Line2D([0], [0], color=EPSILON_COLOR, lw=1)
    handles_ax1.append(epsilon_line)
    labels_ax1.append("epsilon")

    # create a new legend with all lines
    ax1.legend(handles=handles_ax1, labels=labels_ax1)

    # create plot
    # plt.figure(figsize=(9, 5))
    plt.title(f"{'Smoothed' if smooth else ''} Reward and Epsilon over Episodes")
    plt.savefig(plot_file)
