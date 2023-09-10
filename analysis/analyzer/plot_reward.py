from copy import deepcopy
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from analysis.analyzer.utils.coloring import generate_color_mapping
from matplotlib.lines import Line2D

EPSILON_COLOR: Final[str] = "#D95F02"
FIG_SIZE: Final[tuple[int, int]] = (12, 7)
plt.rcParams.update({"font.size": 17})


def plot_reward(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    tail: int | None = None,
    smooth: int | None = None,
) -> None:
    """
    Plot the rewards from reinforcement learning experiments.

    The function plots the rewards over episodes and can also highlight the tail-end episodes used for statistical evaluation. Optionally, the reward values can be smoothed for clarity.

    Args:
        df (pd.DataFrame): Data containing 'reward' and 'episode' columns.
        out_dir (Path): The dir path to save the figure to.
        tail (int?): Number of tail-end episodes used for statistical evaluation.
                              A gray rectangle will be drawn to indicate this span.
                              Defaults to None (no rectangle drawn).
        smooth (int?): Window size for reward smoothing. If provided, rewards
                                will be smoothed using a rolling centered mean. Defaults to None (no smoothing).
    """
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

    # add a gray rectangle for evaluation episodes
    if tail:
        last_episode = df["episode"].max()
        ax1.axvspan(
            last_episode - tail,
            last_episode,
            color="grey",
            alpha=0.2,
            label="evaluation",
            zorder=0,
        )

    # add a horizontal line at y=0
    ax1.axhline(0, color="grey", linestyle="--", linewidth=0.5)
    ax1.set_xlabel("episode", fontweight="bold")
    ax1.set_ylabel("reward", fontweight="bold")

    # create a second y-axis for epsilon, sharing the x-axis with the first one
    ax2 = ax1.twinx()
    ax2.set_ylabel("epsilon", fontweight="bold")

    first_variant = df["variant"].iloc[0]
    eps_df = df[df["variant"] == first_variant]  # assume single epsilon regimen
    eps_df = eps_df[["episode", "epsilon"]]
    eps_df["epsilon"] = eps_df["epsilon"].round(2)

    sns.lineplot(
        data=eps_df,
        x="episode",
        y="epsilon",
        color=EPSILON_COLOR,
        ax=ax2,
        legend=False,  # type:ignore
        linewidth=3,
        errorbar=None,  # to avoid error bars
    )

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
    plt.title("Reward and Epsilon over Episodes", fontsize=22)
    if smooth:
        plt.suptitle(f"(Reward smoothed with window size {smooth})", fontsize=16)
    plt.savefig(out_dir / "reward.svg")
