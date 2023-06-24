import sys
from copy import deepcopy
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SMOOTH_WINDOW = 1


def analyze(log_file: Path) -> None:
    df = pd.read_csv(log_file)
    descr = df["reward"].describe()
    print(descr)


def plot_reward(df: pd.DataFrame, plot_file: Path, smooth: int | None = None) -> None:
    df = deepcopy(df)
    if smooth:
        df["reward"] = df["reward"].rolling(smooth).mean()

    # create a figure and a first axis for the reward
    _, ax1 = plt.subplots()

    # plot the mean reward and confidence intervals on the first y-axis
    sns.lineplot(data=df, x="episode", y="reward", hue="experiment", ax=ax1)

    # add a horizontal line at y=0
    ax1.axhline(0, color="grey", linestyle="--", linewidth=0.5)
    ax1.set_ylabel("Reward")

    # create a second y-axis fr epsilon, sharing the x-axis with the first one
    ax2 = ax1.twinx()

    # plot epsilon on the second y-axis, using the epsilon values of
    #  the first agent's first run as representative
    agent_df = df[(df["experiment"] == "experiment_0")]
    sns.lineplot(
        data=agent_df, x="episode", y="epsilon", color="green", ax=ax2, legend=False
    )
    ax2.set_ylabel("Epsilon")

    # get the handles and labels for all lines
    handles, labels = ax1.get_legend_handles_labels()

    # manually add the Epsilon label
    labels += ["Epsilon"]

    # create a new legend with all lines
    ax1.legend(handles=handles, labels=labels)

    plt.title(f"{'Smoothed' if smooth else ''} Reward and Epsilon over time")
    plt.savefig(plot_file)


def save_summary(df: pd.DataFrame, sum_file: Path) -> None:
    df["time_per_step"] = df["time"] / df["steps"]
    reward_descr = df["reward"].describe()
    time_per_step = df["time_per_step"].describe()
    file_content = str(reward_descr) + "\n" * 2 + str(time_per_step)
    sum_file.write_text(file_content)


def peek(dir_: Path) -> None:
    log_files = [f for f in dir_.rglob("*.csv") if f.is_file()]
    log_files.sort()

    all_data: list[pd.DataFrame] = []
    for i, f in enumerate(log_files):
        df = pd.read_csv(f)
        exp_name = f"experiment_{i}"
        df["experiment"] = exp_name
        save_summary(df, Path(dir_, exp_name + ".txt"))
        all_data.append(df)

    # combine all dataframes
    all_data = pd.concat(all_data, ignore_index=True)

    # plot reward
    plot_reward(all_data, dir_ / "reward_plot.svg", SMOOTH_WINDOW)


if __name__ == "__main__":
    result_dir = Path(sys.argv[1]) / "results"
    peek(result_dir)
