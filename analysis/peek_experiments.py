import sys
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def analyze(log_file: Path) -> None:
    df = pd.read_csv(log_file)
    descr = df["reward"].describe()
    print(descr)


def peek(dir: Path) -> None:
    log_files = [f for f in dir.rglob("*.csv") if f.is_file()]
    log_files.sort()

    all_data: list[pd.DataFrame] = []
    for i, f in enumerate(log_files):
        df = pd.read_csv(f)
        exp_name = f"experiment_{i}"
        df["experiment"] = exp_name
        reward_descr = df["reward"].describe()
        Path(dir, "reward_" + exp_name + ".txt").write_text(str(reward_descr))
        all_data.append(df)

    # combine all dataframes
    all_data = pd.concat(all_data, ignore_index=True)

    # create a figure and a first axis for the reward
    fig, ax1 = plt.subplots()

    # plot the mean reward and confidence intervals on the first y-axis
    sns.lineplot(data=all_data, x="episode", y="reward", hue="experiment", ax=ax1)

    # add a horizontal line at y=0
    ax1.axhline(0, color="grey", linestyle="--", linewidth=0.5)
    ax1.set_ylabel("Reward")

    # create a second y-axis for epsilon, sharing the x-axis with the first one
    ax2 = ax1.twinx()

    # plot epsilon on the second y-axis, using the epsilon values of
    #  the first agent's first run as representative
    agent_df = all_data[(all_data["experiment"] == "experiment_0")]
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

    plt.title("Reward and Epsilon over time")
    plt.savefig(dir / "reward_plot.svg")


if __name__ == "__main__":
    result_dir = Path(sys.argv[1]) / "results"
    peek(result_dir)
