import sys
from copy import deepcopy
from pathlib import Path

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


def save_reward_histogram(df: pd.DataFrame, output_file: str) -> None:
    # Ensure that the 'reward' and 'epsilon' columns exist
    assert "reward" in df.columns, "DataFrame must have a 'reward' column"
    assert "epsilon" in df.columns, "DataFrame must have an 'epsilon' column"

    # Calculate min_epsilon
    min_epsilon = df["epsilon"].min()

    # Filter out the exploration phase
    df = df[df["epsilon"] <= min_epsilon]

    # Create the histogram
    plt.hist(df["reward"], bins=20, edgecolor="black")
    plt.title("Histogram of Rewards")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")

    # Save the histogram to a file
    output_file_path = Path(output_file)
    plt.savefig(output_file_path)

    print(f"Histogram has been saved to {output_file_path.absolute()}")

    # Clear the current figure
    plt.clf()


def save_summary(df: pd.DataFrame, sum_file: Path) -> None:
    df["time_per_step"] = df["time"] / df["steps"]
    reward_descr = df["reward"].describe()
    time_per_step = df["time_per_step"].describe()
    file_content = str(reward_descr) + "\n" * 2 + str(time_per_step)
    sum_file.write_text(file_content)


def calculate_and_write_win_rate(df: pd.DataFrame, output_file: str) -> None:
    # Ensure that the 'reward' and 'epsilon' columns exist
    assert "reward" in df.columns, "DataFrame must have a 'reward' column"
    assert "epsilon" in df.columns, "DataFrame must have an 'epsilon' column"

    # Calculate min_epsilon
    min_epsilon = df["epsilon"].min()

    # Filter out the exploration phase
    df = df[df["epsilon"] <= min_epsilon]

    # Calculate win rate
    total_episodes = len(df)
    wins = len(df[df["reward"] > 0])
    win_rate = wins / total_episodes if total_episodes > 0 else 0

    # Convert win rate to percentage
    win_rate_percent = win_rate * 100

    # Prepare the output string
    output_str = "\n".join(
        (
            f"Total Episodes : {total_episodes}",
            f"Wins           : {wins}",
            f"Win Rate       : {win_rate:.2f} ({win_rate_percent:.2f}%)",
        )
    )

    # Write to the file
    output_file_path = Path(output_file)
    output_file_path.write_text(output_str)

    print(f"Win rate has been written to {output_file_path.absolute()}")


def peek(dir_: Path) -> None:
    log_files = [f for f in dir_.rglob("*.csv") if f.is_file()]
    log_files.sort()

    all_data: list[pd.DataFrame] = []
    for i, f in enumerate(log_files):
        df = pd.read_csv(f)
        exp_name = f"experiment_{i}"
        df["experiment"] = exp_name
        save_summary(df, Path(dir_, exp_name + ".txt"))
        calculate_and_write_win_rate(df, Path(dir_, exp_name + "_winrate" + ".txt"))
        save_reward_histogram(df, Path(dir_, exp_name + "_distribution" + ".svg"))
        plot_reward(df, Path(dir_, exp_name + ".svg"))
        all_data.append(df)

    # combine all dataframes
    all_data = pd.concat(all_data, ignore_index=True)

    # plot reward
    plot_reward(all_data, dir_ / "reward_plot.svg", SMOOTH_WINDOW)


def analyze_results():
    result_dir = Path(sys.argv[1])  # point to result dir
    peek(result_dir)
