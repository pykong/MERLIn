import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from app.utils.file_utils import ensure_empty_dirs

SMOOTH_WINDOW = 25
SKIP_FRAMES = 3500  # ignore exploration and expontential phase


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
    sns.lineplot(
        data=df[(df["experiment"] == "experiment_0")],
        x="episode",
        y="epsilon",
        color="green",
        ax=ax2,
        legend=False,  # type:ignore
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


def plot_reward_histogram(df: pd.DataFrame, plot_file: Path) -> None:
    df = deepcopy(df)

    # Filter out the exploration and exponential phase
    df = df.iloc[SKIP_FRAMES:]

    # Create the histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df,
        x="reward",
        hue="experiment",
        element="step",
        stat="density",
        common_norm=False,
    )
    plt.title("Histogram of Rewards")
    plt.xlabel("Reward")
    plt.ylabel("Density")

    # Save the histogram to a file
    plt.savefig(plot_file)

    print(f"Histogram has been saved to {plot_file.absolute()}")

    # Clear the current figure
    plt.clf()


def calculate_and_write_win_rate(df: pd.DataFrame) -> str:
    # Ensure that the 'reward' and 'epsilon' columns exist
    assert "reward" in df.columns, "DataFrame must have a 'reward' column"
    assert "epsilon" in df.columns, "DataFrame must have an 'epsilon' column"

    # Filter out the exploration and exponential phase
    df = df.iloc[SKIP_FRAMES:]

    # Calculate win rate
    total_episodes = len(df)
    wins = len(df[df["reward"] > 0])
    win_rate = wins / total_episodes if total_episodes > 0 else 0

    # Convert win rate to percentage
    win_rate_percent = win_rate * 100

    # Prepare the output string
    return "\n".join(
        (
            "\n\n" + "-" * 7 + "Win-Rate" + "-" * 7,
            f"Total Episodes : {total_episodes}",
            f"Wins           : {wins}",
            f"Win Rate       : {win_rate:.2f} ({win_rate_percent:.2f}%)",
        )
    )


def summarize_rl_data(df: pd.DataFrame, output_path: Path):
    df = deepcopy(df)

    def skip_frames(group):
        return group.iloc[SKIP_FRAMES:]

    df = df.groupby("experiment").apply(skip_frames).reset_index(drop=True)

    summary_df = (
        df.groupby("experiment")
        .agg(
            {
                "reward": ["count", "mean", "std", "max"],
                "steps": ["mean", "std", "max"],
                "time": "sum",
            }
        )
        .reset_index()
    )

    # rename columns
    summary_df.columns = [
        "experiment",
        "episodes",
        "mean_reward",
        "std_reward",
        "max_reward",
        "mean_steps",
        "std_steps",
        "max_steps",
        "total_time",
    ]

    # compute wins
    wins = (
        df[df["reward"] > 0]
        .groupby("experiment")["reward"]
        .count()
        .reset_index()
        .rename(columns={"reward": "wins"})
    )
    summary_df = summary_df.merge(wins, on="experiment", how="left")

    # fill NaN values in 'wins' with 0 and calculate 'win_rate'
    summary_df["wins"] = summary_df["wins"].fillna(0)
    summary_df["win_rate"] = (summary_df["wins"] / summary_df["episodes"]) * 100

    # compute mean_time_per_step
    summary_df["mean_time_per_step"] = (
        summary_df["total_time"] / df.groupby("experiment")["steps"].sum().values
    )

    # round numerical columns for better readability
    numerical_cols = [
        "mean_reward",
        "std_reward",
        "max_reward",
        "mean_steps",
        "std_steps",
        "max_steps",
        "total_time",
        "win_rate",
    ]
    summary_df[numerical_cols] = summary_df[numerical_cols].round(2)
    summary_df["mean_time_per_step"] = summary_df["mean_time_per_step"].round(5)

    # write summary dataframe to CSV file
    summary_df.to_csv(output_path, index=False)


def peek(dir_: Path) -> None:
    # create sub-dirs
    analysis_dir = dir_ / "analysis"
    reward_dir = analysis_dir / "reward"
    reward_dist_dir = analysis_dir / "reward_dist"
    ensure_empty_dirs(analysis_dir, reward_dir, reward_dist_dir)

    # glob log files
    log_files = [
        f for f in dir_.rglob("*.csv") if f.is_file() and f.parent is not analysis_dir
    ]
    log_files.sort()

    # analyse individual variants
    all_frames: list[pd.DataFrame] = []
    for i, f in enumerate(log_files):
        df = pd.read_csv(f)
        exp_name = f"experiment_{i}"
        df["experiment"] = exp_name
        plot_reward_histogram(df, Path(reward_dist_dir, exp_name + "_dist" + ".svg"))
        plot_reward(df, Path(reward_dir, exp_name + ".svg"))
        all_frames.append(df)

    # combine all dataframes
    merged_frame = pd.concat(all_frames, ignore_index=True)

    # plot reward
    plot_reward(merged_frame, analysis_dir / "all_rewards.svg", SMOOTH_WINDOW)
    plot_reward_histogram(merged_frame, analysis_dir / "all_dist.svg")
    summarize_rl_data(merged_frame, analysis_dir / "summary.csv")


def analyze_results():
    result_dir = Path(sys.argv[1])  # point to result dir
    peek(result_dir)
