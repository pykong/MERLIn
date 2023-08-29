# %%
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.provider.result_synthesizer import (
    EXPERIMENTS,
    NUM_EPISODES,
    RUNS_PER_EXPERIMENT,
    generate_synthetic_data,
)


def calculate_std_dev(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate standard deviation of rewards for each episode across different runs of each experiment.

    Args:
    - data: DataFrame holding synthetic data for all experiments.

    Returns:
    - DataFrame with standard deviations for each episode.
    """
    return data.groupby(["experiment", "episode"])["reward"].std().reset_index()


def print_avg_std_dev(std_dev_data: pd.DataFrame):
    """
    Print average standard deviation for each experiment.

    Args:
    - std_dev_data: DataFrame with standard deviations for each episode.
    """
    avg_std_dev = std_dev_data.groupby("experiment")["reward"].mean()
    for experiment, std_dev in avg_std_dev.items():
        print(f"Avg Std Dev for {experiment}: {std_dev:.2f}")


def export_to_csv(
    std_dev_data: pd.DataFrame,
    output_dir: Path,
    filename: str = "standard_deviations.csv",
):
    """
    Export standard deviations to a CSV file.

    Args:
    - std_dev_data: DataFrame with standard deviations for each episode.
    - filename: Name of the CSV file.
    """
    output_path = output_dir / filename
    std_dev_data.to_csv(output_path, index=False)


def plot_std_dev(std_dev_data: pd.DataFrame):
    """
    Plot standard deviations for each experiment.

    Args:
    - std_dev_data: DataFrame with standard deviations for each episode.
    """
    experiments = std_dev_data["experiment"].unique()
    colors = cm.rainbow(np.linspace(0, 1, len(experiments)))

    for i, exp in enumerate(experiments):
        exp_data = std_dev_data[std_dev_data["experiment"] == exp]
        plt.plot(
            exp_data["episode"],
            exp_data["reward"],
            label=exp,
            color=colors[i],
            alpha=0.7,
        )

    plt.title("Standard Deviation of Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Standard Deviation")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    all_data = generate_synthetic_data(EXPERIMENTS, RUNS_PER_EXPERIMENT, NUM_EPISODES)
    std_dev_data = calculate_std_dev(all_data)
    print_avg_std_dev(std_dev_data)
    export_to_csv(std_dev_data, Path("../results"))
    plot_std_dev(std_dev_data)

# %%
