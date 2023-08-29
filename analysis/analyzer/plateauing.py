# %%
from pathlib import Path

import numpy as np  # type:ignore
import pandas as pd  # type:ignore
from analysis.provider.result_synthesizer import (
    EXPERIMENTS,
    NUM_EPISODES,
    RUNS_PER_EXPERIMENT,
    generate_synthetic_data,
)


def detect_plateau(
    data: pd.DataFrame, window_size: int = 100, threshold: float = 0.5
) -> pd.DataFrame:
    """
    Determine the episode at which each run plateaus.

    Args:
    - data: DataFrame holding synthetic data for all experiments.
    - window_size: Size of the window for rolling average.
    - threshold: Threshold for change in rolling average to consider as plateau.

    Returns:
    - DataFrame with episode of plateau for each run of each experiment.
    """
    plateau_data = []

    for (experiment, run), group in data.groupby(["experiment", "run"]):
        rolling_avg = group["reward"].rolling(window=window_size).mean()
        diff = rolling_avg.diff().abs()
        plateau_episode = diff[diff < threshold].index.min()
        plateau_data.append(
            {"experiment": experiment, "run": run, "plateau_episode": plateau_episode}
        )

    return pd.DataFrame(plateau_data)


def calculate_statistics(plateau_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the statistics for plateauing for each experiment.

    Args:
    - plateau_data: DataFrame with episode of plateau for each run.

    Returns:
    - DataFrame with average, standard deviation, and standard error of plateauing for each experiment.
    """
    stats = plateau_data.groupby("experiment")["plateau_episode"].agg(
        ["mean", "std", "count"]
    )
    # sem = standard error of the mean
    stats["sem"] = stats["std"] / np.sqrt(stats["count"])
    return stats[["mean", "std", "sem"]]


def export_statistics_to_csv(
    stats: pd.DataFrame, output_dir: Path, filename: str = "plateau_statistics.csv"
) -> None:
    """
    Export statistics to a CSV file.

    Args:
    - stats: DataFrame with statistics for each experiment.
    - filename: Name of the CSV file.
    """
    output_path = output_dir / filename
    stats.to_csv(output_path)


if __name__ == "__main__":
    all_data = generate_synthetic_data(EXPERIMENTS, RUNS_PER_EXPERIMENT, NUM_EPISODES)
    plateau_data = detect_plateau(all_data)
    plateau_stats = calculate_statistics(plateau_data)
    export_statistics_to_csv(plateau_stats, Path("../results"))

# %%
