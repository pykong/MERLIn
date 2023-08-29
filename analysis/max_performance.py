# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
from analysis.data_generator import (
    EXPERIMENTS,
    NUM_EPISODES,
    RUNS_PER_EXPERIMENT,
    generate_synthetic_data,
)
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def get_top_runs(data: pd.DataFrame) -> pd.DataFrame:
    """
    Get the top run for each experiment based on mean reward over the final 2000 episodes.

    Args:
    - data: DataFrame holding synthetic data for all experiments.

    Returns:
    - DataFrame with top runs for each experiment.
    """
    top_runs = []

    for experiment, group in data.groupby("experiment"):
        # Compute mean reward over final 2000 episodes for each run
        means = group.groupby("run").apply(lambda x: x["reward"].tail(2000).mean())
        top_run = means.idxmax()
        top_runs.append(group[group["run"] == top_run])

    return pd.concat(top_runs)


def perform_anova(data: pd.DataFrame) -> float:
    """
    Perform one-way ANOVA on the data.

    Args:
    - data: DataFrame holding synthetic data for top runs.

    Returns:
    - p-value from ANOVA.
    """
    groups = [
        data["reward"][data["experiment"] == exp] for exp in data["experiment"].unique()
    ]
    _, p = stats.f_oneway(*groups)
    return p


def tukeys_hsd(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform Tukey's HSD test on the data.

    Args:
    - data: DataFrame holding synthetic data for top runs.

    Returns:
    - DataFrame with pairwise comparison results.
    """
    result = pairwise_tukeyhsd(data["reward"], data["experiment"])
    return pd.DataFrame(result.summary().data[1:], columns=result.summary().data[0])


if __name__ == "__main__":
    all_data = generate_synthetic_data(EXPERIMENTS, RUNS_PER_EXPERIMENT, NUM_EPISODES)
    # Isolate top runs
    top_runs_data = get_top_runs(all_data)

    # Central Performance Metric
    mean_rewards = top_runs_data.groupby("experiment")["reward"].mean()
    std_dev_rewards = top_runs_data.groupby("experiment")["reward"].std()
    print("Mean Rewards:", mean_rewards)
    print("Standard Deviations:", std_dev_rewards)

    # One-way ANOVA
    p_value = perform_anova(top_runs_data)
    print(f"ANOVA p-value: {p_value}")

    # Tukey's HSD (if ANOVA suggests significant differences)
    if p_value < 0.05:
        print("Performing Tukey's HSD Test...")
        tukey_result = tukeys_hsd(top_runs_data)
        print(tukey_result)

# %%
