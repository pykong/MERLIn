from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats


def calculate_ci(
    df: pd.DataFrame, mean_col: str, std_col: str, count_col: str
) -> pd.DataFrame:
    """
    Calculate the 95% confidence interval.

    Args:
        df (pd.DataFrame): The dataframe holding the data.
        mean_col (str): Column name for the mean values.
        std_col (str): Column name for the standard deviation values.
        count_col (str): Column name for the count values.

    Returns:
        pd.DataFrame: DataFrame with the bounds of the 95% confidence interval.
    """
    se = df[std_col] / df[count_col] ** 0.5
    t_value = stats.t.ppf(0.975, df=df[count_col] - 1)
    ci_upper = df[mean_col] + t_value * se
    ci_lower = df[mean_col] - t_value * se
    return pd.DataFrame({"ci_upper": ci_upper, "ci_lower": ci_lower})


def pairwise_mannwhitneyu(tail_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Perform pairwise Mann-Whitney U tests between all variants and export to CSV.

    Args:
        tail_df (pd.DataFrame): The dataframe holding the data.
        tail (int): Number of episodes to consider from the tail for each run.
    """
    variants = sorted(tail_df["variant"].unique())
    n_variants = len(variants)
    p_value_matrix = np.full((n_variants, n_variants), np.nan)

    for i in range(n_variants):
        for j in range(i + 1, n_variants):
            # i+1 to skip diagonal and duplicates (since it's symmetric)
            sample1 = tail_df[tail_df["variant"] == variants[i]]["reward"]
            sample2 = tail_df[tail_df["variant"] == variants[j]]["reward"]
            _, p_value = stats.mannwhitneyu(sample1, sample2, alternative="two-sided")
            # apply Bonferroni correction and fill the symmetric matrix
            corrected_p_value = min(p_value * n_variants * (n_variants - 1) / 2, 1)
            p_value_matrix[i, j] = corrected_p_value
            p_value_matrix[j, i] = corrected_p_value

    # Apply Bonferroni correction
    comparison_matrix = pd.DataFrame(p_value_matrix, index=variants, columns=variants)
    comparison_matrix.to_csv(out_dir / "reward_pairwise.csv")


def export_reward_statistics(result_df: pd.DataFrame, tail: int, out_dir: Path) -> None:
    """
    Calculate and export statistics on the reward of variants.

    Args:
        result_df (pd.DataFrame): The input DataFrame containing results.
        tail (int): The number of last records to consider as an plateau assumption.
        out_dir (Path): The path of the CSV file to export the results to.
    """
    tail_df = result_df.groupby(["variant", "run"]).tail(tail)
    reward_metrics = tail_df.groupby("variant")["reward"].agg(["mean", "std", "count"])
    ci_df = calculate_ci(reward_metrics, "mean", "std", "count")
    reward_metrics = pd.concat([reward_metrics, ci_df], axis=1)

    # rank based on the lower bound of the confidence interval
    ranked_results = reward_metrics.sort_values("ci_lower", ascending=False)
    ranked_results = ranked_results.drop(columns="count")
    ranked_results = ranked_results.round(2)
    ranked_results.to_csv(out_dir / "reward_stats.csv")

    # calculate and export test for statistically significance
    pairwise_mannwhitneyu(tail_df, out_dir)
