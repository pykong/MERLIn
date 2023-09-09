from pathlib import Path

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


def export_reward_statistics(
    result_df: pd.DataFrame, tail: int, out_file: Path
) -> None:
    """
    Calculate and export statistics on the reward of variants.

    Args:
        result_df (pd.DataFrame): The input DataFrame containing results.
        tail (int): The number of last records to consider as an plateau assumption.
        out_file (Path): The path of the CSV file to export the results to.
    """
    tail_df = result_df.groupby(["variant", "run"]).tail(tail)

    reward_metrics = tail_df.groupby("variant")["reward"].agg(["mean", "std", "count"])

    ci_df = calculate_ci(reward_metrics, "mean", "std", "count")
    reward_metrics = pd.concat([reward_metrics, ci_df], axis=1)

    # rank based on the lower bound of the confidence interval
    ranked_results = reward_metrics.sort_values("ci_lower", ascending=False)
    ranked_results = ranked_results.drop(columns="count")
    ranked_results = ranked_results.round(2)
    ranked_results.to_csv(out_file)
