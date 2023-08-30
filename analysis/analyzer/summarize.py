from pathlib import Path

import pandas as pd
import scipy.stats as stats


def summarize(result_df: pd.DataFrame, tail: int, out_file: Path) -> None:
    """
    Summarize and rank experimental results using the plateau phase.

    Args:
        result_df (pd.DataFrame): The frame holding the experimental data.
        tail (int): Tail of episodes to use, as an assumption of plateau.
        out_file (Path): The file path to export to.
    """
    # filter to get only the last X episodes of each run
    tail_df = result_df.groupby(["variant_id", "run_id"]).tail(tail)

    # reward: calculate mean and standard deviation for
    reward_metrics = tail_df.groupby("variant_id")["reward"].agg(
        ["mean", "std", "count"]
    )
    reward_metrics.columns = ["mean_reward", "std_reward", "count"]

    # steps: calculate mean and standard deviation for number of steps
    steps_metrics = tail_df.groupby("variant_id")["steps"].agg(["mean", "std"])
    steps_metrics.columns = ["mean_steps", "std_steps"]

    # compute the standard error (SE) for each variant for reward
    reward_metrics["se"] = reward_metrics["std_reward"] / reward_metrics["count"] ** 0.5

    # calculate the t-value for a 95% confidence interval for reward
    reward_metrics["t_value"] = reward_metrics.apply(
        lambda row: stats.t.ppf(0.975, df=row["count"] - 1), axis=1
    )

    # calculate the lower bound of the 95% confidence interval for the mean reward
    reward_metrics["ci_lower_mean_reward"] = (
        reward_metrics["mean_reward"] - reward_metrics["t_value"] * reward_metrics["se"]
    )

    # rank variants based on the lower bound of the confidence interval
    reward_metrics = reward_metrics.sort_values("ci_lower_mean_reward", ascending=False)

    # join the metrics dataframes
    combined_metrics = pd.concat([reward_metrics, steps_metrics], axis=1)

    # drop intermediate columns used for calculations
    columns_to_keep = [
        "mean_reward",
        "std_reward",
        "mean_steps",
        "std_steps",
        "ci_lower_mean_reward",
    ]
    # and limit decimals for floating point columns
    combined_metrics = combined_metrics[columns_to_keep].round(2)

    # export to CSV
    combined_metrics.to_csv(out_file)
