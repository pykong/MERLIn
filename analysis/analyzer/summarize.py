from pathlib import Path

import pandas as pd


def summarize(result_df: pd.DataFrame, tail: int, out_file: Path) -> None:
    """
    Summarize experimental results using the plateau phase.

    Args:
        result_df (pd.DataFrame): The frame holding the experimental data.
        tail (int): Tail of episodes to use, as an assumption of plateau.
        out_file (Path): The file path to export to.
    """
    # filter to get only the last X episodes of each run
    tail_df = result_df.groupby(["variant_id", "run_id"]).tail(tail)

    # reward: calculate mean and standard deviation for
    reward_metrics = tail_df.groupby("variant_id")["reward"].agg(["mean", "std"])
    reward_metrics.columns = ["mean_reward", "std_reward"]

    # steps: calculate mean and standard deviation for number of steps
    steps_metrics = tail_df.groupby("variant_id")["steps"].agg(["mean", "std"])
    steps_metrics.columns = ["mean_steps", "std_steps"]

    # join the two metrics dataframes
    combined_metrics = pd.concat([reward_metrics, steps_metrics], axis=1)

    # limit decimals for floating point columns
    combined_metrics = combined_metrics.round(2)

    # export to CSV
    combined_metrics.to_csv(out_file)
