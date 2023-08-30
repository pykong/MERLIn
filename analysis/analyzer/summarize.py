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
    # calculate run metrics
    run_metrics = []
    for (variant, run), group in result_df.groupby(["variant_id", "run_id"]):
        tail_data = group.tail(tail)
        run_metrics.append(
            {
                "variant_id": variant,
                "run_id": run,
                "mean_reward": tail_data["reward"].mean(),
                "max_reward": tail_data["reward"].max(),
            }
        )
    run_metrics = pd.DataFrame(run_metrics)

    # calculate variant metrics
    variant_metrics = []
    for (variant), group in run_metrics.groupby(["variant_id"]):
        print(group.head())
        variant_metrics.append(
            {
                "variant_id": variant[0],
                "mean_reward": group["mean_reward"].mean(),
                "max_reward": group["max_reward"].mean(),
            }
        )
    pd.DataFrame(variant_metrics).to_csv(out_file)
