# %%
from typing import Final

import matplotlib.cm as cm
import matplotlib.pyplot as plt  # type:ignore
import numpy as np  # type:ignore
import pandas as pd  # type:ignore

EXPERIMENT_ID: Final[str] = "synthetic_data"


def sigmoid(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Define custom sigmoid function.

    Args:
    - x: Input data.
    - a: Sigmoid's height.
    - b: Position of the inflection point.
    - c: Controls the curve's steepness.
    - d: Raises or lowers the sigmoid's position.

    Returns:
    - Sigmoid function result.
    """
    return a / (1.0 + np.exp(-c * (x - b))) + d


def synthesize_run_data(
    var_id: str, run: int, e: int, max_performance: float, inflection_point: int
) -> pd.DataFrame:
    """
    Generate synthetic data for a single experiment run.

    Args:
    - var_id: Variant ID.
    - run: Run number.
    - e: Number of datapoints (episodes).
    - max_performance: Maximal performance (height of plateau).
    - inflection_point: Episode number of the inflection point.

    Returns:
    - DataFrame holding synthetic data for the experiment run.
    """
    x = np.linspace(0, e, e)
    noise = np.random.normal(
        0, max_performance * 0.05, e
    )  # 5% of max performance as std deviation

    # Modify the sigmoid's parameters for the desired behavior
    y = sigmoid(x, max_performance + 21, inflection_point, 0.005, -21) + noise
    y = np.clip(y, -21, 21)

    # Create the DataFrame
    df = pd.DataFrame(
        {
            "episode": range(1, e + 1),
            "experiment_id": EXPERIMENT_ID,
            "variant_id": [var_id] * e,
            "run_id": [run] * e,
            "epsilon": 0.0,
            "reward": y,
            "loss": 0.0,
            "steps": 0,
            "time": 0.0,
        }
    )

    return df


def synthesize_experiment_results(
    experiments: list[str], runs: int, num_episodes: int
) -> pd.DataFrame:
    """
    Generate synthetic data for all experiment runs.

    Args:
        experiments (list[str]): The experiment ids.
        runs (int): Number of runs per experiment to generate.
        num_episodes (int): Number of episodes to generate per experiment.

    Returns:
        pd.DataFrame: The generated data.
    """
    all_data = pd.DataFrame()
    for i, exp in enumerate(experiments):
        max_performance = (i + 1) * 5 + 10  # Increasing from 15 to 30
        for j in range(runs):
            inflection_point = np.random.randint(1500, num_episodes - 500)
            df = synthesize_run_data(
                exp, j + 1, num_episodes, max_performance, inflection_point
            )
            all_data = pd.concat([all_data, df], ignore_index=True)
    return all_data


# %%
