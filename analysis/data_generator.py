# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Parameters
EXPERIMENTS = ["exp_one", "exp_two", "exp_three", "exp_four"]
RUNS_PER_EXPERIMENT = 3
NUM_EPISODES = 5_000


def sigmoid(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Custom sigmoid function.

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


def generate_synthetic_data(
    e: int, max_performance: float, inflection_point: int
) -> np.ndarray:
    """
    Generate synthetic data for an experiment.

    Args:
    - e: Number of datapoints (episodes).
    - max_performance: Maximal performance (height of plateau).
    - inflection_point: Episode number of the inflection point.

    Returns:
    - Synthetic data for the experiment.
    """
    x = np.linspace(0, e, e)
    noise = np.random.normal(0, max_performance * 0.05, e)
    y = sigmoid(x, max_performance + 21, inflection_point, 0.005, -21) + noise
    y = np.clip(y, -21, 21)
    return y


def plot_experiments(experiments: list, r: int, e: int):
    """
    Plot the generated synthetic data for all experiments.

    Args:
    - experiments: List of experiment ids.
    - r: Number of times each experiment is run.
    - e: Number of datapoints (episodes).
    """
    colors = ["red", "green", "blue", "purple"]
    for i, exp in enumerate(experiments):
        max_performance = (i + 1) * 5 + 10  # Increasing from 15 to 30
        for j in range(r):
            inflection_point = np.random.randint(1500, e - 500)
            y = generate_synthetic_data(e, max_performance, inflection_point)
            plt.plot(y, label=f"{exp}_run_{j+1}", color=colors[i], alpha=0.5)
    plt.legend(loc="upper left")
    plt.title("Synthetic Reinforcement Learning Experiments")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.show()


if __name__ == "__main__":
    plot_experiments(EXPERIMENTS, RUNS_PER_EXPERIMENT, NUM_EPISODES)

# %%
