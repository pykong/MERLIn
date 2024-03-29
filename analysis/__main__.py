import sys
from pathlib import Path

import pandas as pd

from analysis.analyzer.plot_reward import plot_reward
from analysis.analyzer.plot_reward_dist import plot_reward_distribution
from analysis.analyzer.reward_stats import export_reward_statistics
from analysis.provider.result_collector import collect_experiment_results
from analysis.provider.result_synthesizer import synthesize_experiment_results
from app.utils.file_utils import ensure_dirs, ensure_empty_dirs

# analysis parameters
SMOOTH_WINDOW = 5
TAIL_EPISODES: int = 1000

# parameters for result synthesis
VARIANTS = ["var_one", "var_two", "var_three", "var_four"]
RUN_COUNT = 3
EPISODE_COUNT = 5_000


def analyze(result_df: pd.DataFrame, result_dir: Path) -> None:
    anal_dir = result_dir / "analysis"
    ensure_empty_dirs(anal_dir)
    # run analyzers
    export_reward_statistics(result_df, TAIL_EPISODES, anal_dir)
    plot_reward_distribution(result_df, TAIL_EPISODES, anal_dir)
    plot_reward(result_df, anal_dir, tail=TAIL_EPISODES, smooth=SMOOTH_WINDOW)


def collect_and_analyze(result_dir: Path) -> None:
    result_df = collect_experiment_results(result_dir)
    analyze(result_df, result_dir)


def main() -> None:
    result_dir = Path(sys.argv[1])
    if "--simulate" in sys.argv:  # a poor man's CLI ;-)
        print("Simulating analysis with synthetic data.")
        ensure_dirs(result_dir)
        result_df = synthesize_experiment_results(VARIANTS, RUN_COUNT, EPISODE_COUNT)
    else:
        result_df = collect_experiment_results(result_dir)
    analyze(result_df, result_dir)


if __name__ == "__main__":
    main()
