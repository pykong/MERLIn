import sys
from pathlib import Path

import pandas as pd
from analysis.provider.result_collector import collect_experiment_results
from analysis.provider.result_synthesizer import synthesize_experiment_results
from app.utils.file_utils import ensure_dirs

# parameters for data synthesis
VARIANTS = ["var_one", "var_two", "var_three", "var_four"]
RUN_COUNT = 3
EPISODE_COUNT = 5_000


def analyze(result_df: pd.DataFrame, result_dir: Path) -> None:
    print(result_df.head())


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
