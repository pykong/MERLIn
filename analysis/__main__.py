import sys
from pathlib import Path

from analysis.provider.result_collector import collect_experiment_results
from analysis.provider.result_synthesizer import synthesize_experiment_results

# parameters for data synthesis
VARIANTS = ["var_one", "var_two", "var_three", "var_four"]
RUN_COUNT = 3
EPISODE_COUNT = 5_000


def main():
    result_dir = Path(sys.argv[1])  # point to result dir
    result_df = collect_experiment_results(result_dir)
    print(result_df.head())

    syn_result_df = synthesize_experiment_results(VARIANTS, RUN_COUNT, EPISODE_COUNT)
    print(syn_result_df.head())


if __name__ == "__main__":
    main()
