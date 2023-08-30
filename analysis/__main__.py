import sys
from pathlib import Path

from analysis.provider.result_collector import collect_experiment_results


def main():
    result_dir = Path(sys.argv[1])  # point to result dir
    result_df = collect_experiment_results(result_dir)
    print(result_df.head())


if __name__ == "__main__":
    main()
