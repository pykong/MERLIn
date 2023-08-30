from pathlib import Path
from typing import Final

import pandas as pd

LOG_FILE: Final[str] = "train_log.csv"


def collect_experiment_results(result_dir: Path) -> pd.DataFrame:
    """Return all training results as a single data frame.

    Args:
        result_dir (Path): Path to experiment dir.

    Returns:
        pd.DataFrame: The training results.
    """
    frames = [pd.read_csv(f) for f in result_dir.rglob(LOG_FILE)]
    return pd.concat(frames)
