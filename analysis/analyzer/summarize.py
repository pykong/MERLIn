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
    df = result_df.tail(tail)
