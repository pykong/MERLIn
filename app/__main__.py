import json
import sys

sys.dont_write_bytecode = True

from pathlib import Path
from typing import Final, Iterable

from .config import Config
from .loop import loop

EXPERIMENT_DIR: Final[Path] = Path("experiments")
RESULTS_DIR: Final[Path] = Path("results")


def load_experiments() -> Iterable[Config]:
    files = EXPERIMENT_DIR.glob("*.json")
    for file in files:
        json_dict = json.loads(file.read_text())
        yield Config(**json_dict)


def train():
    for i, experiment in enumerate(load_experiments()):
        print(f"Conducting experiment with: {experiment}")  # TODO: Improve logging
        result_dir = RESULTS_DIR / str(i)  # TODO: Make file name more speaking
        loop(experiment, result_dir)


if __name__ == "__main__":
    train()
