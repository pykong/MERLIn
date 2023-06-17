import json
import sys
from dataclasses import asdict

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


def save_experiment(config: Config, file_path: Path) -> None:
    with open(file_path, "w") as f:
        json.dump(asdict(config), f, indent=4)


def train():
    for i, experiment in enumerate(load_experiments()):
        print(f"Conducting experiment with: {experiment}")  # TODO: Improve logging
        result_dir = RESULTS_DIR / str(i)  # TODO: Make file name more speaking
        save_experiment(experiment, result_dir / "experiment.json")  # save parameters
        loop(experiment, result_dir)
        break
    else:
        raise ValueError("No experiments given. Exiting.")


if __name__ == "__main__":
    train()
