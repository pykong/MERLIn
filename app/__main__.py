import json
import sys

sys.dont_write_bytecode = True

from pathlib import Path
from typing import Final, Iterable

from .config import Config
from .loop import CHECKPOINTS_DIR, IMG_DIR, LOG_DIR, VIDEO_DIR, ensure_empty_dirs, loop

EXPERIMENT_DIR: Final[Path] = Path("experiments")


def load_experiments() -> Iterable[Config]:
    files = EXPERIMENT_DIR.glob("*.json")
    for file in files:
        json_dict = json.loads(file.read_text())
        yield Config(**json_dict)


def train():
    ensure_empty_dirs(CHECKPOINTS_DIR, LOG_DIR, VIDEO_DIR, IMG_DIR)
    for experiment in load_experiments():
        print(f"Conducting experiment with: {experiment}")
        loop(experiment)


if __name__ == "__main__":
    train()
