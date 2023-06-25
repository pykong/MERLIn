import json
import pprint
import sys
from dataclasses import asdict

sys.dont_write_bytecode = True

from pathlib import Path
from typing import Final, Iterable

from analysis.peek_experiments import peek

from .config import Config
from .loop import loop
from .utils.file_utils import ensure_empty_dirs

EXPERIMENT_DIR: Final[Path] = Path("experiments")
RESULTS_DIR: Final[Path] = Path("results")


def copy_orginal_files(files: Iterable[Path], dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        destination = dest_dir / file.name
        destination.write_bytes(file.read_bytes())


def unpack_variations(raw_dict: dict) -> list[dict]:
    if "variations" not in raw_dict:
        return [raw_dict]
    variations = raw_dict.pop("variations")
    variations = [c for v in variations for c in unpack_variations(v)]
    return [raw_dict | v for v in variations]


def load_experiments(files: Iterable[Path]) -> list[Config]:
    raw_dicts = [json.loads(f.read_text()) for f in files]
    return [Config(**c) for d in raw_dicts for c in unpack_variations(d)]


def save_experiment(config: Config, file_path: Path) -> None:
    with open(file_path, "w") as f:
        json.dump(asdict(config), f, indent=4)


def pretty_print_config(config: Config) -> None:
    # Note: Will likely not print upon repeated test runs due to sys.stdout deactivated
    print(f"Conducting experiment with:")
    pprint.pprint(asdict(config), sort_dicts=False, indent=2)
    print("\n")


def train():
    experiment_files = EXPERIMENT_DIR.glob("*.json")
    copy_orginal_files(experiment_files, RESULTS_DIR / "orig_files")

    experiments = load_experiments(experiment_files)
    if not experiments:
        raise ValueError("No experiments given. Exiting.")

    for i, experiment in enumerate(experiments):
        pretty_print_config(experiment)

        # create reult dir and persist experiment config
        result_dir = RESULTS_DIR / f"experiment_{i}"
        ensure_empty_dirs(result_dir)
        save_experiment(experiment, result_dir / "experiment.json")  # save parameters

        # start training
        loop(experiment, result_dir)

    # analyze results
    peek(RESULTS_DIR)


if __name__ == "__main__":
    train()
