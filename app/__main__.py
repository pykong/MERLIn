import json
import pprint
import sys
from dataclasses import asdict

sys.dont_write_bytecode = True

from pathlib import Path
from typing import Final, Iterable

from analysis.analyze import peek

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


def unpack_variants(raw_dict: dict) -> list[dict]:
    if "variants" not in raw_dict:
        return [raw_dict]
    variants = raw_dict.pop("variants")
    variants = [c for v in variants for c in unpack_variants(v)]
    return [raw_dict | v for v in variants]


def load_experiments(files: Iterable[Path]) -> list[Config]:
    raw_dicts = [json.loads(f.read_text()) for f in files]
    return [Config(**c) for d in raw_dicts for c in unpack_variants(d)]


def save_experiment(config: Config, file_path: Path) -> None:
    with open(file_path, "w") as f:
        json.dump(asdict(config), f, indent=4)


def pretty_print_config(config: Config) -> None:
    # Note: Will likely not print upon repeated test runs due to sys.stdout deactivated
    print(f"Conducting experiment with:")
    pprint.pprint(asdict(config), sort_dicts=False, indent=2)
    print("\n")


def train():
    # glob experiment files
    experiment_files = [e for e in EXPERIMENT_DIR.glob("*.json")]
    if not experiment_files:
        raise ValueError("No experiment files found. Exiting.")

    # run each experiment
    for experiment_file in experiment_files:
        exp_result_dir = RESULTS_DIR / experiment_file.stem
        copy_orginal_files([experiment_file], exp_result_dir)

        # run training for each variant of experiment
        variants = load_experiments([experiment_file])
        for i, variant in enumerate(variants):
            # print out config to run
            pretty_print_config(variant)

            # create reult dir and persist experiment config
            result_dir = exp_result_dir / f"variant_{i}"
            ensure_empty_dirs(result_dir)
            save_experiment(variant, result_dir / "variant.json")

            # start training
            loop(variant, result_dir)

        # analyze results
        try:
            peek(exp_result_dir)
        except Exception as e:
            print(f"Analysis failed:\n{e}")


if __name__ == "__main__":
    train()
