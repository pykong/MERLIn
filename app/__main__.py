import pprint
import sys
from dataclasses import asdict, replace

from yaml import Loader, dump, load  # type:ignore

sys.dont_write_bytecode = True

from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Final, Iterable

# from analysis.__main__ import analyze
from app.config import Config
from app.loop import loop
from app.utils.file_utils import ensure_dirs

EXPERIMENT_DIR: Final[Path] = Path("experiments")
RESULTS_DIR: Final[Path] = Path("results")
NUM_WORKERS: Final[int] = cpu_count()
NUM_RUNS: Final[int] = 3


def copy_orginal_files(files: Iterable[Path], dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        destination = dest_dir / file.name
        destination.write_bytes(file.read_bytes())


def unpack_variants(raw_dict: dict[str, Any]) -> list[dict[str, Any]]:
    if "variants" not in raw_dict:
        return [raw_dict]
    variants = raw_dict.pop("variants")
    variants = [c for v in variants for c in unpack_variants(v)]
    return [raw_dict | v for v in variants]


def load_experiments(files: Iterable[Path]) -> list[Config]:
    raw_dicts = [load(f.read_text(), Loader) for f in files]
    return [Config(**c) for d in raw_dicts for c in unpack_variants(d)]


def validate_variants(variants: list[Config]) -> None:
    if not variants:
        raise ValueError("No experiment files found. Exiting.")
    if len(variants) != len(set(variants)):
        raise ValueError("Variants found not to be unique.")


def multiply_variants(variants: list[Config], num_runs: int) -> list[Config]:
    return [replace(v, run_id=i) for v in variants for i in range(num_runs)]


def save_experiment(config: Config, file_path: Path) -> None:
    file_path.write_text(dump(asdict(config)))


def pretty_print_config(config: Config) -> None:
    print("Conducting experiment with:")
    pprint.pprint(asdict(config), sort_dicts=False, indent=2)
    print("\n")


def train_variant(variant):
    # ensure result dirs
    exp_result_dir = RESULTS_DIR / variant.experiment_id
    variant_dir = exp_result_dir / variant.variant_id
    run_dir = variant_dir / str(variant.run_id)
    ensure_dirs(exp_result_dir, variant_dir, run_dir)

    # persist config for reproducibility
    save_experiment(replace(variant, run_id=None), variant_dir / "variant.yaml")

    # start training
    loop(variant, run_dir)


def train():
    # glob experiment files
    experiment_files = [e for e in EXPERIMENT_DIR.glob("*.yaml")]
    variants = load_experiments(experiment_files)

    # some validation
    validate_variants(variants)

    # clone config for each run
    variants = multiply_variants(variants, NUM_RUNS)

    # train in parallel
    with Pool(NUM_WORKERS) as p:
        p.map(train_variant, variants)

    # analyze results
    result_dirs = [d for d in RESULTS_DIR.glob("*") if d.is_dir()]
    # with Pool(NUM_WORKERS) as p:
    #     p.map(analyze, result_dirs)


if __name__ == "__main__":
    train()
