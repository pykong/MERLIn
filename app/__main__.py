import pprint
import sys
from dataclasses import asdict, replace

from yaml import Loader, dump, load  # type:ignore

sys.dont_write_bytecode = True

from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Final, Iterable

from analysis.__main__ import collect_and_analyze
from app.config import Config
from app.loop import loop
from app.utils.file_utils import ensure_dirs

EXPERIMENT_DIR: Final[Path] = Path("experiments")
RESULTS_DIR: Final[Path] = Path("results")
NUM_WORKERS: Final[int] = cpu_count()


def copy_orginal_files(files: Iterable[Path], dest_dir: Path) -> None:
    """Persist original experiment files, for reproducibility.

    Args:
        files (Iterable[Path]): The experiment files.
        dest_dir (Path): The destination directory.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        destination = dest_dir / file.name
        destination.write_bytes(file.read_bytes())


def unpack_variants(raw_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Unpack individual variants of a configuration instance.

    Args:
        raw_dict (dict[str, Any]): A configuration instance as a dict.

    Returns:
        list[dict[str, Any]]: The individual variants.
    """
    if "variants" not in raw_dict:
        return [raw_dict]
    variants = raw_dict.pop("variants")
    variants = [c for v in variants for c in unpack_variants(v)]
    return [raw_dict | v for v in variants]


def load_experiments(files: Iterable[Path]) -> list[Config]:
    """Load experiment files into configuration instances.

    Args:
        files (Iterable[Path]): The experiment files.

    Returns:
        list[Config]: The configuration instances.
    """
    raw_dicts = [load(f.read_text(), Loader) for f in files]
    return [Config(**c) for d in raw_dicts for c in unpack_variants(d)]


def validate_variants(variants: list[Config]) -> None:
    """Run basic validation against configuration instances.

    Args:
        variants (list[Config]): The configuration instances.

    Raises:
        ValueError: If no experiments are given, or if experiments are duplicated.
    """
    if not variants:
        raise ValueError("No experiment files found. Exiting.")
    if len(variants) != len(set(variants)):
        raise ValueError("Variants found not to be unique.")


def multiply_variants(variants: list[Config]) -> list[Config]:
    """Multiply variants to individual runs according to run_count.

    Args:
        variants (list[Config]): The configuration instances.

    Returns:
        list[Config]: The multiplied configuration instances.
    """
    return [replace(v, run=i) for v in variants for i in range(v.run_count)]


def save_experiment(config: Config, file_path: Path) -> None:
    """Persist configuration instance to file, for reproducibility.

    Args:
        config (Config): The configuration instance.
        file_path (Path): The file path to save to.
    """
    file_path.write_text(dump(asdict(config)))


def pretty_print_config(config: Config) -> None:
    """Pretty print configuration instance.

    Args:
        config (Config): The configuration instance.
    """
    print("Conducting experiment with:")
    pprint.pprint(asdict(config), sort_dicts=False, indent=2)
    print("\n")


def run_train_loop(variant) -> None:
    """Prepare and conduct the training of a single run.

    Args:
        variant (_type_): The configuration instance of the individual run.
    """
    # ensure result dirs
    exp_result_dir = RESULTS_DIR / variant.experiment
    variant_dir = exp_result_dir / variant.variant
    run_dir = variant_dir / str(variant.run)
    ensure_dirs(exp_result_dir, variant_dir, run_dir)

    # persist config for reproducibility
    save_experiment(replace(variant, run=None), variant_dir / "variant.yaml")

    # start training
    loop(variant, run_dir)


def train() -> None:
    """Main method to coordinate the entire training process."""
    # glob experiment files
    experiment_files = [e for e in EXPERIMENT_DIR.glob("*.yaml")]
    variants = load_experiments(experiment_files)

    # some validation
    validate_variants(variants)

    # clone config for each run
    variants = multiply_variants(variants)

    # train in parallel
    with Pool(NUM_WORKERS) as p:
        p.map(run_train_loop, variants)

    # analyze results
    result_dirs = [d for d in RESULTS_DIR.glob("*") if d.is_dir()]
    with Pool(NUM_WORKERS) as p:
        p.map(collect_and_analyze, result_dirs)


if __name__ == "__main__":
    train()
