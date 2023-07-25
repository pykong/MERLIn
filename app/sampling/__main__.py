from pathlib import Path
from typing import Final

from app.sampling.config import SamplingConfig
from app.sampling.loop import run_multiprocess_loop

DATA_DIR: Final[Path] = Path("data")
SAMPLE_CONFIG_PATH: Final[Path] = DATA_DIR / "sampling.json"


def load_sampling_config() -> SamplingConfig:
    return SamplingConfig()


def sample():
    print("Sample called.")
    config = load_sampling_config()
    run_multiprocess_loop(config)


if __name__ == "__main__":
    sample()
