import csv
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class EpisodeLog:
    episode: int
    epsilon: float
    __start_time: float = None
    reward: float = field(default=0.0, metadata={"decimal_places": 3})
    steps: int = field(default=0, metadata={"decimal_places": 3})
    time: float = field(default=0.0, metadata={"decimal_places": 3})

    def start_timer(self):
        self.__start_time = time.time()

    def stop_timer(self):
        if self.__start_time is None:
            raise ValueError("Timer has not been started.")
        self.time = time.time() - self.__start_time


def log_to_csv(log: EpisodeLog, filename: Path):
    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log.__annotations__.keys())
        if f.tell() == 0:  # file is empty, write a header
            writer.writeheader()
        writer.writerow(asdict(log))
