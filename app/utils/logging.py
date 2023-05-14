import csv
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Self

from loguru import logger


@dataclass
class EpisodeLog:
    episode: int
    epsilon: float
    __start_time: float = field(default=None, init=False, repr=False)  # type: ignore
    reward: float = field(default=0.0, metadata={"decimal_places": 3})
    steps: int = field(default=0, metadata={"decimal_places": 3})
    time: float = field(default=0.0, metadata={"decimal_places": 3})

    def start_timer(self):
        self.__start_time = time.time()

    def stop_timer(self):
        if self.__start_time is None:
            raise ValueError("Timer has not been started.")
        self.time = time.time() - self.__start_time


class EpisodeLogger:
    def __init__(self: Self, log_file: Path):
        self.log_file = log_file
        # log_fmt = " {level.icon} {level} - {message}"
        # logger.level("ERROR", color="<red>", icon="💀")
        # logger.level("VICTORY", no=15, color="<green>", icon="🏆")
        # logger.add(sys.stdout, format=log_fmt)
        # logger.add(sys.stdout, level="VICTORY", format=log_fmt)

    def log(self: Self, episode_log: EpisodeLog) -> None:
        level = "SUCCESS" if episode_log.reward > 0 else "ERROR"
        logger.log(level, episode_log)
        self.__log_to_csv(episode_log)

    def __log_to_csv(self: Self, episode_log: EpisodeLog) -> None:
        with open(self.log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=episode_log.__annotations__.keys())
            if f.tell() == 0:  # file is empty, write a header
                writer.writeheader()
            writer.writerow(asdict(episode_log))
