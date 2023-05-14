import csv
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Self

from loguru import logger


class LogLevel(Enum):
    VICTORY = "VICTORY"
    DEFEAT = "DEFEAT"

    def __str__(self: Self) -> str:
        return self.value


# Remove default handler and add custom format
logger.remove()

# Base format with placeholders for color and icon
format_base = " {{level.icon}} <{color}>{{level}}</> - {{time:HH:mm:ss}} | {{message}}"

logger.level(str(LogLevel.VICTORY), no=48, icon="🏆", color="<green>")
logger.level(str(LogLevel.DEFEAT), no=49, icon="💀", color="<red>")

logger.add(
    sys.stderr,
    format=format_base.format(color="green"),
    filter=lambda record: record["level"].name == str(LogLevel.VICTORY),
)
logger.add(
    sys.stderr,
    format=format_base.format(color="red"),
    filter=lambda record: record["level"].name == str(LogLevel.DEFEAT),
)


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

    def log(self: Self, episode_log: EpisodeLog) -> None:
        level = LogLevel.VICTORY if episode_log.reward > 0 else LogLevel.DEFEAT
        logger.log(str(level), episode_log)
        self.__log_to_csv(episode_log)

    def __log_to_csv(self: Self, episode_log: EpisodeLog) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.touch()
        with open(self.log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=episode_log.__annotations__.keys())
            if f.tell() == 0:  # file is empty, write a header
                writer.writeheader()
            writer.writerow(asdict(episode_log))
