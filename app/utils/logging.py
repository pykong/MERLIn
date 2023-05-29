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
    VIDEO = "VIDEO"
    SAVE = "SAVE"
    GREEN = "GREEN"
    YELLOW = "YELLOW"

    def __str__(self: Self) -> str:
        return self.value


# Remove default handler and add custom format
logger.remove()

# Base format with placeholders for color and icon
msg_fmt = " {{level.icon}} <{color}>{{level:<9}}</> - {{time:HH:mm:ss}} | {{message}}"

logger.level(str(LogLevel.VICTORY), no=48, icon="🏆")
logger.level(str(LogLevel.DEFEAT), no=49, icon="💀")
logger.level(str(LogLevel.VIDEO), no=47, icon="🎥")
logger.level(str(LogLevel.SAVE), no=46, icon="💾")
logger.level(str(LogLevel.GREEN), no=36)

logger.add(
    sys.stderr,
    format=msg_fmt.format(color="green"),
    filter=lambda record: record["level"].name == str(LogLevel.VICTORY),
)
logger.add(
    sys.stderr,
    format=msg_fmt.format(color="red"),
    filter=lambda record: record["level"].name == str(LogLevel.DEFEAT),
)
logger.add(
    sys.stderr,
    format=msg_fmt.format(color="blue"),
    filter=lambda record: record["level"].name == str(LogLevel.VIDEO),
)
logger.add(
    sys.stderr,
    format=msg_fmt.format(color="yellow"),
    filter=lambda record: record["level"].name == str(LogLevel.SAVE),
)
logger.add(
    sys.stderr,
    format="<green>{message}</>",
    filter=lambda record: record["level"].name == str(LogLevel.GREEN),
)
logger.add(
    sys.stderr,
    format="<yellow>{message}</>",
    filter=lambda record: record["level"].name == str(LogLevel.YELLOW),
)


@dataclass
class EpisodeLog:
    episode: int
    epsilon: float
    reward: float = 0.0
    steps: int = 0
    time: float = field(init=False)

    def start_timer(self: Self) -> None:
        self.__start_time = time.time()

    def stop_timer(self: Self) -> None:
        if self.__start_time is None:
            raise ValueError("Timer has not been started.")
        self.time = time.time() - self.__start_time

    def __str__(self: Self) -> str:
        fields = (
            f"{self.episode:05d}",
            f"{self.epsilon:.3f}",
            f"{self.reward:.2f}",
            f"{self.steps:04d}",
            f"{self.time:.2f}",
        )
        return " | ".join(fields)


class EpisodeLogger:
    def __init__(self: Self, log_file: Path):
        self.log_file = log_file

    def log(self: Self, message: EpisodeLog | str, level: LogLevel | str = "") -> None:
        if isinstance(message, EpisodeLog):
            level = LogLevel.VICTORY if message.reward > 0 else LogLevel.DEFEAT
            logger.log(str(level), message)
            self.__log_to_csv(message)
        else:
            logger.log(str(level), message)

    def __log_to_csv(self: Self, episode_log: EpisodeLog) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.touch()
        with open(self.log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=episode_log.__annotations__.keys())
            if f.tell() == 0:  # file is empty, write a header
                writer.writeheader()
            writer.writerow(asdict(episode_log))
