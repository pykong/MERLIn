import csv
from dataclasses import asdict, dataclass, field


@dataclass
class EpisodeLog:
    episode: int
    epsilon: float
    reward: float = field(default=0.0, metadata={"decimal_places": 3})
    steps: int = field(default=0, metadata={"decimal_places": 3})
    time: float = field(default=0.0, metadata={"decimal_places": 3})


def log_to_csv(log: EpisodeLog, filename: str):
    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log.__annotations__.keys())
        if f.tell() == 0:  # file is empty, write a header
            writer.writeheader()
        writer.writerow(asdict(log))
