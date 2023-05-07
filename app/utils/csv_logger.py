from csv import DictWriter
from pathlib import Path
from typing import Self


class CsvLogger:
    def __init__(self: Self, log_file: Path, columns: list[str]) -> None:
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)  # ensure folders
        self.columns = columns
        with open(self.log_file, "w", newline="") as csvfile:
            self.writer = DictWriter(csvfile, fieldnames=columns)
            self.writer.writeheader()

    def log(self: Self, items: dict[str, str | int | float]) -> None:
        with open(self.log_file, "a", newline="") as csvfile:
            self.writer = DictWriter(csvfile, fieldnames=self.columns)
            self.writer.writerow(items)
