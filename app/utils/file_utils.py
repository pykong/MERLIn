import shutil
from pathlib import Path

__all__ = ["ensure_empty_dirs", "ensure_dirs"]


def empty_dir(dir: Path) -> None:
    """Empties directory."""
    for item in dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def ensure_empty_dirs(*dirs: str | Path) -> None:
    """Creates directories if not existing, else empties them."""
    for dir in dirs:
        p = Path(dir)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=False)
        elif not p.is_dir():
            print(f"{dir} is not a valid directory")
        else:
            empty_dir(p)


def ensure_dirs(*dirs: str | Path) -> None:
    """Creates directories if not existing"""
    for dir in dirs:
        p = Path(dir)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=False)
        elif not p.is_dir():
            print(f"{dir} is not a valid directory")
