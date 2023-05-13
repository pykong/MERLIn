import shutil
from pathlib import Path


def empty_directories(*dirs):
    for dir in dirs:
        p = Path(dir)
        if p.is_dir():
            for item in p.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        else:
            print(f"{dir} is not a valid directory")
