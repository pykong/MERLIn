import sys

sys.dont_write_bytecode = True

from config import Config
from loop import CHECKPOINTS_DIR, IMG_DIR, LOG_DIR, VIDEO_DIR, ensure_empty_dirs, loop


def train():
    ensure_empty_dirs(CHECKPOINTS_DIR, LOG_DIR, VIDEO_DIR, IMG_DIR)
    config = Config()
    loop(config)


if __name__ == "__main__":
    train()
