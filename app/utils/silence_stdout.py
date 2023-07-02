import os
import sys
from contextlib import contextmanager


@contextmanager
def silence_stdout():
    orig_stdout = sys.stdout
    try:
        sys.stdout = open(os.devnull, "w")
        yield
    finally:
        sys.stdout.close()
        sys.stdout = orig_stdout
