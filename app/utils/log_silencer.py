import os
import sys


class LogSilencer:
    def __enter__(self):
        self.stdout_orig = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stdout = self.stdout_orig
