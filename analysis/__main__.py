import sys
from pathlib import Path

from analyzer.peek import peek


def main():
    result_dir = Path(sys.argv[1])  # point to result dir
    peek(result_dir)


if __name__ == "__main__":
    main()
