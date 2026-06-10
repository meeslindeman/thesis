#!/usr/bin/env python3
"""Divide Train loss and Train accuracy by 3 to correct for 3x resampler."""

import re
import sys
from pathlib import Path


def fix_line(line: str) -> str:
    def divide(m):
        return m.group(1) + f"{float(m.group(2)) / 3:.4f}"

    line = re.sub(r"(Train loss: )(\d+\.\d+)", divide, line)
    line = re.sub(r"(Train accuracy: )(\d+\.\d+)", divide, line)
    return line


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path/to/logs.out>")
        sys.exit(1)

    src = Path(sys.argv[1])
    dst = src.with_stem(src.stem + "_corrected")

    with src.open() as f:
        lines = f.readlines()

    fixed = [fix_line(line) for line in lines]

    with dst.open("w") as f:
        f.writelines(fixed)

    print(f"Written to {dst}")


if __name__ == "__main__":
    main()
