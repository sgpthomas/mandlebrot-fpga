#!/usr/bin/env python3

import sys
from pathlib import Path
import json


def main():
    directory = Path(sys.argv[1])
    directory.mkdir(exist_ok=True)
    for i in range(3, 32):
        with (directory / f"{i}.json").open("w+") as f:
            data = {
                "width": 256,
                "precision": 64,
                "unroll": 1,
                "n_iters": 16,
                "int_width": i,
            }
            json.dump(data, f)


if __name__ == "__main__":
    main()
