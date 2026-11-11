"""Run unmodified TrackEval with NumPy 2 replacements for removed dtype aliases."""

import runpy
import sys

import numpy as np


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_trackeval_compat.py TRACK_EVAL_SCRIPT [arguments ...]")
    script = sys.argv[1]
    sys.argv = [script, *sys.argv[2:]]
    # TrackEval's old np.float / np.int names were aliases for these Python types.
    # Restore only the names; the upstream metric code remains unchanged.
    np.float = float  # type: ignore[attr-defined]
    np.int = int  # type: ignore[attr-defined]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
