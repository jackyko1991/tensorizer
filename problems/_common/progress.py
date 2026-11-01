"""Progress bars for verbose runs.

for step in steps(n_steps, "time steps"):   # plain range(n) unless --verbose is on
    ...
"""

from __future__ import annotations

import os
import sys

from tqdm import tqdm


def verbose() -> bool:
    return os.environ.get("TENSORIZER_VERBOSE") == "1"


def bar(total: int, desc: str) -> tqdm:
    """A tqdm bar on stdout (so it shows in the app's terminal); disabled unless verbose."""
    return tqdm(total=total, desc=desc, file=sys.stdout, ncols=80, mininterval=0.2, ascii=" █", disable=not verbose())


def steps(n: int, desc: str = "steps"):
    """range(n), wrapped in a progress bar when the run is verbose."""
    if not verbose():
        return range(n)
    return tqdm(range(n), desc=desc, file=sys.stdout, ncols=80, mininterval=0.2, ascii=" █")
