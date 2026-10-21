#!/usr/bin/env python3
"""Joint audio-video generation from text (t2av): no conditioning media."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uniavc.inference.cli import run

if __name__ == "__main__":
    run("t2av", description=__doc__.splitlines()[0])
