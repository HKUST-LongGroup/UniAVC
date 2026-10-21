#!/usr/bin/env python3
"""Audio-to-video generation (ta2v).

The input audio is held clean while the video denoises, so the output audio is
byte-identical to the input.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uniavc.inference.cli import run

if __name__ == "__main__":
    run("ta2v", description=__doc__.splitlines()[0])
