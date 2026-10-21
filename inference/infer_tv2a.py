#!/usr/bin/env python3
"""Video-to-audio generation (tv2a).

The input video is held clean while the audio denoises, so the output video is
byte-identical to the input.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uniavc.inference.cli import run

if __name__ == "__main__":
    run("tv2a", description=__doc__.splitlines()[0])
