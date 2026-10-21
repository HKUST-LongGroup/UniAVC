#!/usr/bin/env python3
"""Joint audio-video editing (tav2av).

  --post-refine video|audio   re-denoise one branch against the other held clean
  --intent-aware video|audio  hold a branch intact by conditioning, not copying
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uniavc.inference.cli import run

if __name__ == "__main__":
    run("tav2av", description=__doc__.splitlines()[0])
