"""Jumps module: segments and numbers individual jumps per athlete.

NEEDS REVIEW -- see the boundary marker in workflow_demo.py.
"""

from backend_script.jumps.segmenter import Jump, build_jumps_for_athlete

__all__ = ["Jump", "build_jumps_for_athlete"]
