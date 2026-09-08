"""Events module: start-of-movement / takeoff / landing detection.

NEEDS REVIEW -- see the boundary marker in workflow_demo.py.
"""

from jumpstart.events.detector import (
    GRAVITY_M_S2,
    JumpEvents,
    compute_kinematics,
    detect_jumps_from_track,
    smooth_hip_height,
)

__all__ = [
    "GRAVITY_M_S2",
    "JumpEvents",
    "compute_kinematics",
    "detect_jumps_from_track",
    "smooth_hip_height",
]
