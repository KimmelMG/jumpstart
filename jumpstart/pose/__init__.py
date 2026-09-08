"""Pose module: per-frame multi-person hip-midpoint detection.

NEEDS REVIEW -- see the boundary marker in workflow_demo.py.
"""

from jumpstart.pose.detector import (
    PersonDetection,
    create_pose_landmarker,
    detect_people,
)
from jumpstart.pose.detector_yolo import (
    create_yolo_pose_model,
    detect_people_yolo,
)

__all__ = [
    "PersonDetection",
    "create_pose_landmarker",
    "detect_people",
    "create_yolo_pose_model",
    "detect_people_yolo",
]
