"""Pose module: per-frame multi-person hip-midpoint detection."""

from backend_script.pose.detector_yolo import (
    create_yolo_pose_model,
    detect_people_yolo,
)

__all__ = [
    "create_yolo_pose_model",
    "detect_people_yolo",
]