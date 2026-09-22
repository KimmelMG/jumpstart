"""YOLO-pose pose estimation for /pose/.

Sinds het verwijderen van de MediaPipe-backend (opschonen A1,
2026-09-17) de enige pose-detectiebackend in dit project, en
inmiddels de geteste/aanbevolen backend -- de eerdere "NEEDS REVIEW,
nog niet tegen echte beelden getest"-kanttekening hieronder klopte
niet meer en is verwijderd (2026-09-22, opschonen-to-do C).

Why this exists: added 2026-08-10 after real-footage debug-CSV
analysis (see jumpstart/tracking/tracker.py and jumpstart/events/
detector.py's docstrings) showed MediaPipe (both "lite" and "full"
variants -- swapping variants made no measurable difference) loses
the athlete's hip position entirely (0 detections that frame, not a
low-confidence one) during the fast/blurry part of a jump, in the
majority of real detected jumps, regardless of camera setup. YOLO-pose
(Ultralytics) is a different model family/architecture and may handle
motion blur differently -- this module makes it a drop-in alternative
so that can actually be tested, without committing to it being better
than MediaPipe (it has not been validated here).

Design: mirrors jumpstart/pose/detector.py's PersonDetection interface
exactly (same dataclass, imported from there rather than redefined) so
the rest of the pipeline (/tracking/, /identify/, /events/) does not
need to know or care which backend produced a given frame's
detections -- see jumpstart/tracking/tracker.py's detect_fn parameter
and jumpstart/workflow_demo.py's --pose-backend flag, which is what
actually chooses between this module and jumpstart/pose/detector.py.

Model file / package: unlike MediaPipe's single .task file, YOLO-pose
needs the `ultralytics` PyPI package installed (`pip install
ultralytics`) in addition to a model weights file (e.g.
yolov8n-pose.pt, yolov8s-pose.pt, ... yolov8x-pose.pt -- n/s/m/l/x are
smallest/fastest to largest/most accurate, or the newer yolo11n-pose.pt
etc. family). Ultralytics' own YOLO() constructor auto-downloads a
named model to its own cache the first time it's used, exactly like
MediaPipe's create_pose_landmarker does for its model file -- see
_load_yolo_model below.

Keypoint convention: YOLO-pose (Ultralytics) uses the standard COCO-17
keypoint layout: 0=nose, 1/2=eyes, 3/4=ears, 5/6=shoulders,
7/8=elbows, 9/10=wrists, 11/12=hips, 13/14=knees, 15/16=ankles. Hip
midpoint here uses indices 11 (left_hip) and 12 (right_hip), the same
two-point hip-midpoint convention as
jumpstart/pose/detector.py's LEFT_HIP_INDEX/RIGHT_HIP_INDEX (23/24 in
MediaPipe's own, different landmark numbering -- not directly
comparable, just the same anatomical points).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class PersonDetection:
    """One detected person's hip midpoint in one frame.

    Attributes:
        x: Horizontal pixel position of the hip midpoint.
        y: Vertical pixel position of the hip midpoint.
        visibility: Mean visibility score of the two hip landmarks
            (0-1, higher is more confident); may be reported as 1.0
            when the backend does not provide a confidence score.
    """

    x: float
    y: float
    visibility: float

LEFT_HIP_KEYPOINT = 11
RIGHT_HIP_KEYPOINT = 12

# COCO class 0 is "person" -- restricting detection to this class stops
# YOLO from wasting time/candidates on other COCO classes it was
# trained on (irrelevant here, but harmless either way since /tracking/
# only ever sees PersonDetection objects built from person detections).
PERSON_CLASS_ID = 0

DEFAULT_YOLO_MODEL_NAME = "yolov8n-pose.pt"


def create_yolo_pose_model(
    model_path: Optional[Path] = None,
    model_name: str = DEFAULT_YOLO_MODEL_NAME,
    device: Optional[str] = None,
):
    """Load (downloading if needed) a YOLO-pose model.

    Args:
        model_path: Path to an already-downloaded .pt weights file. If
            given and it exists, loads directly from there (fully
            offline). If None, falls back to model_name and lets
            Ultralytics resolve/auto-download it to its own cache --
            see the module docstring's caveat that this download path
            has not actually been exercised in this environment.
        model_name: Which YOLO-pose variant to use when model_path is
            not given (e.g. "yolov8n-pose.pt" fastest/least accurate
            through "yolov8x-pose.pt" slowest/most accurate, or a
            newer "yolo11n-pose.pt"-style name). Only used if
            model_path is None or does not exist.
        device: Passed to Ultralytics as the inference device, e.g.
            "cpu", "cuda:0". None lets Ultralytics auto-select (GPU if
            available, else CPU) -- explicitly pass "cpu" to force CPU
            even if a GPU is present, or if auto-detection picks
            something that errors on this machine.

    Returns:
        A loaded ultralytics.YOLO model instance, ready for
        detect_people_yolo().

    Raises:
        ImportError: If the `ultralytics` package is not installed
            (run `pip install ultralytics` first).
        RuntimeError: If loading/downloading the model fails for any
            reason (no internet, bad path, etc.).
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "The 'ultralytics' package is required for the YOLO pose "
            "backend but is not installed. Install it with: "
            "pip install ultralytics"
        ) from exc

    source = str(model_path) if model_path is not None and Path(model_path).exists() else model_name
    try:
        model = YOLO(source)
    except Exception as exc:  # ultralytics raises various exception types
        raise RuntimeError(
            f"Could not load/download YOLO-pose model {source!r}: {exc}. "
            "If this machine has no internet access, download the "
            "weights file manually from "
            "https://docs.ultralytics.com/tasks/pose/#models and pass "
            "its path via model_path."
        ) from exc

    if device is not None:
        model.to(device)
    return model


def detect_people_yolo(
    model, frame_bgr: np.ndarray, conf_threshold: float = 0.25, imgsz: int = 640
) -> List[PersonDetection]:
    """Detect every person's hip midpoint in a single BGR video frame.

    Args:
        model: A YOLO-pose model from create_yolo_pose_model.
        frame_bgr: A single video frame as a BGR uint8 array -- the
            format cv2.VideoCapture.read() returns directly (unlike
            jumpstart/pose/detector.py's detect_people, this does NOT
            need the caller to convert to RGB first -- Ultralytics
            accepts numpy arrays in the same BGR layout OpenCV uses).
        conf_threshold: Minimum YOLO person-detection confidence (not
            the per-keypoint confidence -- see PersonDetection.
            visibility for that) to keep a detected person at all.
            YOLO's default is 0.25; raise this if very low-confidence
            spurious person detections show up in practice.
        imgsz: Longest-side resolution (pixels) Ultralytics resizes
            the frame to internally before running inference.
            Ultralytics' own default is 640; inference cost scales
            roughly with imgsz^2, so a smaller value (e.g. 480 or
            320) is noticeably faster on CPU-only machines, at some
            cost to detecting small/far-away people. Output
            coordinates are always rescaled back to the original
            frame's pixel space regardless of this setting, so
            callers never need to adjust for it. 

    Returns:
        One PersonDetection per person YOLO detected in this frame
        above conf_threshold, in YOLO's own (frame-to-frame unstable,
        same caveat as MediaPipe) order.
    """
    results = model.predict(
        frame_bgr,
        conf=conf_threshold,
        classes=[PERSON_CLASS_ID],
        imgsz=imgsz,
        verbose=False,
    )
    result = results[0]

    detections: List[PersonDetection] = []
    if result.keypoints is None or len(result.keypoints) == 0:
        return detections

    # .xy: (n_people, 17, 2) pixel coordinates. .conf: (n_people, 17)
    # per-keypoint confidence -- None if the loaded model variant
    # doesn't output keypoint confidence (shouldn't happen for the
    # *-pose.pt models this module targets, but guarded just in case).
    keypoints_xy = result.keypoints.xy.cpu().numpy()
    keypoints_conf = (
        result.keypoints.conf.cpu().numpy()
        if result.keypoints.conf is not None
        else None
    )

    for person_index in range(keypoints_xy.shape[0]):
        left_hip = keypoints_xy[person_index, LEFT_HIP_KEYPOINT]
        right_hip = keypoints_xy[person_index, RIGHT_HIP_KEYPOINT]
        # A keypoint YOLO could not locate at all is reported as (0, 0)
        # with ~0 confidence -- skip this person's hip midpoint rather
        # than silently averaging in a bogus (0, 0) corner point that
        # would corrupt /tracking/'s matching.
        left_ok = not (left_hip[0] == 0 and left_hip[1] == 0)
        right_ok = not (right_hip[0] == 0 and right_hip[1] == 0)
        if not left_ok and not right_ok:
            continue
        if left_ok and right_ok:
            x = float((left_hip[0] + right_hip[0]) / 2)
            y = float((left_hip[1] + right_hip[1]) / 2)
        elif left_ok:
            x, y = float(left_hip[0]), float(left_hip[1])
        else:
            x, y = float(right_hip[0]), float(right_hip[1])

        if keypoints_conf is not None:
            left_vis = keypoints_conf[person_index, LEFT_HIP_KEYPOINT]
            right_vis = keypoints_conf[person_index, RIGHT_HIP_KEYPOINT]
            visible_scores = [
                v for v, ok in ((left_vis, left_ok), (right_vis, right_ok)) if ok
            ]
            visibility = float(np.mean(visible_scores)) if visible_scores else 1.0
        else:
            visibility = 1.0

        detections.append(PersonDetection(x=x, y=y, visibility=visibility))

    return detections
