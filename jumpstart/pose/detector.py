"""Pose estimation for /pose/.

NEEDS REVIEW -- see the boundary marker in workflow_demo.py. Built in
one pass and only checked with synthetic landmark data, never against
real Jumpstart footage.

Wraps MediaPipe's PoseLandmarker (Tasks API) to detect every person's
hip midpoint (average of landmarks 23/24) in a single video frame.
This module only looks at one frame at a time; associating detections
across frames into per-athlete tracks is /tracking/'s job -- MediaPipe
does not guarantee the same person keeps the same list position from
one frame to the next.

Model file: end users of the finished product should never have to
manually download anything. create_pose_landmarker() downloads the
pose landmarker model to a per-user cache folder
(~/.jumpstart/models/) the first time it's needed, and reuses it on
every later run. Only pass a custom model_path if you specifically
want a different model variant (e.g. pose_landmarker_full.task for
higher accuracy at the cost of speed) or need to work fully offline
with a model you already have.

NEEDS REVIEW: the download URL below was confirmed via MediaPipe's
own documentation and sample notebooks (see workflow_demo.py's
OPEN REQUIREMENTS note for context), but has not been exercised in
this environment -- this sandbox's network access blocks
storage.googleapis.com, so the actual download has only been
verified to be reachable in principle, not run end to end. Test the
very first run somewhere with normal internet access.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

LEFT_HIP_INDEX = 23
RIGHT_HIP_INDEX = 24

# Lite variant: smallest/fastest of MediaPipe's three pose landmarker
# models, trading a bit of accuracy for speed. Swap for the "full" or
# "heavy" variant (same URL pattern, different middle segment) if
# accuracy matters more than speed once this is tested against real
# footage.
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
DEFAULT_MODEL_PATH = Path.home() / ".jumpstart" / "models" / "pose_landmarker_lite.task"


@dataclass(frozen=True)
class PersonDetection:
    """One detected person's hip midpoint in one frame.

    Attributes:
        x: Horizontal pixel position of the hip midpoint.
        y: Vertical pixel position of the hip midpoint.
        visibility: Mean visibility score of the two hip landmarks
            (0-1, higher is more confident); MediaPipe may leave
            this unset (None) for some models, in which case it is
            reported as 1.0.
    """

    x: float
    y: float
    visibility: float


def _download_model(model_path: Path, model_url: str) -> None:
    """Download a pose landmarker model file to model_path.

    Args:
        model_path: Destination path; parent directories are created
            if needed.
        model_url: URL to download the model file from.

    Raises:
        RuntimeError: If the download fails for any reason (no
            internet, URL moved, etc.), with a message pointing at
            the manual-download page as a fallback.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose landmarker model to {model_path} ...")
    try:
        urllib.request.urlretrieve(model_url, model_path)
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(
            f"Could not download the pose landmarker model from "
            f"{model_url!r}: {exc}. Download it manually from "
            "https://ai.google.dev/edge/mediapipe/solutions/vision/"
            f"pose_landmarker and save it to {model_path}, or pass "
            "an existing model file via model_path."
        ) from exc
    print("Download complete.")


def create_pose_landmarker(
    model_path: Optional[Path] = None,
    num_poses: int = 8,
    download_if_missing: bool = True,
    model_url: str = DEFAULT_MODEL_URL,
    min_pose_detection_confidence: float = 0.5,
    min_pose_presence_confidence: float = 0.5,
) -> vision.PoseLandmarker:
    """Build a MediaPipe PoseLandmarker for per-frame (IMAGE mode) use.

    Args:
        model_path: Path to a ``.task`` pose landmarker model file.
            Defaults to a per-user cache location
            (~/.jumpstart/models/pose_landmarker_lite.task) that is
            shared across every run and every project on this
            machine, so end users only ever download this once, not
            once per analysis.
        num_poses: Maximum number of people to detect per frame. Set
            a bit higher than the number of athletes actually in
            frame so a coach/bystander doesn't crowd out an athlete.
        download_if_missing: If the model file does not exist yet at
            model_path, download it automatically. Set to False for
            a fully offline/controlled setup where you'd rather get
            a clear error than an unexpected network call.
        model_url: Where to download the model from if it's missing.
            Only override this to use a different model variant
            (e.g. pose_landmarker_full.task for more accuracy at the
            cost of speed).
        min_pose_detection_confidence: Added 2026-08-15 (oplossing B1
            uit Jumpstart_oplossingsrichtingen.docx) -- previously
            this was never set explicitly, so MediaPipe silently fell
            back to its own default (~0.5). Made explicit and
            adjustable here so it can be tuned against real footage
            instead of staying an invisible, unverified default --
            lower this (e.g. 0.2-0.3) if real debug-CSV data shows
            MediaPipe reporting "nobody found" on frames where a
            partially-visible athlete is genuinely in frame.
        min_pose_presence_confidence: Added 2026-08-15, same reasoning
            as min_pose_detection_confidence -- MediaPipe's separate
            threshold for whether a detected pose is confidently
            "present" once found at all.

    Returns:
        A configured PoseLandmarker, running in IMAGE mode: every
        call to detect_people treats its frame independently.

    Raises:
        FileNotFoundError: If model_path does not exist and
            download_if_missing is False.
        RuntimeError: If download_if_missing is True but the
            download fails.
    """
    model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not model_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"Pose landmarker model not found: {model_path}. "
                "Download one from the MediaPipe model zoo first (see "
                "this module's docstring), or call with "
                "download_if_missing=True."
            )
        _download_model(model_path, model_url)

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
    )
    return vision.PoseLandmarker.create_from_options(options)


def detect_people(
    landmarker: vision.PoseLandmarker, frame_rgb: np.ndarray
) -> List[PersonDetection]:
    """Detect every person's hip midpoint in a single RGB frame.

    Args:
        landmarker: A PoseLandmarker created by
            create_pose_landmarker.
        frame_rgb: A single video frame as an RGB uint8 array. Note
            OpenCV reads frames as BGR -- convert with
            ``frame[:, :, ::-1]`` before calling this.

    Returns:
        One PersonDetection per person MediaPipe detected in this
        frame, in MediaPipe's own (frame-to-frame unstable) order.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)

    height, width = frame_rgb.shape[:2]
    detections: List[PersonDetection] = []
    for landmarks in result.pose_landmarks:
        left_hip = landmarks[LEFT_HIP_INDEX]
        right_hip = landmarks[RIGHT_HIP_INDEX]
        x = (left_hip.x + right_hip.x) / 2 * width
        y = (left_hip.y + right_hip.y) / 2 * height
        left_vis = left_hip.visibility
        right_vis = right_hip.visibility
        if left_vis is None or right_vis is None:
            visibility = 1.0
        else:
            visibility = (left_vis + right_vis) / 2
        detections.append(PersonDetection(x=x, y=y, visibility=visibility))
    return detections
