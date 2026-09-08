"""Frame extraction for the /who/ step.

Grabs a single reference frame from partway through a video so the
user can assign athlete order-in-frame. The very first frame is
avoided on purpose -- athletes are often still walking into position
at t=0, so a frame further into the clip is far more likely to show
everyone lined up and ready.

This module only reads one frame with OpenCV. It has no UI code, so
it can be unit-tested without a display -- the interactive part
lives in assignment.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def extract_reference_frame(
    video_path: Path, fraction: float = 0.5
) -> Tuple[np.ndarray, int]:
    """Extract one frame from partway through a video.

    Args:
        video_path: Path to the video file.
        fraction: Position in the video to sample from, expressed
            as a fraction of total frame count (0.0 = first frame,
            1.0 = last frame). Defaults to the midpoint, since
            athletes are rarely in position yet at the very start.

    Returns:
        A tuple of (frame, frame_index) where frame is a BGR image
        array (as returned by OpenCV) and frame_index is the 0-based
        index of the frame that was grabbed.

    Raises:
        FileNotFoundError: If video_path does not exist.
        ValueError: If fraction is not between 0 and 1, or if the
            frame could not be read (e.g. corrupt video / codec
            issue).
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            f"fraction must be between 0 and 1, got {fraction!r}"
        )

    capture = cv2.VideoCapture(str(video_path))
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError(
                f"Could not determine frame count for "
                f"{video_path.name!r}; the file may be corrupt or "
                "use an unsupported codec."
            )
        frame_index = min(int(total_frames * fraction), total_frames - 1)

        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success or frame is None:
            raise ValueError(
                f"Could not read frame {frame_index} from "
                f"{video_path.name!r}."
            )
        return frame, frame_index
    finally:
        capture.release()
