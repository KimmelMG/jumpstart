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

def get_frame_count(video_path: Path) -> int:
    """Return the total frame count of a video.

    Args:
        video_path: Path to the video file.

    Returns:
        Total frame count as reported by the container.

    Raises:
        FileNotFoundError: If video_path does not exist.
        ValueError: If the frame count could not be determined.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError(
                f"Could not determine frame count for "
                f"{video_path.name!r}; the file may be corrupt or "
                "use an unsupported codec."
            )
        return total_frames
    finally:
        capture.release()


def extract_frame_near(
    video_path: Path, current_frame_index: int, delta_seconds: float
) -> Tuple[np.ndarray, int, int]:
    """Extract a frame shifted by delta_seconds (+/-) from current_frame_index.

    Added 2026-09-09: the /who/ reference frame is otherwise fixed at
    a single position (see extract_reference_frame above), which
    sometimes lands mid-jump -- no stable hip position to calibrate
    on. This lets the webapp step the shown reference frame forward or
    back by a chosen number of seconds within the SAME video, so the
    user can pick a stable moment instead.

    Args:
        video_path: Path to the video file.
        current_frame_index: 0-based index of the frame currently shown.
        delta_seconds: Seconds to shift by; negative moves earlier.
            Converted to a frame count using the video's own fps (with
            a 30fps fallback if the container doesn't report one).

    Returns:
        A tuple of (frame, new_frame_index, total_frames).
        new_frame_index is clamped to [0, total_frames - 1] -- shifting
        past either end just stops there instead of erroring, so
        repeated clicks near a boundary are harmless.

    Raises:
        FileNotFoundError: If video_path does not exist.
        ValueError: If the frame count/frame could not be read.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError(
                f"Could not determine frame count for "
                f"{video_path.name!r}; the file may be corrupt or "
                "use an unsupported codec."
            )
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0  # fallback, e.g. some containers don't report fps

        new_index = current_frame_index + int(round(delta_seconds * fps))
        new_index = max(0, min(new_index, total_frames - 1))

        # Toegevoegd 2026-09-09 (vervolg): CAP_PROP_FRAME_COUNT is bij
        # sommige containers een OVERSCHATTING van het werkelijk
        # leesbare aantal frames -- een bekend OpenCV/ffmpeg-manco dat
        # vooral dicht bij het einde van het bestand optreedt. Een
        # seek+read op precies "de laatste frame" (of andere frames
        # vlak daarvoor) kan dan alsnog mislukken, ook al is
        # new_index correct geclamped op total_frames - 1. Val in dat
        # geval terug op een paar frames eerder in plaats van te
        # crashen, zodat "1s later" bij het einde van de video gewoon
        # het laatst haalbare frame laat zien.
        frame = None
        read_index = new_index
        for _ in range(5):
            capture.set(cv2.CAP_PROP_POS_FRAMES, read_index)
            success, frame = capture.read()
            if success and frame is not None:
                new_index = read_index
                break
            frame = None
            read_index -= 1
            if read_index < 0:
                break
        if frame is None:
            raise ValueError(
                f"Could not read a frame near index {new_index} from "
                f"{video_path.name!r} (tried up to 5 frames earlier)."
            )
        return frame, new_index, total_frames
    finally:
        capture.release()

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
