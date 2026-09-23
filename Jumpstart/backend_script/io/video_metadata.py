"""Video recording date/time metadata for /io/.

NEEDS REVIEW -- see the boundary marker in workflow_demo.py. The box-
parsing logic itself is unit-tested against hand-built fake MP4
bytes (see the module's own test note below), but not yet run
against a real iPad recording in this environment.

Reads the actual date+time a video was recorded straight out of the
MP4/MOV container's own "movie header" (moov/mvhd) atom -- a fixed,
long-standing part of the ISO/QuickTime file format spec used by
essentially every phone and tablet camera, including the iPad. This
needs NO external tool (no ffmpeg/ffprobe, no PATH or environment
variable to configure) -- just the standard library -- so every user
of the finished product gets this for free, with nothing to install
or set up.

This matters because:
    - The iPad filename convention only encodes a DATE
      (YYYYMMDD), never a time of day.
    - The Jumpstart downsampling pipeline (used only for this
      thesis's own validation work, not the end product) writes new
      output files for every framerate/resolution combination; those
      files' filesystem timestamps reflect when they were PROCESSED,
      not when they were filmed -- and re-encoding is not guaranteed
      to preserve the mvhd creation_time either. This has only been
      confirmed to matter on ORIGINAL, unprocessed recordings, which
      is exactly what the finished product will actually work with.

Known caveat (unverified against a real iPad file in this
environment): the spec says creation_time should be UTC, but some
camera/phone encoders write local time instead. If dates come out
shifted by a fixed number of hours, that is almost certainly why --
worth spot-checking one known recording before trusting this across
a whole dataset.
"""

from __future__ import annotations

import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional, Tuple

# Seconds between the QuickTime/MP4 epoch (1 Jan 1904) and the Unix
# epoch (1 Jan 1970) -- a fixed constant from the file format spec,
# not something that can drift or need updating.
_MAC_EPOCH_OFFSET_S = 2082844800


def get_recording_datetime(video_path: Path) -> Optional[datetime]:
    """Read a video's embedded recording date/time from its mvhd atom.

    Args:
        video_path: Path to an MP4/MOV video file.

    Returns:
        The recording date and time (UTC, see the module's caveat
        about encoders that write local time instead) if the
        container has a movie header atom with a valid
        creation_time, otherwise None -- including for any file that
        isn't a box-structured MP4/MOV, or has no timestamp set.
        Never raises for a malformed/unexpected file; this is best-
        effort metadata, not a validated parse.

    Raises:
        FileNotFoundError: If video_path does not exist.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    try:
        file_size = video_path.stat().st_size
        with open(video_path, "rb") as handle:
            creation_time = _find_mvhd_creation_time(handle, file_size)
    except (OSError, struct.error):
        return None

    if not creation_time:
        return None

    unix_timestamp = creation_time - _MAC_EPOCH_OFFSET_S
    try:
        return datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _iter_boxes(
    handle, start: int, end: int
) -> Iterator[Tuple[str, int, int]]:
    """Yield (box_type, payload_start, box_end) for each box directly
    inside [start, end) of the open file handle.

    Handles the ISO base media file format box header, including the
    64-bit "largesize" extension (size field == 1) and the
    "box extends to end of file" convention (size field == 0). Stops
    (rather than raising) on anything that doesn't look like a valid
    box, since this is best-effort metadata extraction.
    """
    handle.seek(start)
    while handle.tell() < end:
        box_start = handle.tell()
        header = handle.read(8)
        if len(header) < 8:
            return
        size, raw_type = struct.unpack(">I4s", header)
        box_type = raw_type.decode("ascii", errors="replace")

        header_size = 8
        if size == 1:
            large_size_bytes = handle.read(8)
            if len(large_size_bytes) < 8:
                return
            size = struct.unpack(">Q", large_size_bytes)[0]
            header_size = 16
        elif size == 0:
            size = end - box_start

        if size < header_size:
            return
        box_end = box_start + size
        payload_start = box_start + header_size

        yield box_type, payload_start, box_end
        handle.seek(box_end)


def _find_mvhd_creation_time(handle, file_size: int) -> Optional[int]:
    """Find the top-level 'moov' box, then its 'mvhd' child, and
    return the raw creation_time field (seconds since 1904-01-01,
    per the QuickTime/MP4 spec) -- or None if either box is missing.
    """
    for box_type, payload_start, box_end in _iter_boxes(handle, 0, file_size):
        if box_type != "moov":
            continue
        for inner_type, inner_payload_start, _inner_end in _iter_boxes(
            handle, payload_start, box_end
        ):
            if inner_type == "mvhd":
                handle.seek(inner_payload_start)
                return _read_mvhd_creation_time(handle)
        return None
    return None


def _read_mvhd_creation_time(handle) -> Optional[int]:
    """Parse an mvhd box body (handle positioned at its payload
    start) and return its creation_time field, or None if it's
    unset (0) or the box is truncated."""
    version_and_flags = handle.read(4)
    if len(version_and_flags) < 4:
        return None
    version = version_and_flags[0]

    if version == 1:
        # 64-bit creation_time (rare in practice, but part of spec).
        payload = handle.read(8)
        if len(payload) < 8:
            return None
        creation_time = struct.unpack(">Q", payload)[0]
    else:
        payload = handle.read(4)
        if len(payload) < 4:
            return None
        creation_time = struct.unpack(">I", payload)[0]

    return creation_time or None
