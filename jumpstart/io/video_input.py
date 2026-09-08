"""Video registration for a recording session.

This module only registers videos and their basic metadata. It does
not open, decode, or analyze video content -- that responsibility
belongs to /pose/ and later modules. Keeping /io/ dumb-but-reliable
means batch uploads (e.g. a whole team, 3-5 jumps per athlete) can be
queued quickly without waiting on any analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List

# Widened 2026-08-28 (open requirement #1, see jumpstart/workflow_demo.py's
# "OPEN REQUIREMENTS" note) to cover the range of formats athletes'
# phones/cameras might realistically produce, not just the iPad's own
# .mp4/.mov/.m4v. This is an EXTENSION check only -- it says nothing
# about whether this machine's OpenCV build can actually decode a given
# file's codec. That is checked separately, once per video, right
# before /tracking/ opens it for real, in workflow_demo.py's main()
# (see _check_video_decodable there) -- deliberately NOT here, so
# batch registration stays fast and /io/ stays "dumb" per the module
# docstring above (no per-file decode probe on every registered video,
# which would mean opening every phone-shot video in a batch upload
# just to register it).
SUPPORTED_EXTENSIONS = {
    ".mp4", ".mov", ".m4v",   # original set (iPad/iPhone)
    ".avi", ".mkv",           # older/some Android phones
    ".webm"                   # some Android phones, web exports
}


@dataclass(frozen=True)
class VideoFile:
    """A single registered video, not yet linked to an athlete.

    Athlete linkage happens later in /who/ (user assigns order in
    frame) and /identify/ (that order is applied post-tracking).

    Attributes:
        video_id: Stable unique identifier (e.g. "V001").
        path: Absolute path to the video file on disk.
        session_id: Identifier of the recording session this video
            belongs to (e.g. one training/warm-up).
        registered_at: Timestamp of when the video was registered.
    """

    video_id: str
    path: Path
    session_id: str
    registered_at: datetime = field(default_factory=datetime.now)


def discover_videos_in_folder(
    folder: Path, recursive: bool = True
) -> List[Path]:
    """Find every supported video file within a folder.

    Lets a whole session's worth of footage (e.g. one folder per
    recording day, with per-camera or per-athlete subfolders) be
    registered in one go instead of listing every file by hand.

    Args:
        folder: Folder to search.
        recursive: If True (default), also search subfolders --
            handles footage organized in nested subfolders (e.g.
            .../20260611_S01/IPadH/) in one call. Pass False to only
            look directly inside folder, not its subfolders.

    Returns:
        Sorted list of paths to every file under folder whose
        extension is in SUPPORTED_EXTENSIONS (case-insensitive, so
        both .MOV and .mov are picked up).

    Raises:
        FileNotFoundError: If folder does not exist or is not a
            directory.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Video folder not found: {folder}")

    pattern = "**/*" if recursive else "*"
    found = [
        path
        for path in folder.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(found)


def register_videos(
    video_paths: List[Path], session_id: str
) -> List[VideoFile]:
    """Register a batch of video files for a session.

    Supports batch registration of an entire team's videos at once,
    so uploads can run in the background during training/warm-up
    while the actual analysis queue processes them one by one later.

    Args:
        video_paths: Paths to video files to register.
        session_id: Identifier for the session these videos belong
            to.

    Returns:
        List of VideoFile objects, one per valid input path.

    Raises:
        FileNotFoundError: If any path does not exist.
        ValueError: If any file has an unsupported extension.
    """
    registered = []
    for index, raw_path in enumerate(video_paths, start=1):
        video_path = Path(raw_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        if video_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported video format {video_path.suffix!r} for "
                f"{video_path.name!r}. Supported: {SUPPORTED_EXTENSIONS}"
            )
        video_id = f"V{index:03d}"
        registered.append(
            VideoFile(
                video_id=video_id,
                path=video_path,
                session_id=session_id,
            )
        )
    return registered
