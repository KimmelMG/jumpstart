"""IO module: registers videos for a recording session."""

from jumpstart.io.video_filename_parser import ParsedVideoInfo, parse_ipad_filename
from jumpstart.io.video_input import (
    SUPPORTED_EXTENSIONS,
    VideoFile,
    discover_videos_in_folder,
    register_videos,
)
from jumpstart.io.video_metadata import get_recording_datetime

__all__ = [
    "VideoFile",
    "register_videos",
    "discover_videos_in_folder",
    "SUPPORTED_EXTENSIONS",
    "ParsedVideoInfo",
    "parse_ipad_filename",
    "get_recording_datetime",
]
