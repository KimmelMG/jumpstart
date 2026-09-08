"""Parsing of the iPad video filename convention.

This module reads structure out of a video's filename for internal
use only -- e.g. normalizing the condition label so it can be used
to build the same "Key (intern)" string as the validation
spreadsheet (participant|condition|framerate|quality|jump). It does
not rename or otherwise touch any file on disk, and its output is
not meant to be duplicated as separate columns in the Jumpstart
export table -- the raw filename itself is kept as the single
"video" reference there.

Naming convention (iPad only; camcorder videos are not analysed by
the Jumpstart script and are therefore not covered here):
    YYYYMMDD_S##_ipad{H|L}_C##_###fps_###p

Example:
    20260611_S01_ipadH_C00_120fps_1080p
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

IPAD_FILENAME_PATTERN = re.compile(
    r"^(?P<date>\d{8})_"
    r"(?P<session>S\d{2})_"
    r"ipad(?P<camera_position>[HL])_"
    r"C(?P<condition>\d{2})_"
    r"(?P<framerate>\d+)fps_"
    r"(?P<resolution>\d+p)$"
)


@dataclass(frozen=True)
class ParsedVideoInfo:
    """Structured fields extracted from an iPad video filename.

    Attributes:
        date: Recording date as YYYYMMDD.
        session: Session identifier, e.g. "S01".
        camera_position: "H" (ipad hoog) or "L" (ipad laag).
        condition: Normalized condition, e.g. "c0" -- leading zero
            and capital C stripped to match the convention already
            used in the validation spreadsheet's Key column.
        framerate: Framerate in fps.
        resolution: Resolution label, e.g. "1080p".
    """

    date: str
    session: str
    camera_position: str
    condition: str
    framerate: int
    resolution: str


def parse_ipad_filename(
    filename: Union[str, Path]
) -> Optional[ParsedVideoInfo]:
    """Parse an iPad video filename into structured fields.

    This is read-only: it never renames or modifies the file. Use
    it wherever normalized fields are needed internally (e.g. to
    build a matching Key for the validation spreadsheet), while the
    Jumpstart export table itself keeps the plain filename as a
    single "video" field.

    Args:
        filename: Filename, filename stem, or full path of an iPad
            video. Extension (if present) is ignored.

    Returns:
        A ParsedVideoInfo on a successful match, or None if the
        name does not follow the expected convention (e.g. a
        camcorder video).
    """
    stem = Path(filename).stem
    match = IPAD_FILENAME_PATTERN.match(stem)
    if match is None:
        return None
    fields = match.groupdict()
    condition_number = int(fields["condition"])
    return ParsedVideoInfo(
        date=fields["date"],
        session=fields["session"],
        camera_position=fields["camera_position"],
        condition=f"c{condition_number}",
        framerate=int(fields["framerate"]),
        resolution=fields["resolution"],
    )
