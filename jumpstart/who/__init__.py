"""Who module: interactive athlete order-in-frame assignment.

/who/ is the step where the user tells the script which participant
profile corresponds to which athlete position in frame, left to
right. Filling this in for a video is what starts that video's place
in the analysis queue (/pose/, /tracking/, /identify/, ...).
"""

from jumpstart.who.assignment import (
    FrameAssignment,
    PixelsPerMeterCache,
    assign_athlete_order,
    pixel_height_for_athlete,
    pixels_per_meter_from_click,
)
from jumpstart.who.frame_picker import extract_reference_frame

__all__ = [
    "FrameAssignment",
    "PixelsPerMeterCache",
    "assign_athlete_order",
    "pixel_height_for_athlete",
    "pixels_per_meter_from_click",
    "extract_reference_frame",
]
