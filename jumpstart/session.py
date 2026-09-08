"""Session container: bundles profiles and registered videos.

A Session is the hand-off point between the preparation steps
(/profiles/, /io/) and the analysis pipeline (/who/, /pose/,
/tracking/, /identify/, /events/, /jumps/, /qc/, /calc/, /export/).
Filling in /who/ (order-of-athletes-in-frame) is what starts the
job queue -- this module stops right before that point.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple
from uuid import uuid4

from jumpstart.io.video_input import VideoFile
from jumpstart.profiles.models import Participant
from jumpstart.who.assignment import FrameAssignment


@dataclass
class Session:
    """A single recording session ready to be handed off to /who/.

    Attributes:
        session_id: Unique identifier for this session.
        participants: Profiles of athletes taking part.
        videos: Registered videos belonging to this session.
        who_assignments: Athlete order-in-frame per video, keyed by
            video_id. Populated by /who/ once the user has assigned
            it; an empty dict means no video has been assigned yet.
        pixels_per_meter_by_resolution: Real pixels-per-metre
            established so far this run, keyed by
            (frame_width_px, frame_height_px) -- the ACTUAL decoded
            resolution, not a filename label. Shared across every
            video's /who/ assignment so calibrating one athlete once
            per resolution is enough (see
            jumpstart.who.assignment's 2026-07-29 docstring note).
            NOT persisted between separate script runs -- restart
            the script and this resets, requiring one fresh
            calibration click per resolution again.
        created_at: Timestamp of session creation.
    """

    session_id: str = field(default_factory=lambda: uuid4().hex[:8])
    participants: List[Participant] = field(default_factory=list)
    videos: List[VideoFile] = field(default_factory=list)
    who_assignments: Dict[str, FrameAssignment] = field(
        default_factory=dict
    )
    pixels_per_meter_by_resolution: Dict[Tuple[int, int], float] = field(
        default_factory=dict
    )
    created_at: datetime = field(default_factory=datetime.now)

    def summary(self) -> str:
        """Return a short, human-readable summary of the session."""
        lines = [
            f"Session {self.session_id} "
            f"(created {self.created_at:%Y-%m-%d %H:%M})",
            f"  Participants: {len(self.participants)}",
        ]
        for participant in self.participants:
            lines.append(
                f"    - {participant.participant_id}: "
                f"{participant.name}"
            )
        lines.append(f"  Videos registered: {len(self.videos)}")
        for video in self.videos:
            assignment = self.who_assignments.get(video.video_id)
            if assignment is None:
                status = "order-in-frame not yet assigned"
            else:
                status = (
                    f"{len(assignment.participant_ids)} athlete(s) "
                    f"assigned (frame {assignment.frame_index})"
                )
            lines.append(
                f"    - {video.video_id}: {video.path.name} "
                f"[{status}]"
            )
        if self.pixels_per_meter_by_resolution:
            lines.append(
                "  Pixel-kalibraties dit run: "
                f"{len(self.pixels_per_meter_by_resolution)} resolutie(s) "
                f"({', '.join(f'{w}x{h}' for w, h in sorted(self.pixels_per_meter_by_resolution))})"
            )
        if len(self.who_assignments) < len(self.videos):
            lines.append(
                "  Next step: assign athlete order-in-frame in "
                "/who/ for the remaining video(s) to start the "
                "analysis queue."
            )
        else:
            lines.append(
                "  All videos assigned -- ready for /pose/ to start "
                "the analysis queue."
            )
        return "\n".join(lines)
