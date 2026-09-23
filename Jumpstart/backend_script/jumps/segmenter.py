"""Jump segmentation for /jumps/.

NEEDS REVIEW -- see the boundary marker in workflow_demo.py.

Turns the local, array-index jump events found by /events/ into
absolute video frame numbers (an athlete's track does not necessarily
start at video frame 0), numbers them per athlete in chronological
order, and gives each one a stable identifier for /calc/ and
/export/.

IMPORTANT (added 2026-07-29, confirmed on real footage): pixels-per-
metre is no longer applied as one flat constant across every video.
An earlier version of this module rescaled one manually-guessed
reference value by each video's decoded resolution -- that fixed the
resolution-driven part of the scale mismatch (confirmed on real
data: the same physical session, encoded once at 1080p and once at
480p, gave wildly inconsistent jump counts per athlete under a
single flat constant, because the free-flight acceleration check in
/events/ compares against a fixed physical tolerance around -9.81
m/s^2, and a wrong pixel scale shifts computed acceleration out of
that band by an amount that depends on the video), but the reference
value itself was still never measured against anything real.

This has now been replaced: pixels_per_meter is passed into
build_jumps_for_athlete already resolved to a real value for this
athlete's actual resolution. Pixels-per-metre is a property of the
camera setup (distance/zoom/resolution) at a given depth, not of any
one athlete, so /who/ only measures it once per resolution -- from
one athlete's known height against their own head-to-feet pixel span
-- and shares that value across every other athlete at the same
resolution, in this video and every later video in the same run (see
jumpstart.who.assignment's 2026-07-29 docstring note and
Session.pixels_per_meter_by_resolution). The real-world calibration
gap flagged in the /calc/ module docstring is now addressed at the
source, not worked around.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from backend_script.events.detector import (
    ONSET_DROP_THRESHOLD_M,
    detect_jumps_from_track,
)
from backend_script.tracking.tracker import AthleteTrack

# Below this many frames with an actual hip position, event detection
# (Savitzky-Golay smoothing + np.gradient in /events/) has too little
# signal to be meaningful, and np.gradient outright crashes below 2.
# A track this short means /tracking/ never matched this athlete to a
# detection beyond the /who/ seed frame -- e.g. occluded, out of frame
# for the whole video, or a /who/ click that didn't land near anyone.
# Confirmed on real footage: without this guard, one such athlete/
# video combination crashes the entire batch run after already paying
# the full (many-minutes) pose-detection cost for that video.
MIN_TRACK_FRAMES_FOR_EVENT_DETECTION = 5


@dataclass(frozen=True)
class Jump:
    """One identified jump for one athlete in one video.

    Attributes:
        participant_id: Which athlete this jump belongs to.
        video_id: Which video this jump was detected in.
        jump_number: 1-based index of this jump within this
            athlete's video, in chronological order.
        start_frame: Absolute video frame index of start-of-movement.
        takeoff_frame: Absolute video frame index of takeoff.
        landing_frame: Absolute video frame index of landing.
        fps: Frames per second, carried along so /calc/ does not
            need the original track.
    """

    participant_id: str
    video_id: str
    jump_number: int
    start_frame: int
    takeoff_frame: int
    landing_frame: int
    fps: float


def build_jumps_for_athlete(
    track: AthleteTrack,
    pixels_per_meter: float,
    tolerance_m_s2: float = 3.0,
    smoothing_window_s: float = 7 / 120,
    min_flight_duration_s: float = 0.13,
    refine_edges: bool = True,
    onset_mode: str = "drop",
    onset_drop_threshold_m: float = ONSET_DROP_THRESHOLD_M,
) -> List[Jump]:
    """Detect and number every jump for one athlete's track.

    Args:
        track: The athlete's AthleteTrack from /tracking/ +
            /identify/.
        pixels_per_meter: The real pixel-to-metre scale factor for
            this athlete's actual video resolution -- established
            once per resolution via /who/ (see the module docstring)
            and shared across every athlete/video at that
            resolution. Callers should skip an athlete entirely (not
            call this function with a guessed value) if no
            calibration exists yet for their resolution -- see the
            module docstring's
            2026-07-29 note.
        tolerance_m_s2: Passed through to detect_jumps_from_track.
        smoothing_window_s: Passed through to detect_jumps_from_track
            (see that module's 2026-07-29 docstring note on the
            derivative-noise fix this now feeds into).
        min_flight_duration_s: Passed through to
            detect_jumps_from_track -- see that module's 2026-07-30
            docstring note on why this needed raising after real
            footage still showed far too many false-positive jumps.

    Returns:
        Jump records in chronological order, numbered starting at 1.
        Empty if the track has too few frames to attempt event
        detection at all (see MIN_TRACK_FRAMES_FOR_EVENT_DETECTION) --
        printed as a warning rather than raised, so one bad athlete/
        video combination doesn't take down an entire batch run.
    """
    if len(track.frame_indices) < MIN_TRACK_FRAMES_FOR_EVENT_DETECTION:
        print(
            f"    {track.video_id}/{track.participant_id}: only "
            f"{len(track.frame_indices)} frame(s) with a hip position "
            "in this video (likely never matched a detection beyond "
            "the /who/ seed frame) -- need at least "
            f"{MIN_TRACK_FRAMES_FOR_EVENT_DETECTION} to attempt event "
            "detection. Reporting 0 jumps for this athlete/video."
        )
        return []

    local_events = detect_jumps_from_track(
        hip_y_px=track.hip_y,
        fps=track.fps,
        pixels_per_meter=pixels_per_meter,
        tolerance_m_s2=tolerance_m_s2,
        smoothing_window_s=smoothing_window_s,
        min_flight_duration_s=min_flight_duration_s,
        refine_edges=refine_edges,
        onset_mode=onset_mode,
        onset_drop_threshold_m=onset_drop_threshold_m,
    )

    jumps: List[Jump] = []
    for jump_number, events in enumerate(local_events, start=1):
        jumps.append(
            Jump(
                participant_id=track.participant_id,
                video_id=track.video_id,
                jump_number=jump_number,
                start_frame=int(track.frame_indices[events.start_frame]),
                takeoff_frame=int(
                    track.frame_indices[events.takeoff_frame]
                ),
                landing_frame=int(
                    track.frame_indices[events.landing_frame]
                ),
                fps=track.fps,
            )
        )
    return jumps
