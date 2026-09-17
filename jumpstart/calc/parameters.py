"""RTP parameter calculation for /calc/.

NEEDS REVIEW -- see the boundary marker in workflow_demo.py.

Computes Tier 1 (flight time, jump height, takeoff velocity, RSImod)
and Tier 2 (countermovement depth, peak power) parameters from a
Jump's start/takeoff/landing frames.

The canonical export row deliberately excludes condition/framerate/
resolution/video_id -- those live in the video's filename and can be
parsed out on demand (jumpstart.io.video_filename_parser) by anyone
who needs them, e.g. a study-specific importer script. This module
only carries what every user of the product needs regardless of
their own file naming: which video and athlete, which jump, when the
events happened, and the resulting parameters.

Known gap: countermovement depth needs a real pixel-to-metre
calibration (e.g. a known reference distance visible in frame) to be
reported in metres -- that calibration step does not exist anywhere
in this codebase yet, so it is reported in PIXELS here and clearly
named/flagged as such. Flight-time-derived parameters (flight time,
jump height, takeoff velocity, RSImod) do NOT need this calibration,
since they only use time. Peak power (Sayers) also does not need it,
since it only uses jump height (from flight time) and body mass.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from jumpstart.calc.physiological_bounds import classify_flight_time
from jumpstart.jumps.segmenter import Jump
from jumpstart.profiles.models import Participant
from jumpstart.tracking.tracker import AthleteTrack

GRAVITY_M_S2 = 9.81


@dataclass(frozen=True)
class JumpParameters:
    """RTP parameters computed for one jump.

    Attributes:
        video: The video's filename (e.g.
            "20260611_S01_ipadH_C00_120fps_1080p.mp4"), kept as the
            single video reference for this row. Condition,
            framerate, etc. are intentionally not duplicated as
            separate columns -- parse them from this field if needed.
        recorded_at: Date and time the video was actually shot, if it
            could be determined (see
            jumpstart.io.video_metadata.get_recording_datetime).
            None if it could not be determined -- the filename alone
            only carries a date, not a time.
        participant_id: Which athlete this belongs to.
        jump_number: 1-based index of this jump within this athlete's
            video, in chronological order.
        start_frame: Absolute video frame index of start-of-movement.
        takeoff_frame: Absolute video frame index of takeoff.
        landing_frame: Absolute video frame index of landing.
        start_time_s: start_frame converted to seconds from video
            start.
        takeoff_time_s: takeoff_frame converted to seconds from video
            start.
        landing_time_s: landing_frame converted to seconds from video
            start.
        flight_time_s: Time in the air (Tier 1).
        jump_height_m: Jump height from flight time (Tier 1).
        takeoff_velocity_m_s: Vertical velocity at takeoff, derived
            from flight time (Tier 1).
        time_to_takeoff_s: Duration of the countermovement
            (start-of-movement to takeoff).
        rsi_modified: Jump height / time-to-takeoff (Tier 1,
            conditional on time-to-takeoff reliability, per the
            project's RTP parameter framework).
        pixels_per_meter: The pixel-to-metre scale this jump was
            computed with, carried through so the export shows which
            calibration produced these numbers. None when the caller
            did not pass it.
        pixels_per_meter_implied: What this jump's own flight phase
            implies the scale should be (see
            jumpstart.calc.calibration_check). A large gap with
            pixels_per_meter means the calibration is off. None when
            the flight window was too short to estimate it.
        outside_physiological_bounds: True when the flight time falls
            outside what is physiologically possible (see
            jumpstart.calc.physiological_bounds) -- almost certainly a
            detection artifact, not a real jump. Flag only, never used
            to drop the row.
        unusually_low_jump: True when the jump is real but unusually
            low for an athlete (see jumpstart.calc.physiological_bounds),
            e.g. a return-to-play jump. Worth a second look, not an
            error.
        countermovement_depth_px: Maximum downward hip displacement
            during the countermovement, in PIXELS -- not metres. Not
            a finished Tier 2 parameter: needs pixel-to-metre
            calibration (does not exist yet) to be meaningful.
        peak_power_w: Peak power via the Sayers equation (Tier 2).
            None if the participant has no body mass on file.
    """

    video: str
    recorded_at: Optional[datetime]
    participant_id: str
    jump_number: int
    start_frame: int
    takeoff_frame: int
    landing_frame: int
    start_time_s: float
    takeoff_time_s: float
    landing_time_s: float
    flight_time_s: float
    jump_height_m: float
    takeoff_velocity_m_s: float
    time_to_takeoff_s: float
    rsi_modified: float
    countermovement_depth_px: float
    peak_power_w: Optional[float]
    # Toegevoegd 2026-09-03: de px/m waarmee deze sprong is
    # doorgerekend, plus wat de vluchtfase zelf impliceert (zie
    # jumpstart/calc/calibration_check.py). Naast elkaar in de export
    # maken ze een verkeerde kalibratie meteen zichtbaar.
    pixels_per_meter: Optional[float] = None
    pixels_per_meter_implied: Optional[float] = None
    # Toegevoegd 2026-09-09: fysiologische-grenzenfilter, zie
    # jumpstart/calc/physiological_bounds.py.
    outside_physiological_bounds: bool = False
    unusually_low_jump: bool = False
    # Toegevoegd 2026-09-03: de px/m waarmee deze sprong is
    # doorgerekend, zodat achteraf in de export te zien is welke
    # kalibratie gebruikt is.
    pixels_per_meter: Optional[float] = None


def _sayers_peak_power_w(jump_height_cm: float, body_mass_kg: float) -> float:
    """Estimate peak power (W) with the Sayers equation.

    PP = 60.7 * jump_height_cm + 45.3 * body_mass_kg - 2055

    From Sayers et al. (1999); developed for countermovement jumps
    performed without an arm swing. Verify that assumption matches
    how the Jumpstart jumps are actually performed before trusting
    the output.
    """
    return 60.7 * jump_height_cm + 45.3 * body_mass_kg - 2055


def calculate_jump_parameters(
    jump: Jump,
    track: AthleteTrack,
    participant: Participant,
    video_filename: str,
    recorded_at: Optional[datetime] = None,
    pixels_per_meter: Optional[float] = None,
    pixels_per_meter_implied: Optional[float] = None,
) -> JumpParameters:
    """Calculate RTP parameters for one detected jump.

    Args:
        jump: The Jump (start/takeoff/landing frames) to calculate
            parameters for.
        track: The athlete's AthleteTrack this jump was found in
            (used for the pixel-based countermovement-depth measure).
        participant: The athlete's profile (used for body mass in
            the Sayers peak-power estimate).
        video_filename: The source video's filename, carried through
            as the single video reference for this row (see the
            JumpParameters.video docstring).
        recorded_at: Date and time the video was shot, if known (see
            JumpParameters.recorded_at). Defaults to None.

    Returns:
        A JumpParameters with every Tier 1 parameter filled in, plus
        the pixel-based countermovement depth and (if body mass is
        known) peak power.
    """
    flight_time_s = (jump.landing_frame - jump.takeoff_frame) / jump.fps
    jump_height_m = GRAVITY_M_S2 * flight_time_s**2 / 8
    takeoff_velocity_m_s = GRAVITY_M_S2 * flight_time_s / 2
    time_to_takeoff_s = (jump.takeoff_frame - jump.start_frame) / jump.fps
    rsi_modified = (
        jump_height_m / time_to_takeoff_s
        if time_to_takeoff_s > 0
        else float("nan")
    )
    # Toegevoegd 2026-09-09: fysiologische-grenzenfilter, werkt op de
    # vluchttijd -- zie jumpstart/calc/physiological_bounds.py voor de
    # onderbouwing van de grenzen.
    bounds_check = classify_flight_time(flight_time_s)

    frame_mask = (track.frame_indices >= jump.start_frame) & (
        track.frame_indices <= jump.takeoff_frame
    )
    hip_y_during_countermovement = track.hip_y[frame_mask]
    if hip_y_during_countermovement.size > 0:
        standing_y = track.hip_y[track.frame_indices == jump.start_frame]
        baseline_y = (
            float(standing_y[0])
            if standing_y.size > 0
            else float(hip_y_during_countermovement[0])
        )
        # Pixel y increases downward, so the deepest countermovement
        # point is the MAXIMUM y reached before takeoff.
        countermovement_depth_px = float(
            np.max(hip_y_during_countermovement) - baseline_y
        )
    else:
        countermovement_depth_px = float("nan")

    peak_power_w = None
    if participant.weight_kg is not None:
        peak_power_w = _sayers_peak_power_w(
            jump_height_cm=jump_height_m * 100,
            body_mass_kg=participant.weight_kg,
        )

    return JumpParameters(
        video=video_filename,
        recorded_at=recorded_at,
        participant_id=jump.participant_id,
        jump_number=jump.jump_number,
        start_frame=jump.start_frame,
        takeoff_frame=jump.takeoff_frame,
        landing_frame=jump.landing_frame,
        start_time_s=jump.start_frame / jump.fps,
        takeoff_time_s=jump.takeoff_frame / jump.fps,
        landing_time_s=jump.landing_frame / jump.fps,
        flight_time_s=flight_time_s,
        jump_height_m=jump_height_m,
        takeoff_velocity_m_s=takeoff_velocity_m_s,
        time_to_takeoff_s=time_to_takeoff_s,
        rsi_modified=rsi_modified,
        countermovement_depth_px=countermovement_depth_px,
        peak_power_w=peak_power_w,
        pixels_per_meter=pixels_per_meter,
        pixels_per_meter_implied=pixels_per_meter_implied,
        outside_physiological_bounds=bounds_check.outside_physiological_bounds,
        unusually_low_jump=bounds_check.unusually_low_jump,
    )
