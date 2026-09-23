"""Sanity check on the pixels-per-metre calibration, derived from the
jump itself (added 2026-09-03).

WHY THIS EXISTS
---------------
The whole /events/ chain compares the athlete's vertical acceleration
against -9.81 m/s^2, and that comparison only means anything if the
pixel-to-metre scale is right. Nothing in the pipeline ever checked
that: the scale came from one head+feet click in /who/ and was taken
on trust from there on.

That went wrong on real footage. Until 2026-09-03,
Session.pixels_per_meter_by_resolution held ONE value per resolution,
shared across every video in the run. In the try-out set the camera
stood at 8 m for conditions C00-C03 and at 6 m for C04-C07, so a
single value could not be right for both. Measured against the
manual reference, the shared value was 25-28% too high for the 8 m
videos, which made their flight phase read about -7.3 m/s^2 instead
of -9.81 -- right at the edge of the +/- 3 m/s^2 tolerance band
instead of in the middle of it. Result: 8 of 21 jumps missed
outright at 60fps, 5 of 21 at 240fps, and roughly double the
flight-time shrink of the 6 m videos. With the scale corrected, both
dropped to 0 of 21. (See
claude/hoogte-accuracy-diagnose-2026-09-03.md in the project for the
full analysis and the numbers.)

The per-video calibration added the same day fixes the cause. This
module is the net underneath it: a wrong or forgotten calibration
click can still happen, and this makes it visible in the log
straight away instead of only after a manual validation study.

HOW IT WORKS
------------
During flight the only force on the athlete is gravity, so the hip
traces a parabola whose second derivative is g -- in PIXELS per
second squared. Fit a parabola to the raw hip_y_px over the detected
flight window and that second derivative divided by 9.81 is the
pixels-per-metre the footage itself implies. Compare it against the
value actually used; a large gap means the calibration click is off
(or belongs to a different camera distance).

This is deliberately independent of the calibration click: it uses
no athlete height, no clicked positions, only the raw tracked hip
signal and the known value of g. That independence is the point --
it is what made the 8 m error findable in the first place.

WHAT IT IS NOT
--------------
Not a replacement calibration. The estimate is noisy per jump (a
short flight window at 60fps gives few samples, and interpolated
frames flatten the curve), so it is reported as a median over an
athlete's jumps and used only to raise a warning, never to silently
rescale anything. A hip landmark that drifts relative to the centre
of mass during flight (knees tucking, for example) also biases it
slightly. Treat a flagged track as "look at this", not as "the true
value is X".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

GRAVITY_M_S2 = 9.81

# How far the implied pixels-per-metre may sit from the used value
# before it is worth a warning. 0.15 (15%) is chosen to sit well
# above the per-jump scatter seen on the try-out set (a few percent
# between jumps of the same athlete) but well below the error that
# actually caused trouble there (25-28%).
DEFAULT_TOLERANCE_FRACTION = 0.15

# A quadratic fit needs a handful of samples to mean anything. At
# 60fps a minimum-length flight window (0.13s) is about 8 frames, so
# 5 leaves room for the margin frames below.
MIN_SAMPLES_FOR_FIT = 5


@dataclass(frozen=True)
class CalibrationCheck:
    """Outcome of comparing the used pixels-per-metre against what the
    athlete's own flight phase implies.

    Attributes:
        used_pixels_per_meter: The value the pipeline actually ran
            with for this video.
        implied_pixels_per_meter: Median of the per-jump estimates
            from the flight parabola, or None if no jump had a usable
            flight window.
        ratio: used / implied. 1.0 is perfect; >1 means the used
            value is too high (acceleration reads too small, jumps
            get missed); <1 means too low.
        jumps_used: How many jumps produced a usable estimate.
        within_tolerance: False when the ratio is far enough from 1
            to be worth flagging. True when it is fine, and also True
            when there was nothing to check (no usable jumps) -- an
            absent measurement is not evidence of a problem.
    """

    used_pixels_per_meter: float
    implied_pixels_per_meter: Optional[float]
    ratio: Optional[float]
    jumps_used: int
    within_tolerance: bool


def implied_pixels_per_meter_for_jump(
    track,
    jump,
    margin_frames: int = 1,
) -> Optional[float]:
    """Estimate pixels-per-metre from one jump's flight phase.

    Args:
        track: The athlete's AthleteTrack (needs frame_indices,
            hip_y and fps).
        jump: A Jump (or JumpEvents) carrying takeoff_frame and
            landing_frame.
        margin_frames: Frames to drop at each end of the flight
            window before fitting. The detected window already sits
            INSIDE the true flight (the detector's edges are
            conservative -- see events/detector.py), so 1 is enough
            to stay clear of the transition frames.

    Returns:
        The implied pixels-per-metre, or None when the flight window
        is too short to fit, the frames are not in this track, or the
        fit comes out non-physical (an upward "acceleration", which
        means the window is not a real free flight).
    """
    frame_indices = list(track.frame_indices)
    try:
        takeoff_local = frame_indices.index(jump.takeoff_frame)
        landing_local = frame_indices.index(jump.landing_frame)
    except ValueError:
        return None

    first = takeoff_local + 1 + margin_frames
    last = landing_local - margin_frames
    if last - first < MIN_SAMPLES_FOR_FIT:
        return None

    times_s = np.asarray(frame_indices[first:last], dtype=float) / float(track.fps)
    # Pixel y grows downward, so flip the sign to get a height-like
    # signal -- same convention as events/detector.py.
    height_px = -np.asarray(track.hip_y[first:last], dtype=float)
    if not np.all(np.isfinite(height_px)):
        return None

    coefficients = np.polyfit(times_s, height_px, 2)
    acceleration_px_s2 = 2.0 * float(coefficients[0])
    if acceleration_px_s2 >= 0:
        return None
    return abs(acceleration_px_s2) / GRAVITY_M_S2


def check_calibration(
    track,
    jumps: Sequence,
    used_pixels_per_meter: float,
    tolerance_fraction: float = DEFAULT_TOLERANCE_FRACTION,
) -> CalibrationCheck:
    """Compare the used pixels-per-metre against the athlete's jumps.

    Args:
        track: The athlete's AthleteTrack.
        jumps: The jumps detected for this athlete in this video.
        used_pixels_per_meter: The value the pipeline ran with.
        tolerance_fraction: Relative gap allowed before flagging --
            see DEFAULT_TOLERANCE_FRACTION.

    Returns:
        A CalibrationCheck. Never raises: a track with no usable
        flight window simply comes back with implied_pixels_per_meter
        None and within_tolerance True.
    """
    estimates = [
        estimate
        for estimate in (
            implied_pixels_per_meter_for_jump(track, jump) for jump in jumps
        )
        if estimate is not None
    ]
    if not estimates or not used_pixels_per_meter:
        return CalibrationCheck(
            used_pixels_per_meter=used_pixels_per_meter,
            implied_pixels_per_meter=None,
            ratio=None,
            jumps_used=0,
            within_tolerance=True,
        )

    implied = float(np.median(estimates))
    ratio = used_pixels_per_meter / implied
    return CalibrationCheck(
        used_pixels_per_meter=used_pixels_per_meter,
        implied_pixels_per_meter=implied,
        ratio=ratio,
        jumps_used=len(estimates),
        within_tolerance=abs(ratio - 1.0) <= tolerance_fraction,
    )
