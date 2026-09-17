"""Physiological/outlier bounds on flight-time-derived jump parameters.

WHY THIS EXISTS
----------------
Not every detected "jump" is a real jump: tracking noise, a bend-over
while handling the camera, or an oversegmented flight window can all
produce a flight-time reading that no human being could actually
achieve, or one so short it is really the detector's own noise floor.
Nothing in the pipeline flagged that -- a value like this exported
silently, indistinguishable from a real measurement.

DECISION (Mathijs, 2026-09-03, threshold set 2026-09-09)
----------------------------------------------------------
Fixed, absolute bounds -- not anchored to the run's own distribution.
Reason: this is software other people run on their own participants.
A bound that moves with the dataset means a flag on an unfamiliar set
tells you nothing about whether the jump is wrong or the rest of the
set is. (A median-anchored version was tested and works well as a
validation tool for comparing against manual analysis, but was
rejected as the product default for exactly this reason -- see
claude/fysiologische-grenzen-voorstel.md in the project for the full
empirical comparison of variants.)

Two severity levels, not one line:
1. Hard "impossible" -- below HARD_MIN_FLIGHT_TIME_S or above the
   flight time implied by HARD_MAX_JUMP_HEIGHT_M. Below/above this is
   not a low jump, it's an artifact.
2. Soft "unusually low for an athlete" -- below the flight time
   implied by SOFT_LOW_JUMP_HEIGHT_M, Mathijs' call (2026-09-09):
   10 cm. This is NOT physiologically impossible -- a return-to-play
   participant, or an older/deconditioned one, can genuinely jump
   4-10 cm, and that is real, clinically meaningful data for exactly
   this target group. A single hard floor at 10 cm would silently
   invalidate those measurements, so it gets a milder flag with a
   different meaning ("check whether this is real or a detection
   error"), not the same treatment as an impossible value.

There was already a lower bound: `min_flight_duration_s = 0.13 s` in
`_find_flight_windows` (events/detector.py) corresponds to exactly
2.1 cm of jump height -- that is why HARD_MIN_FLIGHT_TIME_S below is
set to the same 0.13 s rather than some other number: a filter set any
higher would just duplicate the detector's own floor with a second,
different value, and any lower would never fire. Keep the two in sync
by hand if either one ever changes; there is no shared constant
between the two modules to enforce it automatically.

WHICH PARAMETER
----------------
Applied to FLIGHT TIME, not to each derived measure separately. Jump
height, takeoff velocity and RSImod are all functions of flight time,
so a bound on flight time bounds the rest automatically and avoids a
jump being flagged on height but not on velocity. Bounds are defined
here in METRES (more intuitive) and converted internally via the same
h = g*t^2/8 relationship used in calc/parameters.py -- the relation is
monotonic, so this is equivalent to bounding height directly.
`countermovement_depth_px` is NOT covered by this filter: it's a pixel
measure tied to calibration and would need its own check.

FLAG, NEVER DROP
------------------
This is not a detail. The two errors found on 2026-09-03 (calibration,
savgol window) produced NO impossible values -- every one of the 340
test jumps fell inside real physiological bounds, while the whole set
was structurally ~0.15s short. A filter that silently dropped rows
would have made that bias invisible instead of visible, and a
tightly-set bound would have thrown away 24% of otherwise-correct
detections (see the empirical table in
claude/fysiologische-grenzen-voorstel.md). So: flag columns only.
Actually dropping rows is an explicit choice the user makes later,
during analysis -- never a default in this module or the export.
"""

from __future__ import annotations

from dataclasses import dataclass

GRAVITY_M_S2 = 9.81

# Hard lower bound: 0.13 s / ~2.1 cm. Deliberately equal to
# `min_flight_duration_s`'s default in events/detector.py -- see the
# module docstring above for why.
HARD_MIN_FLIGHT_TIME_S = 0.13

# Hard upper bound, expressed in metres and converted below. 1.25 m is
# a full second in the air; elite CMJ's sit at 0.6-0.8 s, so this
# practically never fires -- which is exactly what you want from an
# upper bound.
HARD_MAX_JUMP_HEIGHT_M = 1.25

# Soft "unusually low for an athlete" bound, in metres. Mathijs' call
# (2026-09-09): 10 cm. Below this a jump is not physiologically
# impossible -- see "DECISION" above -- just worth a second look.
SOFT_LOW_JUMP_HEIGHT_M = 0.10


def _flight_time_for_height_s(height_m: float) -> float:
    """Invert h = g*t^2/8 (the same formula calc/parameters.py uses to
    go from flight time to jump height) to get the flight time a given
    height corresponds to.
    """
    return (8.0 * height_m / GRAVITY_M_S2) ** 0.5


# Precomputed once at import time -- both bounds above are constants,
# so there's no reason to recompute this per jump.
HARD_MAX_FLIGHT_TIME_S = _flight_time_for_height_s(HARD_MAX_JUMP_HEIGHT_M)
SOFT_LOW_FLIGHT_TIME_S = _flight_time_for_height_s(SOFT_LOW_JUMP_HEIGHT_M)


@dataclass(frozen=True)
class PhysiologicalBoundsCheck:
    """Outcome of classifying one jump's flight time against the bounds.

    Attributes:
        outside_physiological_bounds: True when the flight time is
            below HARD_MIN_FLIGHT_TIME_S or above HARD_MAX_FLIGHT_TIME_S
            -- outside what is physiologically possible, almost
            certainly a detection artifact rather than a real jump.
        unusually_low_jump: True when the flight time is below
            SOFT_LOW_FLIGHT_TIME_S but still within the hard bounds --
            a real but low jump (e.g. return-to-play), worth a second
            look rather than a flag that something is wrong. Always
            False when outside_physiological_bounds is True -- the
            hard flag takes precedence, a jump is never both.
    """

    outside_physiological_bounds: bool
    unusually_low_jump: bool


def classify_flight_time(flight_time_s: float) -> PhysiologicalBoundsCheck:
    """Classify a jump's flight time against the physiological bounds.

    Flags only -- this function never decides to drop a row, and
    nothing downstream should either by default. See the module
    docstring's "FLAG, NEVER DROP" section for why.

    Args:
        flight_time_s: The jump's flight time in seconds (landing
            frame minus takeoff frame, divided by fps).

    Returns:
        A PhysiologicalBoundsCheck with exactly one flag set, or
        neither if the jump falls within normal bounds.
    """
    if flight_time_s < HARD_MIN_FLIGHT_TIME_S or flight_time_s > HARD_MAX_FLIGHT_TIME_S:
        return PhysiologicalBoundsCheck(
            outside_physiological_bounds=True, unusually_low_jump=False
        )
    if flight_time_s < SOFT_LOW_FLIGHT_TIME_S:
        return PhysiologicalBoundsCheck(
            outside_physiological_bounds=False, unusually_low_jump=True
        )
    return PhysiologicalBoundsCheck(
        outside_physiological_bounds=False, unusually_low_jump=False
    )
