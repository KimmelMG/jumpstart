"""Jump-event detection for /events/.

NEEDS REVIEW -- see the boundary marker in workflow_demo.py. Built in
one pass and only verified against synthetic jump-physics signals,
never real Jumpstart footage.

Detects start-of-movement, takeoff and landing from an athlete's
smoothed hip-height signal, using a free-flight heuristic: during the
airborne phase of a jump, the only force acting on the body's centre
of mass is gravity, so vertical acceleration should sit at
approximately -g for the whole flight.

Conventions (agreed 2026-07-15, adapted from standard force-plate
onset methodology -- see Bishop et al. 2022 on video-based jump
timing for the general problem this addresses):

    Takeoff: the LAST frame still in contact with the floor (one or
        two feet), i.e. one frame before the detected free-flight
        window begins.
    Landing: the FIRST frame back in contact with the floor, i.e.
        one frame after the detected free-flight window ends.
        (This "contact-to-contact" flight time is the conventional
        practical definition in video/jump-mat based analysis, and
        is what _find_flight_windows's window edges get shifted by
        in detect_jumps_from_track.)
    Start of movement (onset): NOT a fixed absolute velocity
        threshold. Instead:
            1. Signal: vertical hip velocity (derivative of the
               Savitzky-Golay-smoothed hip trajectory), not raw
               position -- no metric calibration needed, since the
               threshold is relative to per-trial baseline noise.
            2. Baseline window: computed per trial (i.e. per
               detected jump, not pooled across a whole video or
               across conditions, since pose-detection noise scales
               with camera distance/resolution). Found adaptively:
               scan backward from takeoff (up to ~2s, or back to the
               previous jump's landing frame if that is closer) for
               the ~15-20 frame sub-window with the lowest rolling
               standard deviation of velocity -- the "quietest"
               standing segment available before this particular
               jump.
            3. Threshold: threshold = baseline_mean - k * baseline_SD
               (k=5 by default, the standard force-plate convention).
               This is a DIRECTIONAL threshold (looking for velocity
               dropping below the baseline, i.e. downward movement
               starting), not a symmetric +/- band.
            4. Sustained crossing + back-trace: velocity must stay
               below threshold for >= min_consecutive_frames frames
               (default 3, ~25ms at 120fps) before being accepted --
               this filters single-frame pose-detection jitter, which
               is a real noise source on top of the smoothing.
               Once a qualifying run is found, onset is reported as
               the FIRST frame of that run, not the frame where the
               N-frame requirement was satisfied -- otherwise onset
               would be systematically late by min_consecutive_frames
               frames.

    Sanity check to run once real footage is available: plot each
    trial's baseline_SD against recording condition (distance/
    resolution/fps). It should increase as resolution drops or
    distance grows -- if it doesn't, the baseline window is probably
    not actually isolating pose-detection noise.

This is conceptually the same approach already built and used
separately in kmm_detector.py, reimplemented here as pure, unit-
testable functions (no interactive viewer) so it can run unattended
as part of the pipeline. All thresholds below are starting points,
not calibrated against real footage -- in particular, k=5 and
min_consecutive_frames=3 are worth a sensitivity check (e.g. k=3 vs
k=5) once real data is available, since they directly affect
time-to-takeoff and therefore RSImod.

Verified only against synthetic jump-physics signals in this
environment: on a clean, noiseless simulated countermovement jump,
detected flight time was within roughly 15-20% of the true simulated
value, and jump height within roughly 30%. That gap comes from the
Savitzky-Golay smoothing softening the sharp takeoff/landing edges --
expect to need to retune the smoothing/gap/onset durations below and
tolerance_m_s2 once real footage (with its own, different noise
characteristics) is available.

IMPORTANT (added 2026-07-28, confirmed on real footage): every
smoothing/gap/onset window below is expressed in SECONDS, not
frames, and converted to a frame count via fps internally. This
module originally used raw frame counts (e.g. a 7-frame smoothing
window), which meant the SAME frame count covered very different
real durations at different recording framerates -- 7 frames is
~29ms at 240fps but ~58ms at 60fps, so smoothing was effectively 4x
weaker at 240fps just because of the framerate, not the actual
motion. This produced inconsistent detection behaviour purely from
fps: on the same physical trial, the 240fps/1080p video detected
0/5 athletes' jumps while the 60fps/480p encode of the same
recording detected 1 jump each for 3/5 athletes. Since this
project's own validation design directly compares across 60fps and
240fps, that inconsistency would have confounded the comparison.
Time-based windows (converted to frames per-call via each track's
own fps) fix this: the same real-world smoothing/gap/onset behaviour
now applies regardless of recording framerate. All time defaults
below were chosen to match the previous frame defaults' *duration*
at the ~120fps they were originally tuned against (e.g. 7 frames at
120fps = 7/120s), so this is a like-for-like conversion, not a
retune.

IMPORTANT (added 2026-07-29, confirmed on real footage): even with
correct pixel-to-metre calibration (see jumpstart/who/assignment.py)
and a resolution-consistent tracking match tolerance (see
jumpstart/tracking/tracker.py), real footage still detected almost
no jumps. Root cause: velocity and acceleration used to be computed
by smoothing the position signal ONCE and then calling np.gradient
on it -- for acceleration, that means differentiating twice, and
each finite-difference (np.gradient) step divides by dt = 1/fps.
At high fps, dt is tiny, so this arithmetic massively AMPLIFIES
whatever pixel-level noise is still left after smoothing -- confirmed
with a synthetic test using this project's own real calibrated
pixels-per-metre values (201 px/m at 1080p, 89 px/m at 480p) plus a
modest, realistic amount of per-frame pixel jitter: the resulting
acceleration signal was too noisy for ANY reasonable tolerance to
reliably find the true flight window (either 0 windows, or several
spurious short ones from noise, depending on the tolerance).

Fix: velocity and acceleration are now computed directly via
scipy.signal.savgol_filter's own analytic derivative mode (deriv=1,
deriv=2), which fits ONE polynomial to a window of raw samples and
reads the derivative off that fit -- mathematically much better
behaved than smoothing once and then finite-differencing separately,
because the derivative comes from the same local fit as the
smoothing, instead of compounding two independent, noise-amplifying
operations. The smoothing/derivative window is also now floored at
MIN_DERIVATIVE_WINDOW_SAMPLES actual samples (not just a fixed real-
world duration), because at low fps a fixed-duration window covers
too few raw samples to average out noise, regardless of how long
that duration is in seconds.

Confirmed via synthetic testing with realistic noise: this reliably
recovers a single, correctly-located flight window per jump at high
fps (240fps), and a single window in most (not all) cases at low fps
(60fps) -- low-fps footage has less raw data to average over per
unit time and may still occasionally split one real jump into two
detected windows, or need further tuning once compared against real,
independently-verified flight times. Detected flight times remain
somewhat shorter than the true value (smoothing still softens the
sharp takeoff/landing edges, as already noted above) -- worth
checking against a manual/visual reference on a few real trials.

IMPORTANT (added 2026-07-30, confirmed via --debug-csv-dir exports of
real footage): even with the above fix, real 1080p/240fps footage
still showed near-total jump-detection failure, while the same
session's 480p/60fps encode worked much better. Comparing the actual
per-frame acceleration signal (not just synthetic tests) explained
why: MIN_DERIVATIVE_WINDOW_SAMPLES=25 is a floor on the number of
SAMPLES, not on real time, so it covers very different real
durations at different frame rates -- only ~105ms at 240fps (25/240)
versus ~417ms at 60fps (25/60). The Savitzky-Golay second derivative
used for acceleration amplifies per-frame position noise roughly by
1/dt^2, so a window that is both shorter in real time AND paired
with a 16x-smaller dt (240fps vs 60fps) is dramatically noisier --
confirmed on real debug-CSV data: acceleration standard deviation was
roughly 8-27 m/s^2 at 1080p/240fps versus 0.3-2.3 m/s^2 at 480p/60fps
for the same physical jumps, making tolerance_m_s2=3.0 essentially
useless at high fps. A real-time duration floor
(MIN_DERIVATIVE_WINDOW_DURATION_S) fixes this: at 60fps it changes
nothing (25 samples already covers ~417ms, more than the 0.3s floor),
but at 240fps it forces roughly 3x more real time into the window
than the sample-count floor alone would, and cut real-footage
acceleration noise roughly 5-10x in testing against this project's
own debug-CSV data (down to a range comparable to 480p).

This is a genuine trade-off, not a free win: a wider real-time window
also softens genuine short-duration signal more, and in testing
against real footage it sometimes causes brief noise excursions near
-9.81 m/s^2 (outside any real jump) to now last long enough to be
misread as very short "jumps" -- i.e. this fix trades some real
jumps being missed (before) for some risk of extra short false
positives (after), rather than eliminating noise sensitivity
entirely. tolerance_m_s2, max_gap_s and the minimum flight duration
in _find_flight_windows may need further tuning against more real
footage once this is validated; see also
jumpstart.tracking.tracker's 2026-07-30 docstring note for a
complementary fix on the /tracking/ side (rejecting some of the
physically-impossible position jumps before they ever reach this
module).

IMPORTANT (added 2026-07-30, same day, after a second real
full-pipeline run with both the above fix AND the /tracking/-side
fix applied): real footage still showed far too many detected jumps
(e.g. one athlete going from an expected 3 to 15-17), across BOTH
1080p and 480p. Inspecting the false positives directly (using the
frame ranges from --debug-csv-dir's detected_jump_phase column)
showed they were mostly scattered throughout the WHOLE video
timeline, not clustered around the real jumps -- i.e. brief (under
~150ms), noise-driven dips into the tolerance band during ordinary
non-jumping activity (standing, walking, adjusting stance), not
fragments of real jumps.

A grid search directly against this project's own real acceleration
signal (from the debug CSVs, so this is not a synthetic-noise
guess) over tolerance_m_s2, max_gap_s and the minimum flight-window
duration found two things:
    1. Counterintuitively, INCREASING max_gap_s (to bridge more
       noise) made results WORSE, not better: gap-bridging happens
       BEFORE the minimum-duration filter, so a bigger max_gap_s
       stitches together multiple individually-too-short noise blips
       into one now-long-enough-to-count fake window, adding false
       positives rather than removing them. max_gap_s is left at its
       existing default.
    2. Raising the minimum flight-window duration (previously an
       implicit ~0.1s) to 0.13s, combined with tolerance_m_s2=3.0,
       cut the total absolute deviation from the expected 3 jumps
       across 16 real athlete/video combinations from roughly 107-184
       down to 24 -- a real improvement, but not a complete fix: some
       athletes/videos (particularly a couple of 1080p tracks) still
       show noticeably too many or too few detected jumps. This is
       reported honestly as incremental progress, not a final answer;
       further tuning (or a shape-based check, e.g. requiring a real
       velocity swing before/after the window, which was tried
       informally but did not yet hold up reliably with a single
       global threshold) will likely be needed once more real footage
       is validated against ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter

GRAVITY_M_S2 = 9.81

# Floor for the Savitzky-Golay window used by _savgol_derivative, in
# actual SAMPLES rather than a fixed real-world duration. Confirmed
# necessary on real footage: a fixed-duration window covers too few
# raw samples to average out noise at low fps, regardless of how long
# that duration is in seconds -- see the module docstring's 2026-07-29
# note. 25 was found empirically (synthetic testing with realistic
# noise) to give reliable single-window flight detection at 240fps
# and mostly-reliable detection at 60fps; revisit if real footage at
# other framerates needs a different value.
MIN_DERIVATIVE_WINDOW_SAMPLES = 25

# Floor for the Savitzky-Golay window in real SECONDS, regardless of
# how many samples that covers -- added 2026-07-30 alongside
# MIN_DERIVATIVE_WINDOW_SAMPLES (see the module docstring's 2026-07-30
# note). A sample-count floor alone covers very different real
# durations at different frame rates (e.g. 25 samples is ~105ms at
# 240fps but ~417ms at 60fps), and the second derivative's noise
# amplification is sensitive to real time (roughly 1/dt^2), not just
# sample count -- confirmed on real footage: 1080p/240fps acceleration
# noise was 8-27 m/s^2 versus 0.3-2.3 m/s^2 on the same session's
# 480p/60fps encode, until this floor was added. 0.3s was found
# empirically (testing against this project's own real debug-CSV
# data) to bring 1080p/240fps noise down to a range comparable to
# 480p/60fps; revisit once more real footage (especially other frame
# rates) is available -- see the module docstring for the trade-off
# this introduces (softer, sometimes over-fragmented detection).
MIN_DERIVATIVE_WINDOW_DURATION_S = 0.3


def _savgol_derivative(
    position_signal: np.ndarray,
    fps: float,
    deriv: int,
    window_duration_s: float,
    polyorder: int = 3,
    min_window_samples: int = MIN_DERIVATIVE_WINDOW_SAMPLES,
    min_window_duration_s: float = MIN_DERIVATIVE_WINDOW_DURATION_S,
) -> np.ndarray:
    """Differentiate a position-like signal via one Savitzky-Golay fit.

    Deliberately does NOT smooth the signal and then separately call
    np.gradient -- that compounds two independent noise-amplifying
    operations (see the module docstring's 2026-07-29 note for why
    this matters so much for real footage). Instead, scipy's own
    analytic derivative mode reads the derivative directly off the
    SAME local polynomial fit used for smoothing, which is far less
    sensitive to per-frame pixel noise, especially at high fps where
    dt is small and finite-difference noise amplification is worst.

    Args:
        position_signal: A "height-like" signal (up = positive), in
            whatever unit the caller wants the deriv-th derivative
            expressed per second in (pixels for smooth_hip_height's
            deriv=0 pixel output, metres for velocity/acceleration in
            detect_jumps_from_track).
        fps: Frames per second -- sets dt = 1/fps for the derivative
            and converts window_duration_s into a sample count.
        deriv: Which derivative to return (0 = smoothed position,
            1 = velocity, 2 = acceleration).
        window_duration_s: Intended smoothing/derivative window
            length in SECONDS, converted to a sample count via fps,
            then raised to min_window_samples if that's larger (see
            MIN_DERIVATIVE_WINDOW_SAMPLES) and ALSO raised to
            min_window_duration_s converted to samples via fps if
            THAT's larger still (see MIN_DERIVATIVE_WINDOW_DURATION_S
            and the module docstring's 2026-07-30 note -- this second
            floor is the one that actually binds at high fps), then
            forced odd for savgol_filter, then reduced automatically
            for very short signals.
        polyorder: Polynomial order for the Savitzky-Golay fit.
        min_window_samples: Floor on the window length in actual
            samples -- see MIN_DERIVATIVE_WINDOW_SAMPLES.
        min_window_duration_s: Floor on the window length in real
            SECONDS -- see MIN_DERIVATIVE_WINDOW_DURATION_S.

    Returns:
        The deriv-th derivative of position_signal with respect to
        time (per second, per second^2, etc. matching deriv), same
        length as position_signal. Zeros (for deriv >= 1) or the
        unmodified signal (for deriv == 0) if the signal is too short
        to fit the requested polynomial/derivative combination.
    """
    window_length = max(
        int(round(window_duration_s * fps)),
        min_window_samples,
        int(round(min_window_duration_s * fps)),
    )
    window_length = min(window_length, len(position_signal))
    if window_length % 2 == 0:
        window_length -= 1
    if window_length <= polyorder or window_length <= deriv:
        return position_signal.copy() if deriv == 0 else np.zeros_like(position_signal)
    dt = 1.0 / fps
    return savgol_filter(position_signal, window_length, polyorder, deriv=deriv, delta=dt)


@dataclass(frozen=True)
class JumpEvents:
    """Start/takeoff/landing frame indices for one detected jump.

    Attributes:
        start_frame: Frame index where the countermovement (start of
            movement) begins.
        takeoff_frame: Frame index of takeoff (last ground contact).
        landing_frame: Frame index of landing (first ground contact
            after flight).
    """

    start_frame: int
    takeoff_frame: int
    landing_frame: int


def compute_kinematics(
    hip_y_px: np.ndarray,
    fps: float,
    pixels_per_meter: float,
    smoothing_window_s: float = 7 / 120,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute height/velocity/acceleration from a raw hip-y track.

    Pulled out of detect_jumps_from_track (added 2026-07-30) so a
    caller can inspect the actual signal that free-flight detection
    runs on, on real footage, without duplicating this arithmetic --
    added specifically to debug near-total jump-detection failures
    on real footage that synthetic-noise testing did not fully
    reproduce (see the module docstring's 2026-07-29 note): synthetic
    i.i.d. Gaussian pixel noise is not necessarily representative of
    real MediaPipe landmark jitter (e.g. correlated frame-to-frame
    drift, brief occlusion spikes), so the most reliable way to retune
    tolerance_m_s2/smoothing_window_s against real conditions is to
    look directly at this signal for a specific athlete/video rather
    than keep guessing from synthetic tests alone.

    Args:
        hip_y_px: Raw hip-midpoint y pixel position per frame.
        fps: Frames per second of the source video.
        pixels_per_meter: Pixel-to-metre scale factor for this
            video's resolution (see jumpstart/who/assignment.py).
        smoothing_window_s: Savitzky-Golay smoothing/derivative
            window length in seconds (see _savgol_derivative).

    Returns:
        (height_m, velocity_m_s, acceleration_m_s2), each the same
        length as hip_y_px.
    """
    height_m = -np.asarray(hip_y_px, dtype=float) / pixels_per_meter
    velocity_m_s = _savgol_derivative(
        height_m, fps, deriv=1, window_duration_s=smoothing_window_s
    )
    acceleration_m_s2 = _savgol_derivative(
        height_m, fps, deriv=2, window_duration_s=smoothing_window_s
    )
    return height_m, velocity_m_s, acceleration_m_s2


def smooth_hip_height(
    hip_y_px: np.ndarray,
    fps: float,
    window_duration_s: float = 7 / 120,
    polyorder: int = 3,
) -> np.ndarray:
    """Smooth a raw pixel-y hip trajectory with a Savitzky-Golay filter.

    Also flips sign so the output increases when the athlete moves
    UP (pixel y increases downward on screen, the opposite of real
    height) -- everything downstream in this module works in this
    "height-like" convention.

    Args:
        hip_y_px: Raw hip-midpoint y pixel position per frame.
        fps: Frames per second of the source video -- converts
            window_duration_s into an actual frame count, so the
            smoothing window covers the same real time regardless of
            recording framerate (see the module docstring's 2026-07-28
            note for why this matters), floored at
            MIN_DERIVATIVE_WINDOW_SAMPLES actual samples (see the
            2026-07-29 note).
        window_duration_s: Smoothing window length in SECONDS
            (default ~58ms, i.e. what 7 frames was at the 120fps this
            was originally tuned against). Converted to a frame count
            via fps, then forced odd for the Savitzky-Golay filter,
            and reduced automatically for very short signals.
        polyorder: Polynomial order for the Savitzky-Golay fit.

    Returns:
        Smoothed "height" signal (arbitrary units, up = positive),
        same length as hip_y_px.
    """
    height_px = -np.asarray(hip_y_px, dtype=float)
    return _savgol_derivative(
        height_px, fps, deriv=0,
        window_duration_s=window_duration_s, polyorder=polyorder,
    )


def _find_flight_windows(
    acceleration: np.ndarray,
    fps: float,
    tolerance_m_s2: float = 3.0,
    min_flight_frames: Optional[int] = None,
    max_gap_s: float = 3 / 120,
    min_flight_duration_s: float = 0.13,
) -> List["_FlightWindow"]:
    """Find contiguous windows where acceleration matches free fall.

    Args:
        acceleration: Acceleration signal in m/s^2 (see
            detect_jumps_from_track for how this is derived).
        fps: Frames per second, used to convert min_flight_duration_s
            and max_gap_s into frame counts.
        tolerance_m_s2: How far from -g the acceleration may drift
            and still count as "free flight". Needs tuning once
            pixel-to-metre calibration exists (see /calc/).
        min_flight_frames: Minimum window length in frames to count
            as a real jump rather than noise. If not given, computed
            from min_flight_duration_s via fps.
        min_flight_duration_s: Minimum window length in SECONDS,
            converted to a frame count via fps if min_flight_frames
            is not given directly (added 2026-07-30, raised from an
            implicit 0.1s -- see the module docstring's 2026-07-30
            note on why this needed real-footage-driven tuning, not
            just a round default).
        max_gap_s: Short below-tolerance blips of up to this many
            SECONDS (caused by smoothing/derivative noise right at
            the edge of a flight phase, converted to a frame count
            via fps) are bridged rather than treated as ending the
            window. Without this, a single noisy frame in the middle
            of a real flight phase can fragment it into several
            short, wrong-looking "jumps". Default ~25ms, i.e. what 3
            frames was at the 120fps this was originally tuned
            against (see the module docstring's 2026-07-28 note).
            IMPORTANT (2026-07-30): counterintuitively, RAISING this
            made real-footage results worse, not better -- see the
            module docstring's 2026-07-30 note.

    Returns:
        A list of _FlightWindow, one per candidate jump, in order.
    """
    if min_flight_frames is None:
        min_flight_frames = max(2, int(round(min_flight_duration_s * fps)))
    max_gap_frames = max(0, int(round(max_gap_s * fps)))

    is_free_fall = np.abs(acceleration - (-GRAVITY_M_S2)) <= tolerance_m_s2
    is_free_fall = _bridge_short_gaps(is_free_fall, max_gap_frames)

    windows: List[_FlightWindow] = []
    start: Optional[int] = None
    for index, free in enumerate(is_free_fall):
        if free and start is None:
            start = index
        elif not free and start is not None:
            if index - start >= min_flight_frames:
                windows.append(_FlightWindow(start, index - 1))
            start = None
    if start is not None and len(is_free_fall) - start >= min_flight_frames:
        windows.append(_FlightWindow(start, len(is_free_fall) - 1))
    return windows


def _bridge_short_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Fill in False runs of length <= max_gap_frames that sit
    between two True samples, so brief noise doesn't split one real
    flight window into several fragments."""
    if max_gap_frames <= 0:
        return mask
    bridged = mask.copy()
    gap_start: Optional[int] = None
    for index, value in enumerate(mask):
        if not value and gap_start is None:
            gap_start = index
        elif value and gap_start is not None:
            gap_length = index - gap_start
            if gap_start > 0 and gap_length <= max_gap_frames:
                bridged[gap_start:index] = True
            gap_start = None
    return bridged


@dataclass(frozen=True)
class _FlightWindow:
    """Internal: a candidate airborne window, in signal-index units."""

    start_index: int
    end_index: int


# Hoever buiten het grove vluchtvenster trap 2 mag zoeken naar de
# snelheidsextremen. Nodig omdat trap 1's randen BINNEN de echte
# vluchtfase liggen, dus de echte takeoff/landing zitten er net
# buiten. 0,30 s is ruim meer dan de grootste gemeten krimp per rand
# (~0,08 s) zonder een buursprong te raken: op 149 echte sprongen
# lag het extremum geen enkele keer op de zoekgrens.
REFINE_SEARCH_MARGIN_S = 0.30


def _refine_flight_edges(
    velocity_m_s: np.ndarray,
    fps: float,
    window: "_FlightWindow",
    search_margin_s: float = REFINE_SEARCH_MARGIN_S,
) -> Tuple[int, int]:
    """Trap 2: leg takeoff/landing precies binnen een grof venster.

    Trap 1 (_find_flight_windows) zegt WAAR de sprong zit; dit zegt
    waar hij precies begint en eindigt. De regel is fysisch, geen
    afgestelde drempel: de verticale snelheid is MAXIMAAL op takeoff
    (daarna werkt alleen zwaartekracht, dus hij kan alleen dalen) en
    MINIMAAL op landing (daarna remt de grond af). De twee extremen,
    elk aan hun eigen kant van het venster gezocht, ZIJN dus de twee
    events.

    Returns:
        (takeoff_index, landing_index) als signaalindices, direct --
        NIET verschoven met 1 zoals de grove conventie. Het
        snelheidsextremum ligt al op het laatste/eerste contactframe.
        Valt terug op de grove randen als het venster te kort is of
        het resultaat niet-fysisch uitkomt.
    """
    last_index = len(velocity_m_s) - 1
    coarse_takeoff = max(window.start_index - 1, 0)
    coarse_landing = min(window.end_index + 1, last_index)

    margin_frames = max(1, int(round(search_margin_s * fps)))
    search_start = max(0, window.start_index - margin_frames)
    search_end = min(last_index, window.end_index + margin_frames)
    if search_end - search_start < 3:
        return coarse_takeoff, coarse_landing

    # Splits op het midden van het grove venster zodat de
    # takeoff-kant en de landingskant apart doorzocht worden --
    # anders zou één argmax/argmin over het hele stuk beide events
    # uit dezelfde helft kunnen halen.
    middle = (window.start_index + window.end_index) // 2
    middle = min(max(middle, search_start), search_end)

    takeoff_index = search_start + int(
        np.argmax(velocity_m_s[search_start:middle + 1])
    )
    landing_index = middle + int(
        np.argmin(velocity_m_s[middle:search_end + 1])
    )
    if landing_index <= takeoff_index:
        return coarse_takeoff, coarse_landing
    return takeoff_index, landing_index


# How far outside the coarse flight window stage 2 may look for the
# velocity extremes (see the module docstring's 2026-09-03 note).
# Needed because stage 1's edges sit INSIDE the real flight phase, so
# the true takeoff/landing are usually just outside the coarse window.
# 0.30s is comfortably more than the largest shrink observed on real
# footage (~0.08s per edge) without reaching into a neighbouring jump:
# on 149 real jumps the extreme never once landed on the search
# boundary at takeoff and never at landing, so the margin is not the
# binding constraint.
REFINE_SEARCH_MARGIN_S = 0.30


def _refine_flight_edges(
    velocity_m_s: np.ndarray,
    fps: float,
    window: "_FlightWindow",
    search_margin_s: float = REFINE_SEARCH_MARGIN_S,
) -> Tuple[int, int]:
    """Stage 2: place takeoff/landing precisely inside a coarse window.

    Stage 1 (_find_flight_windows) says WHERE the jump is; this says
    exactly where it starts and ends. See the module docstring's
    2026-09-03 note for why the coarse edges are biased and why the
    velocity is used instead of the acceleration band.

    The rule is physical, not a tuned threshold: vertical velocity is
    at its MAXIMUM at takeoff (after that only gravity acts, so it can
    only decrease) and at its MINIMUM at landing (after that the
    ground decelerates the body). So the two extremes, searched on
    their own side of the window, ARE the two events.

    Args:
        velocity_m_s: The velocity signal compute_kinematics already
            produced for this track -- deliberately reused rather than
            recomputed with a narrower window, which was tested and
            gave no reliable improvement (module docstring).
        fps: Frames per second, used to convert search_margin_s into
            frames.
        window: The coarse flight window from stage 1.
        search_margin_s: How far outside the coarse window to look --
            see REFINE_SEARCH_MARGIN_S.

    Returns:
        (takeoff_index, landing_index) as signal indices, directly --
        NOT offset by one the way the coarse convention is. The
        velocity extreme already sits on the last/first contact frame,
        which is exactly the convention this module documents at the
        top. Falls back to the coarse edges (with their +/-1 shift) if
        the window is too short to search or the result comes out
        non-physical (landing at or before takeoff).
    """
    last_index = len(velocity_m_s) - 1
    coarse_takeoff = max(window.start_index - 1, 0)
    coarse_landing = min(window.end_index + 1, last_index)

    margin_frames = max(1, int(round(search_margin_s * fps)))
    search_start = max(0, window.start_index - margin_frames)
    search_end = min(last_index, window.end_index + margin_frames)
    if search_end - search_start < 3:
        return coarse_takeoff, coarse_landing

    # Split at the middle of the coarse window so the takeoff side and
    # the landing side are searched separately -- otherwise a single
    # argmax/argmin over the whole stretch could pick both events from
    # the same half.
    middle = (window.start_index + window.end_index) // 2
    middle = min(max(middle, search_start), search_end)

    takeoff_index = search_start + int(
        np.argmax(velocity_m_s[search_start:middle + 1])
    )
    landing_index = middle + int(
        np.argmin(velocity_m_s[middle:search_end + 1])
    )
    if landing_index <= takeoff_index:
        return coarse_takeoff, coarse_landing
    return takeoff_index, landing_index


def _select_adaptive_baseline_window(
    velocity: np.ndarray,
    search_start: int,
    search_end: int,
    fps: float,
    rolling_window_s: float = 18 / 120,
) -> Tuple[int, int]:
    """Find the quietest (lowest rolling-SD) sub-window of velocity
    within [search_start, search_end), to use as this trial's onset
    baseline.

    Args:
        velocity: Full velocity signal for the track.
        search_start: First index to consider (inclusive).
        search_end: Last index to consider (exclusive) -- normally
            the takeoff index, since baseline must come from before
            the jump.
        fps: Frames per second, used to convert rolling_window_s into
            an actual frame count.
        rolling_window_s: Length of the candidate baseline window, in
            SECONDS (~15-20 frames' worth at the 120fps this was
            originally tuned against -- see the module docstring's
            2026-07-28 note for why this is time-based, not a raw
            frame count).

    Returns:
        (baseline_start, baseline_end) indices (end exclusive) of the
        quietest window found.
    """
    rolling_window_frames = max(2, int(round(rolling_window_s * fps)))
    available = max(2, search_end - search_start)
    window = min(rolling_window_frames, available)
    if window < 2:
        return search_start, search_start + 2

    best_start = search_start
    best_sd = float("inf")
    last_possible_start = search_end - window
    for start in range(search_start, last_possible_start + 1):
        segment = velocity[start:start + window]
        sd = float(np.std(segment))
        if sd < best_sd:
            best_sd = sd
            best_start = start
    return best_start, best_start + window


# Hoeveel de heup onder de stahoogte moet zakken voordat de
# countermovement als begonnen geldt (verplaatsings-onset, 2026-09-10).
# Vervangt de snelheidsdrempel als standaard: die zette de drempel op
# baseline_mean - 5*SD van het STILSTE 0,15s-venster, en die SD is op
# een gesmoothed signaal ~7x kleiner dan de echte ruis, waardoor de
# effectieve drempel op ~0,7 SD lag en de onset ~0,43 s te vroeg viel
# (mediane heupverplaatsing op dat moment: 13 mm, in 19% van de
# sprongen minder dan 5 mm). Zie claude/onset-detectie-diagnose-
# 2026-09-10.md. 8 mm is de conventie van de beoordelaar: de eerst
# zichtbare beweging van de heup uit de standfase.
ONSET_DROP_THRESHOLD_M = 0.008


def _find_start_of_movement_by_drop(
    height_m: np.ndarray,
    velocity: np.ndarray,
    fps: float,
    takeoff_index: int,
    earliest_allowed_index: int = 0,
    drop_threshold_m: float = ONSET_DROP_THRESHOLD_M,
    max_lookback_s: float = 2.0,
    rolling_window_s: float = 18 / 120,
    smoothing_window_s: float = 7 / 120,
) -> Optional[int]:
    """Onset = eerste frame waarop de heup >= drop_threshold_m onder
    de stahoogte is gezakt, gevonden door TERUG te lopen vanaf het
    laagste punt van de countermovement.

    Waarom terug vanaf het laagste punt en niet vooruit vanaf de
    baseline: vooruit scannen accepteert het EERSTE dipje in een
    venster van maximaal 2 s, dus een toevallige rimpel wint van het
    echte begin van de beweging. Het laagste punt is eenduidig, en
    tussen onset en dat laagste punt daalt de heup monotoon, dus de
    drempel wordt op de terugweg precies een keer gekruist.

    Args:
        height_m: Hoogte-signaal (op-positief, meters) zoals
            compute_kinematics dat teruggeeft -- RUW, dus hier zelf
            gladgestreken voordat de drempel erop losgelaten wordt.
        velocity: Snelheidssignaal, alleen gebruikt om via
            _select_adaptive_baseline_window het stilste stukje te
            vinden waaruit de STAHOOGTE komt (de mediane hoogte daar).
            Let op: van dat venster wordt hier alleen de POSITIE
            gebruikt, niet de SD -- juist die SD was het probleem.
        fps: Frames per seconde.
        takeoff_index: Takeoff-frame van deze sprong.
        earliest_allowed_index: Niet verder terugkijken dan dit frame
            (de landing van de vorige sprong).
        drop_threshold_m: Zakking t.o.v. de stahoogte die als begin
            van de beweging telt -- zie ONSET_DROP_THRESHOLD_M.
        max_lookback_s: Hoever voor takeoff er gezocht mag worden.
        rolling_window_s: Lengte van het kandidaat-baselinevenster.
        smoothing_window_s: Savitzky-Golay-venster voor het gladstrijken
            van de positie (zelfde parameter als de rest van de module;
            de ruwe positieruis is mediaan 2,3 mm maar p90 11,9 mm, dus
            zonder gladstrijken kan de ruis zelf de 8mm-drempel halen).

    Returns:
        Het onset-frame, of None als er binnen het zoekvenster nooit
        meer dan drop_threshold_m gezakt wordt -- de aanroeper valt dan
        terug op _find_start_of_movement.
    """
    lookback_frames = int(round(max_lookback_s * fps))
    search_start = max(earliest_allowed_index, takeoff_index - lookback_frames)
    if takeoff_index - search_start < 2:
        return None

    baseline_start, baseline_end = _select_adaptive_baseline_window(
        velocity, search_start, takeoff_index,
        fps=fps, rolling_window_s=rolling_window_s,
    )
    smoothed = _savgol_derivative(
        np.asarray(height_m, dtype=float), fps, deriv=0,
        window_duration_s=smoothing_window_s,
    )
    standing = float(np.median(smoothed[baseline_start:baseline_end]))

    segment = smoothed[search_start:takeoff_index + 1]
    if segment.size < 2:
        return None
    drop = standing - segment
    bottom = int(np.argmax(drop))
    if drop[bottom] < drop_threshold_m:
        return None

    index = bottom
    while index > 0 and drop[index] >= drop_threshold_m:
        index -= 1
    return search_start + min(index + 1, bottom)


def _find_start_of_movement(
    velocity: np.ndarray,
    takeoff_index: int,
    earliest_allowed_index: int = 0,
    fps: float = 120.0,
    max_lookback_s: float = 2.0,
    rolling_window_s: float = 18 / 120,
    noise_multiplier: float = 5.0,
    min_consecutive_s: float = 3 / 120,
) -> int:
    """Find where the countermovement (start-of-movement) begins.

    Per the project's onset convention: find the quietest available
    per-trial baseline window before takeoff, set a directional
    threshold at noise_multiplier standard deviations below that
    baseline's mean velocity, then scan FORWARD from the baseline
    toward takeoff for the first run of min_consecutive_s worth of
    consecutive frames below that threshold. Onset is reported as
    the first frame of that run (not the frame where the run becomes
    long enough), so onset isn't systematically delayed.

    Args:
        velocity: Vertical hip velocity signal for the whole track
            ("height-like" convention: positive = moving up), same
            one used to derive acceleration for takeoff/landing.
        takeoff_index: Index of the takeoff frame for this jump;
            baseline and onset search never look past this point.
        earliest_allowed_index: Do not search for a baseline before
            this index -- pass the previous jump's landing_frame
            (or 0 for the first jump in a track) so one athlete's
            later jumps don't borrow a "quiet" window from inside an
            earlier jump's flight or landing impact.
        fps: Frames per second, used to convert every *_s duration
            below into an actual frame count, so onset detection
            behaves the same in real time regardless of recording
            framerate (see the module docstring's 2026-07-28 note).
        max_lookback_s: How far before takeoff (in seconds) to search
            for a quiet baseline window, capped by
            earliest_allowed_index.
        rolling_window_s: Candidate baseline window length, in
            SECONDS (~15-20 frames' worth at the 120fps this was
            originally tuned against).
        noise_multiplier: k in threshold = baseline_mean - k * SD.
            5 is the standard force-plate convention; worth a
            sensitivity check (e.g. k=3) against real footage.
        min_consecutive_s: Minimum run length, in SECONDS, below
            threshold required before accepting onset, to filter
            single-frame pose-detection jitter (default ~25ms, i.e.
            what 3 frames was at 120fps).

    Returns:
        The index of the first frame considered part of the
        countermovement (start-of-movement).
    """
    lookback_frames = int(round(max_lookback_s * fps))
    search_start = max(earliest_allowed_index, takeoff_index - lookback_frames)
    search_end = takeoff_index

    if search_end - search_start < 2:
        return search_start

    baseline_start, baseline_end = _select_adaptive_baseline_window(
        velocity, search_start, search_end,
        fps=fps, rolling_window_s=rolling_window_s,
    )
    baseline = velocity[baseline_start:baseline_end]
    baseline_mean = float(np.mean(baseline))
    baseline_sd = float(np.std(baseline)) or 1e-6
    threshold = baseline_mean - noise_multiplier * baseline_sd

    min_consecutive_frames = max(1, int(round(min_consecutive_s * fps)))

    run_start: Optional[int] = None
    for index in range(baseline_end, takeoff_index + 1):
        if velocity[index] < threshold:
            if run_start is None:
                run_start = index
            if index - run_start + 1 >= min_consecutive_frames:
                return run_start
        else:
            run_start = None

    # No sustained (>= min_consecutive_frames) run found -- fall back
    # to the first single-frame threshold crossing, if any, else just
    # report the end of the baseline window (no detected onset delay).
    for index in range(baseline_end, takeoff_index + 1):
        if velocity[index] < threshold:
            return index
    return baseline_end


def detect_jumps_from_track(
    hip_y_px: np.ndarray,
    fps: float,
    pixels_per_meter: float,
    tolerance_m_s2: float = 3.0,
    smoothing_window_s: float = 7 / 120,
    max_gap_s: float = 3 / 120,
    min_flight_duration_s: float = 0.13,
    refine_edges: bool = True,
    refine_search_margin_s: float = REFINE_SEARCH_MARGIN_S,
    onset_mode: str = "drop",
    onset_drop_threshold_m: float = ONSET_DROP_THRESHOLD_M,
    onset_k: float = 5.0,
    onset_min_consecutive_s: float = 3 / 120,
    onset_baseline_window_s: float = 18 / 120,
    onset_max_lookback_s: float = 2.0,
) -> List[JumpEvents]:
    """Detect every jump's start/takeoff/landing in one athlete track.

    Args:
        hip_y_px: Raw hip-midpoint y pixel position per frame,
            covering the whole video (or the whole session for this
            athlete).
        fps: Frames per second of the source video. Used throughout
            to convert every *_s duration below into an actual frame
            count, so detection behaves the same in real time
            regardless of recording framerate (see the module
            docstring's 2026-07-28 note).
        pixels_per_meter: Pixel-to-metre scale factor, needed to
            convert pixel acceleration into m/s^2 so it can be
            compared against gravity. NOTE: this calibration step
            does not exist yet anywhere in the pipeline -- see the
            /calc/ module docstring. Pass a placeholder value (with
            the understanding that detected windows will be right in
            *shape* but the tolerance_m_s2 gating will be wrong in
            *scale*) until real calibration is added.
        tolerance_m_s2: Passed through to the free-flight window
            search.
        smoothing_window_s: Savitzky-Golay smoothing/derivative window
            length, in SECONDS, passed through to _savgol_derivative
            for both velocity and acceleration (floored at
            MIN_DERIVATIVE_WINDOW_SAMPLES actual samples -- see the
            module docstring's 2026-07-29 note).
        max_gap_s: Short below-tolerance blips of up to this many
            SECONDS are bridged rather than treated as ending a
            flight window, passed through to _find_flight_windows.
        min_flight_duration_s: Minimum flight-window length in
            SECONDS to count as a real jump rather than noise,
            passed through to _find_flight_windows (added 2026-07-30,
            raised from an implicit 0.1s after real-footage tuning --
            see the module docstring's 2026-07-30 note. On this
            project's own real debug-CSV data, raising this to 0.13s
            cut total false-positive-vs-expected deviation across 16
            real athlete/video combinations from roughly 107-184 down
            to 24 -- a real improvement, though not a complete fix;
            some athletes/videos still show too many or too few
            detected jumps and may need further tuning).
        onset_k: How many baseline standard deviations below the
            per-trial quiet-baseline mean velocity counts as
            start-of-movement (see the module docstring's onset
            convention). 5 is the standard force-plate value.
        onset_min_consecutive_s: Minimum sustained run length, in
            SECONDS, below the onset threshold required before
            accepting it, to filter single-frame pose jitter.
        onset_baseline_window_s: Candidate baseline window length, in
            SECONDS, used when adaptively searching for the quietest
            pre-jump segment.
        onset_max_lookback_s: How far before each jump's takeoff (in
            seconds) to search for its quiet baseline window.

    Returns:
        One JumpEvents per detected jump, in chronological order.
        takeoff_frame is the LAST frame in contact with the floor
        (one frame before the detected free-flight window) and
        landing_frame is the FIRST frame back in contact with the
        floor (one frame after it) -- see the module docstring's
        takeoff/landing convention.
    """
    # np.gradient needs at least 2 points; a track this short means
    # there's no usable signal anyway (e.g. an athlete only matched
    # at the /who/ seed frame and never picked up again). Callers
    # such as jumpstart.jumps.segmenter.build_jumps_for_athlete
    # already guard against this with a more informative message and
    # a higher threshold -- this is just a defense-in-depth check so
    # this function is safe to call directly too.
    if len(hip_y_px) < 2:
        return []

    # Work directly in metres (rather than smoothing in pixels and
    # converting afterward) so velocity_m_s and acceleration_m_s2 come
    # straight out of _savgol_derivative in their final physical
    # units -- see that function and the module docstring's
    # 2026-07-29 note for why this replaces smoothing once and then
    # calling np.gradient twice (noise amplification).
    height_m, velocity_m_s, acceleration_m_s2 = compute_kinematics(
        hip_y_px, fps, pixels_per_meter, smoothing_window_s=smoothing_window_s
    )

    flight_windows = _find_flight_windows(
        acceleration_m_s2,
        fps,
        tolerance_m_s2=tolerance_m_s2,
        max_gap_s=max_gap_s,
        min_flight_duration_s=min_flight_duration_s,
    )

    last_index = len(hip_y_px) - 1
    jumps: List[JumpEvents] = []
    previous_landing_frame = 0
    for window in flight_windows:
        if refine_edges:
            # Stage 2: stage 1 found WHERE the jump is; place the
            # edges precisely inside that stretch. See the module
            # docstring's 2026-09-03 note.
            takeoff_frame, landing_frame = _refine_flight_edges(
                velocity_m_s,
                fps,
                window,
                search_margin_s=refine_search_margin_s,
            )
        else:
            takeoff_frame = max(window.start_index - 1, 0)
            landing_frame = min(window.end_index + 1, last_index)

        start_index = None
        if onset_mode == "drop":
            # Verplaatsings-onset (2026-09-10). Valt terug op de
            # snelheidsdrempel als de heup binnen het zoekvenster
            # nooit ver genoeg zakt.
            start_index = _find_start_of_movement_by_drop(
                height_m,
                velocity_m_s,
                fps,
                takeoff_index=takeoff_frame,
                earliest_allowed_index=previous_landing_frame,
                drop_threshold_m=onset_drop_threshold_m,
                max_lookback_s=onset_max_lookback_s,
                rolling_window_s=onset_baseline_window_s,
                smoothing_window_s=smoothing_window_s,
            )
        if start_index is None:
            start_index = _find_start_of_movement(
                velocity_m_s,
                takeoff_index=takeoff_frame,
                earliest_allowed_index=previous_landing_frame,
                fps=fps,
                max_lookback_s=onset_max_lookback_s,
                rolling_window_s=onset_baseline_window_s,
                noise_multiplier=onset_k,
                min_consecutive_s=onset_min_consecutive_s,
            )
        jumps.append(
            JumpEvents(
                start_frame=start_index,
                takeoff_frame=takeoff_frame,
                landing_frame=landing_frame,
            )
        )
        previous_landing_frame = landing_frame
    return jumps
