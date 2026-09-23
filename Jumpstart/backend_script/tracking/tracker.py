"""Athlete tracking for /tracking/.

NEEDS REVIEW -- see the boundary marker in workflow_demo.py. Built in
one pass and only verified with synthetic detections, never against
real Jumpstart footage.

Design decision (resolves the open question in the workflow doc
about whether athlete distinction is part of /pose/ or separate):
MediaPipe detects hip positions per frame but does NOT keep the same
person at the same list position across frames. Instead of a general
multi-object re-identification approach, this module leans on the
fact that /who/ already gives one known, human-verified (participant,
pixel position) pair per athlete at a specific reference frame. That
seed is used to greedily match each athlete to the nearest detection,
frame by frame, forward and backward from the reference frame. This
only works because the project has a small number of well-separated
athletes and a mostly static camera -- it will NOT hold up for
crowded scenes or a moving camera.

IMPORTANT (added 2026-07-29, confirmed on real footage): the
matching distance between an athlete's last known position and a new
detection was a FIXED pixel radius (max_match_distance, default
120px) -- the exact same fixed-pixel-in-a-varying-resolution mistake
already found and fixed for pixels-per-metre (see
jumpstart/who/assignment.py and jumpstart/jumps/segmenter.py). The
same real hip movement/jitter between two consecutive frames covers
MORE pixels at a higher resolution, so a radius that's generous
enough at low resolution can be too tight at high resolution -- even
though the athlete never actually left frame. Confirmed on real
data: the same session's low-resolution video tracked an athlete
through the whole video, while the high-resolution video lost that
same, still-in-frame athlete after the very first frame (matching
never recovered once the first forward/backward step failed, since
each step matches against the athlete's LAST matched position, not
the original seed). track_all_athletes now expresses this tolerance
in METRES and converts it to pixels using this video's own
calibrated pixels-per-metre (see /who/), so the real-world tolerance
stays the same regardless of resolution.

IMPORTANT (added 2026-07-30, confirmed on real footage via the
--debug-csv-dir export in workflow_demo.py): even with the above fix,
matching a candidate purely by a flat distance still let through
physically impossible single-frame jumps. On the project's own real
1080p/240fps footage, 1-8% of ALL frames in a track implied an
instantaneous hip speed above 8 m/s when checked frame-to-frame --
some over 60 m/s, which is not achievable by a real human hip (a
strong countermovement jump's peak vertical velocity is roughly
3.5-4 m/s). The 0.3m flat distance budget from the 2026-07-29 fix,
applied to a single frame step at 240fps (dt ~= 4ms), was equivalent
to allowing up to 0.3 / (1/240) =~ 72 m/s -- generous enough to have
been designed for bridging a multi-frame gap, but far too generous
for a normal single-frame step at high fps, so /tracking/ accepted
these mismatches as real hip movement. Each one then became a
genuine but spurious large jump in the position signal, which
/events/'s Savitzky-Golay derivatives amplify heavily (especially the
second derivative used for acceleration) -- this was a major
remaining source of the near-total jump-detection failure seen on
real footage even after the 2026-07-29 fixes.

Fix: matching tolerance is now expressed as a maximum plausible SPEED
(max_match_speed_m_s, metres/second) instead of a flat distance, and
the actual allowed pixel distance for a given match is that speed
times however much real time has actually elapsed since that
athlete's last successful match (converted to pixels via this
video's calibrated pixels-per-metre) -- so a normal single-frame step
gets a tight, physically sane budget, while recovering from a real
multi-frame gap (e.g. brief occlusion) still gets a proportionally
larger one.

CORRECTION (added 2026-07-30, same day, after a real full-pipeline
re-run): the first version of this fix used max_match_speed_m_s=8.0,
reasoned from real jump kinematics (peak hip velocity ~3.5-4 m/s)
with what seemed like a generous margin. On real footage this made
things WORSE, not better -- jump counts went UP across the board
(e.g. one athlete went from 3 correctly-detected jumps to 9), on
BOTH 1080p and 480p, including videos that were already working
fine. Comparing frame-to-frame implied speed across this project's
own real debug-CSV data at several thresholds showed why: real
MediaPipe hip-position noise is a continuous, heavy-tailed
distribution with NO clean separation between "normal jitter" and
"genuine mismatch" -- the fraction of frames above a given implied
speed shrinks gradually as the threshold rises (roughly halving
every +5-10 m/s) rather than dropping sharply to near-zero at some
natural cutoff. At 8 m/s, this gate was rejecting a meaningful
fraction of perfectly ordinary noisy-but-real single-candidate
detections (not genuine wrong-person mismatches), throwing away real
data and replacing it with linear-interpolation "kinks" at every
rejection -- and each kink is itself a discontinuity that /events/'s
derivatives react to, adding noise instead of removing it. The
default is now 25.0 m/s: real data shows this only rejects the rare
(well under 1% of frames), clearly-extreme mismatches (this project's
footage: instantaneous speeds above 25 m/s are rare outliers, while
8-15 m/s covers a meaningful chunk of ordinary jitter) -- a smaller
threshold trades away more real data than it protects against bad
matches. This is a genuine limitation, not fully solved: distance/
speed-based gating alone cannot cleanly separate ordinary jitter from
a genuine wrong-person mismatch when the noise itself is this
heavy-tailed; a more robust fix would need additional information
MediaPipe already provides but this project doesn't yet use --
PersonDetection.visibility (see jumpstart/pose/detector.py) -- to
prefer/require higher-confidence detections rather than distance
alone.
"""

from __future__ import annotations

import pickle
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from backend_script.pose.detector_yolo import PersonDetection
from backend_script.who.assignment import FrameAssignment

# Standard gravity, m/s^2 -- NOT currently used directly (kept here in
# case a future version of the adaptive-skip trigger below wants a
# true acceleration/free-fall check instead of the simpler speed
# check _TriggerTracker actually uses). Added 2026-08-21 for the
# adaptive frame-skip experiment (frame-skip-experiment-2026-08-21),
# ported into the real pipeline 2026-09-03 (see workflow_demo.py's
# --adaptive-skip-frames and frontend_webapp's frame_skip checkbox).
_GRAVITY_M_S2 = 9.81

# Weight given to a candidate detection's visibility score when
# ordering match candidates -- added 2026-08-14 alongside
# had_eligible_candidate/decode_failed instrumentation, see the
# _greedy_match docstring. Distance still decides ELIGIBILITY (a
# candidate outside max_match_distance is never considered, regardless
# of visibility); this only breaks ties/orders candidates that are
# already within budget, preferring a slightly-further-but-more-
# visible detection over a slightly-closer-but-low-confidence one.
VISIBILITY_WEIGHT = 0.4

# A detect_fn is any callable frame_bgr -> List[PersonDetection] --
# added 2026-08-10 alongside jumpstart/pose/detector_yolo.py so this
# module doesn't hardcode MediaPipe. Previously this module imported
# jumpstart.pose.detector.detect_people directly and always converted
# frame_bgr -> frame_rgb itself before calling it (MediaPipe's own
# requirement); that conversion is now each backend's own
# responsibility (see jumpstart/workflow_demo.py's --pose-backend,
# which builds the actual detect_fn closure for whichever backend was
# chosen -- MediaPipe's closure does the BGR->RGB conversion, YOLO's
# does not need to since Ultralytics accepts BGR arrays directly).


@dataclass
class AthleteTrack:
    """A tracked athlete's hip-midpoint trajectory across a video.

    Attributes:
        participant_id: Which athlete this track belongs to.
        video_id: Which video this track was extracted from.
        fps: Frames per second of the source video (needed to turn
            frame indices into timestamps in /events/ and /calc/).
        frame_width_px: ACTUAL decoded frame width in pixels, read
            directly from the video file via cv2 -- not parsed from
            the filename's resolution label, since the label can
            disagree with what the file actually decodes to. Needed
            (together with frame_height_px) so /jumps/ can scale a
            single manually-calibrated pixels-per-metre reference
            value to this specific video's real pixel density (see
            jumpstart.jumps.segmenter.build_jumps_for_athlete).
        frame_height_px: ACTUAL decoded frame height in pixels --
            see frame_width_px.
        frame_indices: Frame indices with a hip position, ascending,
            with no gaps (missing detections are linearly
            interpolated -- see was_interpolated).
        hip_x: Hip-midpoint x pixel position per frame_indices entry.
        hip_y: Hip-midpoint y pixel position per frame_indices entry.
        was_interpolated: Same length as frame_indices; True where
            MediaPipe missed the detection and the position shown
            here was filled in by linear interpolation instead of
            measured directly.
        visibility: Same length as frame_indices; the matched
            PersonDetection's visibility score (see
            jumpstart/pose/detector.py) for frames that were actually
            matched to a real detection, and NaN elsewhere -- i.e.
            wherever was_interpolated is True (no detection to read a
            score from), AND for seed/carried-over frames where the
            greedy match found no acceptable candidate and the
            previous (or click) position was reused as-is (added
            2026-08-07 to help diagnose whether tracking/event-
            detection problems trace back to MediaPipe losing
            confidence during the fast part of a jump -- see
            workflow_demo.py's --debug-csv-dir help and this module's
            2026-07-30 CORRECTION docstring note, which already
            flagged visibility as unused).
        raw_detection_count: Same length as frame_indices; the TOTAL
            number of PersonDetections MediaPipe returned for that
            video frame, regardless of whether any of them got
            matched to THIS athlete -- i.e. a frame-level count, the
            same value for every athlete's track at a given frame
            index, not an athlete-specific one (added 2026-08-07 to
            tell apart two very different causes of a high
            was_interpolated fraction: raw_detection_count == 0 means
            MediaPipe found nobody at all in that frame -- a /pose/ or
            model-choice problem -- while raw_detection_count > 0 on a
            frame this athlete's track still shows as interpolated
            means MediaPipe DID see someone there, but /tracking/'s
            greedy matching (distance budget, or another athlete's
            claim beating this one to it) rejected every candidate --
            a /tracking/ problem instead. See workflow_demo.py's
            --debug-csv-dir help; this is the direct follow-up to
            investigating why was_interpolated came back so high
            (85-96%) on real footage even outside jump moments).
        decode_failed: Same length as frame_indices; True where
            cv2.VideoCapture.read() itself failed/returned no frame
            for that video frame (added 2026-08-14 to rule decode
            failure in or out as a cause of dropout -- see
            _detect_all_frames). Frame-level, same value for every
            athlete's track at a given frame index, like
            raw_detection_count.
        had_eligible_candidate: Same length as frame_indices;
            per-ATHLETE (unlike raw_detection_count/decode_failed)
            True where /tracking/'s greedy matching found at least one
            detection within this athlete's distance budget that
            frame, whether or not this athlete actually won it against
            another athlete's claim (added 2026-08-14). Lets
            was_interpolated frames be split three ways: no detection
            in the frame at all (raw_detection_count == 0, a /pose/
            problem), a detection existed but not within this
            athlete's budget (had_eligible_candidate == False, a
            distance-tolerance problem), or a detection was in budget
            but lost the greedy match to another athlete
            (had_eligible_candidate == True, a matching-priority
            problem).
    """

    participant_id: str
    video_id: str
    fps: float
    frame_width_px: int
    frame_height_px: int
    frame_indices: np.ndarray
    hip_x: np.ndarray
    hip_y: np.ndarray
    was_interpolated: np.ndarray
    visibility: np.ndarray
    raw_detection_count: np.ndarray
    decode_failed: np.ndarray
    had_eligible_candidate: np.ndarray


@dataclass
class _MutableTrack:
    """Accumulator used while sweeping through a video; not public."""

    participant_id: str
    frame_indices: List[int] = field(default_factory=list)
    hip_x: List[float] = field(default_factory=list)
    hip_y: List[float] = field(default_factory=list)
    # NaN means "no real detection behind this sample" (a carried-over
    # seed/previous position, not an actual matched PersonDetection) --
    # see AthleteTrack.visibility.
    visibility: List[float] = field(default_factory=list)

    def append(
        self, frame_index: int, x: float, y: float, visibility: float = float("nan")
    ) -> None:
        self.frame_indices.append(frame_index)
        self.hip_x.append(x)
        self.hip_y.append(y)
        self.visibility.append(visibility)

    def finalize(
        self,
        video_id: str,
        fps: float,
        frame_width_px: int,
        frame_height_px: int,
        raw_detection_count_by_frame: Optional[List[int]] = None,
        decode_failed_by_frame: Optional[List[bool]] = None,
        had_eligible_candidate_by_frame: Optional[Dict[int, bool]] = None,
        frame_skipped_by_frame: Optional[List[bool]] = None,
    ) -> AthleteTrack:
        """Sort samples by frame and fill any gaps by interpolation.

        Args:
            raw_detection_count_by_frame: index i holds the total
                number of PersonDetections MediaPipe returned for
                absolute video frame i, regardless of athlete --
                see AthleteTrack.raw_detection_count. Optional (and
                filled with -1, meaning "unknown", if omitted) purely
                so this method stays usable in isolation/tests without
                threading the whole video's detection list through.
            decode_failed_by_frame: index i holds whether
                cv2.VideoCapture.read() failed on absolute video frame
                i -- see AthleteTrack.decode_failed. Optional (filled
                with False if omitted), same reasoning as
                raw_detection_count_by_frame.
            had_eligible_candidate_by_frame: absolute frame index ->
                whether THIS athlete had at least one in-budget
                candidate that frame -- see
                AthleteTrack.had_eligible_candidate. Optional (filled
                with False if omitted, or for any frame missing from
                the dict), same reasoning as raw_detection_count_by_frame.
            frame_skipped_by_frame: index i holds whether the adaptive
                frame-skip feature deliberately skipped detection
                on absolute video frame i -- see
                AthleteTrack.frame_skipped. Optional (filled with
                False if omitted, i.e. "adaptive skip wasn't used"),
                same reasoning as raw_detection_count_by_frame. Added
                2026-08-21.
        """
        order = np.argsort(self.frame_indices)
        frames_sorted = np.asarray(self.frame_indices)[order]
        x_sorted = np.asarray(self.hip_x)[order]
        y_sorted = np.asarray(self.hip_y)[order]
        vis_sorted = np.asarray(self.visibility)[order]

        if frames_sorted.size == 0:
            return AthleteTrack(
                participant_id=self.participant_id,
                video_id=video_id,
                fps=fps,
                frame_width_px=frame_width_px,
                frame_height_px=frame_height_px,
                frame_indices=frames_sorted,
                hip_x=x_sorted,
                hip_y=y_sorted,
                was_interpolated=np.array([], dtype=bool),
                visibility=np.array([], dtype=float),
                raw_detection_count=np.array([], dtype=int),
                decode_failed=np.array([], dtype=bool),
                had_eligible_candidate=np.array([], dtype=bool),
            )

        dense_frames = np.arange(frames_sorted[0], frames_sorted[-1] + 1)
        dense_x = np.interp(dense_frames, frames_sorted, x_sorted)
        dense_y = np.interp(dense_frames, frames_sorted, y_sorted)
        was_interpolated = ~np.isin(dense_frames, frames_sorted)

        # Visibility is NOT interpolated like x/y -- a made-up score
        # for a made-up position would be misleading. Every explicitly
        # recorded frame (real match or carried-over seed) keeps its
        # own visibility (NaN for carried-over), and every gap frame
        # filled in by finalize's interpolation above is NaN too, so
        # NaN always means "not an actual measured detection here."
        # Last value wins on duplicate frame indices (shouldn't happen
        # in practice, but matches how x/y already behave via np.interp
        # tolerating duplicates).
        vis_by_frame = dict(zip(frames_sorted.tolist(), vis_sorted.tolist()))
        dense_visibility = np.array(
            [vis_by_frame.get(int(f), float("nan")) for f in dense_frames]
        )

        # Unlike visibility, this IS filled in for every frame
        # (interpolated or not) -- it's a property of the video frame
        # itself (how many people MediaPipe saw there), not of whether
        # this particular athlete was matched, so there is nothing
        # made-up about reporting it on an interpolated frame too.
        if raw_detection_count_by_frame is None:
            dense_raw_detection_count = np.full(dense_frames.shape, -1, dtype=int)
        else:
            dense_raw_detection_count = np.array(
                [raw_detection_count_by_frame[int(f)] for f in dense_frames]
            )

        # Frame-level, same reasoning as raw_detection_count above --
        # a real property of the video frame itself, filled in for
        # every frame regardless of interpolation.
        if decode_failed_by_frame is None:
            dense_decode_failed = np.zeros(dense_frames.shape, dtype=bool)
        else:
            dense_decode_failed = np.array(
                [bool(decode_failed_by_frame[int(f)]) for f in dense_frames]
            )

        # Per-athlete -- unlike the two arrays above, missing frames
        # (this athlete never had a match attempt recorded there, e.g.
        # the empty-track edge case) default to False rather than
        # "unknown", since "no eligibility info recorded" and "no
        # eligible candidate found" both mean this frame's position is
        # not backed by real matching evidence.
        if had_eligible_candidate_by_frame is None:
            dense_had_eligible_candidate = np.zeros(dense_frames.shape, dtype=bool)
        else:
            dense_had_eligible_candidate = np.array(
                [
                    bool(had_eligible_candidate_by_frame.get(int(f), False))
                    for f in dense_frames
                ]
            )

        return AthleteTrack(
            participant_id=self.participant_id,
            video_id=video_id,
            fps=fps,
            frame_width_px=frame_width_px,
            frame_height_px=frame_height_px,
            frame_indices=dense_frames,
            hip_x=dense_x,
            hip_y=dense_y,
            was_interpolated=was_interpolated,
            visibility=dense_visibility,
            raw_detection_count=dense_raw_detection_count,
            decode_failed=dense_decode_failed,
            had_eligible_candidate=dense_had_eligible_candidate,
        )


def _greedy_match(
    known_positions: Dict[str, Tuple[float, float]],
    detections: List[PersonDetection],
    max_match_distance_by_pid: Dict[str, float],
) -> Tuple[Dict[str, Tuple[float, float]], Set[str], Dict[str, float], Set[str]]:
    """Greedily match each known position to its nearest detection.

    Args:
        known_positions: participant_id -> last known (x, y).
        detections: Candidate detections for the current frame.
        max_match_distance_by_pid: Maximum pixel distance to accept a
            match, PER ATHLETE -- added 2026-07-30 (was previously one
            shared value for every athlete) so the caller can scale
            each athlete's own budget by how much real time has
            actually elapsed since their own last successful match
            (see track_all_athletes and its 2026-07-30 docstring
            note), rather than one flat distance regardless of frame
            rate or how long an athlete has been missing.

    Returns:
        A tuple of:
            - Updated positions dict with the same keys as
              known_positions. Athletes with no acceptable match keep
              their previous position (assumed not to have moved).
            - The set of participant_ids that were actually matched
              to a detection this frame (as opposed to carried over
              unchanged).
            - matched_visibility: participant_id -> the matched
                detection's visibility score, for every pid in the
                matched-pids set above (added 2026-08-07 so callers
                can carry PersonDetection.visibility through to
                AthleteTrack.visibility -- previously computed by
                /pose/ but never used anywhere downstream, per this
                module's 2026-07-30 CORRECTION docstring note).
            - eligible_pids: the set of participant_ids that had AT
                LEAST ONE detection within their own max_match_distance
                this frame, regardless of whether they actually won
                that detection against another athlete's claim (added
                2026-08-14 -- see AthleteTrack.had_eligible_candidate).
                A strict superset of matched_pids: every matched pid
                was necessarily eligible, but an eligible pid can lose
                its only in-budget candidate to another athlete with a
                better (closer/more visible) score, in which case it's
                in eligible_pids but not matched_pids.

    Candidates within budget are scored by distance weighted by the
    candidate detection's visibility (VISIBILITY_WEIGHT, added
    2026-08-14) rather than raw distance alone, so a slightly further
    but higher-confidence detection is preferred over a slightly
    closer but low-confidence one when two athletes' budgets overlap.
    Eligibility itself (whether a candidate is considered AT ALL) is
    still decided purely by raw distance <= max_match_distance --
    visibility only affects scoring among already-eligible candidates,
    it never makes an out-of-budget detection eligible.

    UPDATED 2026-08-15 (oplossing C1 uit
    Jumpstart_oplossingsrichtingen.docx): the actual assignment is now
    solved with scipy.optimize.linear_sum_assignment (the Hungarian
    algorithm) instead of the previous greedy, sort-and-claim loop.
    The greedy approach assigned each candidate pair to the locally
    best-scoring match one at a time, which can miss a globally better
    overall assignment when two athletes are close enough to compete
    for overlapping candidates (relevant at 0.5m athlete spacing) --
    e.g. greedily giving athlete A its single best (lowest-score)
    candidate first can force athlete B into a far worse (or no) match,
    when swapping which athlete gets which candidate would have let
    both match acceptably. The Hungarian algorithm finds the
    assignment that minimizes the TOTAL score summed across all
    athletes at once, so it cannot be fooled by this kind of locally-
    greedy ordering. This only changes WHO wins a contested detection
    when multiple athletes have overlapping eligible candidates -- it
    does not change eligibility itself (see the module docstring's
    2026-08-15 note in track_all_athletes' had_eligible_candidate
    discussion for why that distinction matters).
    """
    pids = list(known_positions.keys())
    eligible_pids: Set[str] = set()

    # Cost matrix: rows = athletes, columns = detections. Anything
    # outside this athlete's own max_match_distance gets a very high
    # (but finite) cost so linear_sum_assignment never prefers it over
    # a genuinely eligible pairing, and is filtered back out below --
    # linear_sum_assignment always assigns SOMETHING to every row/
    # column it can, even when nothing is actually acceptable, so the
    # infeasible-cost sentinel plus the post-hoc filter together
    # reproduce "no match" for a pid with zero eligible candidates.
    INFEASIBLE_COST = 1e12
    n_pids = len(pids)
    n_dets = len(detections)
    cost_matrix = np.full((n_pids, n_dets), INFEASIBLE_COST, dtype=float)
    for row, pid in enumerate(pids):
        px, py = known_positions[pid]
        max_match_distance = max_match_distance_by_pid.get(pid, 0.0)
        for col, det in enumerate(detections):
            distance = ((det.x - px) ** 2 + (det.y - py) ** 2) ** 0.5
            if distance <= max_match_distance:
                eligible_pids.add(pid)
                cost_matrix[row, col] = distance * (1.0 - VISIBILITY_WEIGHT * det.visibility)

    updated = dict(known_positions)
    matched_pids: Set[str] = set()
    matched_visibility: Dict[str, float] = {}
    if n_pids > 0 and n_dets > 0:
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] >= INFEASIBLE_COST:
                continue  # this pid had no eligible candidate at all
            pid = pids[row]
            det = detections[col]
            updated[pid] = (det.x, det.y)
            matched_pids.add(pid)
            matched_visibility[pid] = det.visibility
    return updated, matched_pids, matched_visibility, eligible_pids


class _ProgressReporter:
    """Prints progress at most every interval_s seconds -- time-based,
    not frame-count-based, so a slow phase still reports in promptly
    instead of going silent for minutes (see track_all_athletes'
    docstring for why frame-count-based reporting isn't enough here).
    Always prints on the very first frame so there's instant
    confirmation the pass actually started.

    UPDATE 2026-09-16 (webapp live voortgangsindicator): optionele
    on_update-callback, aangeroepen op EXACT hetzelfde throttle-moment
    als de bestaande print (dus ook hoogstens elke interval_s
    seconden) met een dict {frames_done, total_steps, percent,
    elapsed_s, eta_s, label} -- zodat een aanroeper (frontend_webapp/
    pipeline.py) dit percentage ook buiten dit stdout-printen om kan
    opvangen, bv. om te laten zien in de webapp. None (default) laat
    het bestaande gedrag (alleen printen) volledig ongewijzigd.
    """

    def __init__(
        self,
        total_steps: int,
        label: str,
        interval_s: float = 5.0,
        on_update: Optional[Callable[[str, float, float, Optional[float]], None]] = None,
    ) -> None:
        self.total_steps = total_steps
        self.label = label
        self.interval_s = interval_s
        self.start_time = time.time()
        self.last_print_time = self.start_time
        # Toegevoegd 2026-09-22 (opschonen-to-do E): optionele callback,
        # zelfde ingebouwde rate-limit als de print hieronder (dus geen
        # aparte throttle nodig) -- laat de webapp dit percentage in de
        # UI tonen i.p.v. alleen naar de server-cmd te printen. None
        # (CLI-gebruik) is exact het oude gedrag.
        self.on_update = on_update

    def update(self, step: int) -> None:
        """Call once per processed frame with its 0-based step index."""
        if self.total_steps <= 0:
            return
        now = time.time()
        is_first = step == 0
        is_last = step == self.total_steps - 1
        if not (is_first or is_last or (now - self.last_print_time) >= self.interval_s):
            return
        self.last_print_time = now

        frames_done = step + 1
        elapsed_s = now - self.start_time
        percent = 100 * frames_done / self.total_steps
        rate_fps = frames_done / elapsed_s if elapsed_s > 0 else 0.0
        remaining = self.total_steps - frames_done
        eta_s = remaining / rate_fps if rate_fps > 0 else None
        eta_text = f"{eta_s:.0f}s" if eta_s is not None else "unknown"
        print(
            f"    {self.label}: {frames_done}/{self.total_steps} frames "
            f"({percent:.0f}%, {elapsed_s:.0f}s elapsed, "
            f"~{rate_fps:.2f} frames/s, ETA {eta_text})"
        )
        if self.on_update is not None:
            self.on_update({
                "label": self.label,
                "frames_done": frames_done,
                "total_steps": self.total_steps,
                "percent": percent,
                "elapsed_s": elapsed_s,
                "eta_s": eta_s,
            })


class _TriggerTracker:
    """Cheap, athlete-agnostic proxy tracker used ONLY to decide when
    _detect_all_frames' adaptive frame-skip mode should switch from
    sparse to dense detection (added 2026-08-21, ported into the real
    pipeline 2026-09-03). This is NOT athlete identification (see
    _greedy_match/track_all_athletes for that, which needs the /who/
    seed and runs afterwards, purely in memory, over whatever this
    function recorded) -- it just asks, per MEASURED frame, "did
    anything in frame move implausibly fast since the last measured
    frame", as a cheap stand-in for "a jump probably just started",
    per Mathijs' own framing: skip frames by default until a speed
    threshold is exceeded, then rewind and go dense for a while.

    Deliberately speed-based (distance between two MEASURED frames,
    divided by the real elapsed time between them) rather than the
    acceleration/free-fall check /events/detector.py uses for actual
    jump-phase detection on a matched, smoothed AthleteTrack -- at
    this stage there is no athlete identity or smoothing yet, only
    raw per-frame detections, so a 3-point acceleration estimate
    would be noisy and require carrying more history for little
    benefit. A single-pair speed check is what was described
    ("totdat de snelheidsdrempel overschreden wordt") and is enough
    to catch the start of a jump's rapid upward motion.

    UPDATED 2026-08-24 (real-footage test, S01 C02): on the first real
    test, this fired far more often than there were real jumps (10
    dense bursts vs. 2 real jumps on one athlete's 240fps/1080p
    track) -- confirmed via the debug CSV's new frame_skipped column
    plus raw_detection_count, which showed 4-5 people in frame at
    once (other athletes/coaches) while the TRACKED athlete's own
    hip_y barely moved. Because this tracker is deliberately
    athlete-agnostic (see above -- no identity exists yet at this
    stage), it was comparing ALL detected people frame-to-frame, so
    ANY of those other people moving (walking, adjusting, their own
    jump) tripped the trigger just as easily as the tracked athlete's
    real jump. _detect_all_frames now restricts which detections this
    tracker ever sees to those near a known athlete's /who/ seed
    position (see its seed_positions_px/adaptive_trigger_radius_m
    arguments) specifically to fix this -- this class itself is
    unchanged, it just receives an already-filtered detections list.
    """

    def __init__(self, pixels_per_meter: float, fps: float, trigger_speed_m_s: float) -> None:
        self._pixels_per_meter = pixels_per_meter
        self._fps = fps
        self._trigger_speed_m_s = trigger_speed_m_s
        self._prev_frame_index: Optional[int] = None
        self._prev_positions: List[Tuple[float, float]] = []

    def update(self, frame_index: int, detections: List[PersonDetection]) -> bool:
        """Feed in one measured frame's detections.

        Returns True if the fastest implied real-world speed between
        any of this frame's detections and the nearest detection from
        the previous measured frame exceeds trigger_speed_m_s.
        """
        positions = [(det.x, det.y) for det in detections]
        triggered = False
        if self._prev_positions and positions and self._prev_frame_index is not None:
            elapsed_frames = frame_index - self._prev_frame_index
            elapsed_s = elapsed_frames / self._fps if self._fps > 0 else 0.0
            if elapsed_s > 0:
                for px, py in positions:
                    nearest_distance_px = min(
                        ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5
                        for qx, qy in self._prev_positions
                    )
                    speed_m_s = (nearest_distance_px / self._pixels_per_meter) / elapsed_s
                    if speed_m_s > self._trigger_speed_m_s:
                        triggered = True
                        break
        self._prev_frame_index = frame_index
        self._prev_positions = positions
        return triggered


def _detections_near_seeds(
    detections: List[PersonDetection],
    seed_positions_px: List[Tuple[float, float]],
    radius_px: float,
) -> List[PersonDetection]:
    """Keep only the detections within radius_px of ANY seed position.

    ADDED 2026-08-24, alongside the _TriggerTracker "UPDATED
    2026-08-24" docstring note above -- used to restrict what
    _detect_all_frames' adaptive-skip trigger ever sees to detections
    plausibly belonging to one of THIS video's assigned athletes,
    instead of every person YOLO/MediaPipe found in frame. Does NOT
    touch all_detections (the real, full per-frame detection list
    returned to callers and used for actual matching later) -- only
    the copy handed to _TriggerTracker.update().

    Args:
        detections: One frame's full detection list.
        seed_positions_px: Every assigned athlete's /who/ seed click
            position (pixels) for this video -- a STATIC set, taken
            once from the reference frame, not updated as athletes
            move around later in the video. Known limitation: on a
            long multi-attempt video where an athlete walks well away
            from their original click position (e.g. queuing up
            between attempts), detections near their later, actual
            position will fall outside every seed's radius and won't
            be considered for the trigger there -- the trigger may
            then fire late (or only on the fallback sparse-sampling
            cadence) for that stretch. This is a deliberate simplicity
            tradeoff over building a second, cheap position-tracking
            pass just for the trigger -- if it turns out to matter in
            practice, seed_positions_px could instead be refreshed
            periodically from the most recent measured detections.
        radius_px: Maximum distance (pixels) from a seed position for
            a detection to be considered "probably one of our
            athletes" -- see adaptive_trigger_radius_m.

    Returns:
        The subset of detections within radius_px of at least one
        seed position. If seed_positions_px is empty, returns
        detections unchanged (no filtering possible without any
        seeds).
    """
    if not seed_positions_px:
        return detections
    kept = []
    for det in detections:
        for seed_x, seed_y in seed_positions_px:
            distance = ((det.x - seed_x) ** 2 + (det.y - seed_y) ** 2) ** 0.5
            if distance <= radius_px:
                kept.append(det)
                break
    return kept


def _detect_all_frames(
    capture: cv2.VideoCapture,
    detect_fn,
    total_frames: int,
    pixels_per_meter: Optional[float] = None,
    fps: float = 30.0,
    adaptive_skip_frames: Optional[int] = None,
    adaptive_rewind_samples: int = 5,
    adaptive_dense_duration_s: float = 2.5,
    adaptive_trigger_speed_m_s: float = 2.0,
    seed_positions_px: Optional[List[Tuple[float, float]]] = None,
    adaptive_trigger_radius_m: float = 1.5,
    progress_callback: Optional[Callable[[str, float, float, Optional[float]], None]] = None,
) -> Tuple[List[List[PersonDetection]], List[bool], List[bool]]:
    """Run pose detection over every frame via one sequential pass.

    Deliberately reads frames with successive capture.read() calls
    and never seeks (no cv2.CAP_PROP_POS_FRAMES .set() calls) --
    confirmed on real 240fps 1080p footage that per-frame seeking
    runs at roughly 1.3 frames/second (757s measured for 1019
    frames), which would make an 8+ hour pipeline per video. Video
    codecs are built for cheap sequential decoding; seeking forces a
    re-seek-and-decode-from-the-nearest-keyframe on every call, even
    when moving to the very next frame. This function pays the
    decode+inference cost exactly once per frame, in one pass, and
    caches the (lightweight) detections so the forward/backward
    matching below never has to touch the video file again.

    ADDED 2026-08-21 (adaptive frame-skip feature, ported from the
    frame-skip-experiment-2026-08-21 dated copy into this real module
    2026-09-03): when adaptive_skip_frames is set, this function still
    DECODES every frame (video decode is sequential/cheap and can't be
    skipped -- see above) but only runs the (much more expensive)
    detect_fn on every (adaptive_skip_frames + 1)-th frame by default,
    using _TriggerTracker as a cheap trip-wire on those measured
    frames. The moment the trip-wire fires, it rewinds -- via an
    in-memory ring buffer of recently-decoded-but-not-yet-inferred raw
    frames, NOT by re-seeking the video -- to adaptive_rewind_samples
    measured frames ago, runs detect_fn on every frame from there
    forward, and keeps running every frame (no skipping) for
    adaptive_dense_duration_s seconds before returning to sparse
    skipping. Frames that stay skipped get an empty detection list,
    exactly like a genuine decode failure or a frame with nobody in
    it -- the existing forward/backward matching in track_all_athletes
    already interpolates over those without any change, since it only
    ever sees this function's output, never the skip decision itself.
    frame_skipped_by_frame (new return value, see below) is what lets
    the debug CSV tell a deliberate skip apart from a genuine miss.

    When adaptive_skip_frames is None (the default), behaviour is
    IDENTICAL to before this feature existed -- every frame is
    decoded and measured, frame_skipped_by_frame is all False, and
    none of the adaptive_* arguments or pixels_per_meter/fps are used
    for anything.

    UPDATED 2026-08-24 (real-footage test, S01 C02 -- see
    _TriggerTracker's "UPDATED 2026-08-24" docstring note): the
    trigger was firing far more often than there were real jumps,
    traced to it being athlete-agnostic -- ANY of several people in
    frame (other athletes, coaches) moving would fire it, not just
    the tracked athlete. Fixed by restricting what the trigger ever
    SEES to detections near a known athlete's /who/ seed position
    (seed_positions_px, filtered via _detections_near_seeds) --
    _TriggerTracker itself is unchanged, it just no longer receives
    detections for people who were never assigned as an athlete in
    this video. adaptive_dense_duration_s's default also dropped from
    8.0 to 2.5 the same day: a real detected jump's whole
    countermovement+flight+landing window measured only ~1.4s on the
    S01 C02 test athlete, so 8s was several times more dense-mode
    time than a real jump needs -- 2.5s keeps a margin for /events/'s
    onset detection (which looks slightly before the flight window)
    without paying for as much unnecessary dense time per trigger,
    real OR still-false. NOTE: the webapp (frontend_webapp) passes
    its own default of 4.5s for this parameter, per the 2026-08-26
    manual-analysis comparison run -- see pipeline.run_analysis.

    Args:
        capture: An opened cv2.VideoCapture, positioned at frame 0
            (this function seeks there once, up front, then never
            again).
        detect_fn: A callable frame_bgr -> List[PersonDetection] --
            added 2026-08-10, replacing a hardcoded MediaPipe
            landmarker + RGB conversion, so this module works with any
            pose backend (see jumpstart/pose/detector_yolo.py and
            jumpstart/workflow_demo.py's --pose-backend, which builds
            this closure). Receives the raw BGR frame exactly as
            cv2.VideoCapture.read() returns it -- any needed colour
            conversion (e.g. MediaPipe's RGB requirement) is the
            closure's own responsibility, not this function's.
        total_frames: Number of frames to read.
        pixels_per_meter: This video's calibrated pixels-per-metre
            (see /who/), needed to convert _TriggerTracker's pixel
            distances into real m/s. If None, adaptive skipping is
            disabled for this video (falls back to dense, exactly
            like adaptive_skip_frames=None) and a warning is printed,
            since the speed trigger can't be trusted without a real
            calibration -- see track_all_athletes' own pixels_per_meter
            docstring note for why a fixed-pixel fallback isn't used
            here the way it is for matching distance.
        fps: This video's frames per second, needed to convert elapsed
            frame counts into real seconds for both the speed trigger
            and adaptive_dense_duration_s.
        adaptive_skip_frames: Number of frames to skip between
            measured frames while in sparse mode (e.g. 3 means 1 in
            every 4 frames is measured). None (the default) disables
            adaptive skipping entirely -- unchanged original
            behaviour.
        adaptive_rewind_samples: How many previously-measured frames
            back to rewind to once the trigger fires.
        adaptive_dense_duration_s: How many seconds (converted to a
            frame count via fps) to keep measuring every single frame
            after a trigger, before returning to sparse skipping.
        adaptive_trigger_speed_m_s: The implied real-world speed
            (m/s) between two measured frames' nearest detections that
            counts as "something is probably jumping" -- see
            _TriggerTracker. Deliberately well below a real
            countermovement jump's peak hip velocity (documented
            elsewhere in this codebase as roughly 3.5-4 m/s) so the
            trigger fires early into the jump rather than only near
            its peak, but this default is NOT yet validated as
            optimal against real footage -- see workflow_demo.py's
            --adaptive-trigger-speed-m-s help.
        seed_positions_px: ADDED 2026-08-24 -- every athlete assigned
            to this video's /who/ seed click position (pixels), used
            to filter what the trigger sees -- see
            _detections_near_seeds and adaptive_trigger_radius_m.
            None or empty disables the filter (trigger sees every
            detection in frame again, the pre-2026-08-24 behaviour) --
            track_all_athletes always passes this when adaptive
            skipping is on, so in practice this only stays None/empty
            via a direct/test call to this function.
        adaptive_trigger_radius_m: ADDED 2026-08-24 -- real-world
            radius (metres, converted to pixels via pixels_per_meter)
            around each seed_positions_px entry within which a
            detection counts as "probably one of our athletes" for
            trigger purposes. 1.5m is a rough starting point (roomier
            than a standing person's own footprint, to allow for
            normal movement during a jump attempt) -- NOT yet
            validated as optimal against real footage; see
            _detections_near_seeds' docstring for the known tradeoff
            (a static seed position, not updated as an athlete moves
            around later in a long video).

    Returns:
        A tuple of:
            - A list of length total_frames; index i holds the
              PersonDetection list for frame i (empty list for any
              frame that failed to decode OR was deliberately skipped,
              so callers don't have to special-case gaps).
            - decode_failed_by_frame: a list of length total_frames;
                index i is True if cv2.VideoCapture.read() itself
                failed/returned no frame for frame i (added 2026-08-14
                to test decode failure in or out as a cause of the
                high real-footage interpolation rate -- see
                AthleteTrack.decode_failed).
            - frame_skipped_by_frame: a list of length total_frames;
                index i is True if detection was deliberately NOT run
                on frame i by the adaptive skip logic above (added
                2026-08-21 -- see AthleteTrack.frame_skipped). Always
                all-False when adaptive_skip_frames is None.
    """
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    all_detections: List[List[PersonDetection]] = [[] for _ in range(total_frames)]
    decode_failed_by_frame: List[bool] = [False] * total_frames
    frame_skipped_by_frame: List[bool] = [False] * total_frames
    reporter = _ProgressReporter(total_frames, "Pose detection", on_update=progress_callback)

    use_adaptive_skip = adaptive_skip_frames is not None and adaptive_skip_frames > 0
    if use_adaptive_skip and pixels_per_meter is None:
        print(
            "    Adaptive frame-skip: nog geen pixels-per-metre "
            "kalibratie voor deze resolutie -- kan de snelheidstrigger "
            "niet betrouwbaar in echte m/s uitrekenen, dus deze video "
            "wordt gewoon elk frame gemeten (geen frames overgeslagen)."
        )
        use_adaptive_skip = False

    if not use_adaptive_skip:
        # Unchanged original behaviour.
        for frame_index in range(total_frames):
            success, frame_bgr = capture.read()
            if not success or frame_bgr is None:
                decode_failed_by_frame[frame_index] = True
                reporter.update(frame_index)
                continue
            all_detections[frame_index] = detect_fn(frame_bgr)
            reporter.update(frame_index)
        return all_detections, decode_failed_by_frame, frame_skipped_by_frame

    step = adaptive_skip_frames + 1  # e.g. skip=3 -> 1 measured frame in every 4
    dense_frames_target = max(1, round(adaptive_dense_duration_s * fps))
    trigger_tracker = _TriggerTracker(pixels_per_meter, fps, adaptive_trigger_speed_m_s)
    # Pixel radius the trigger's ROI filter uses -- see
    # _detections_near_seeds and adaptive_trigger_radius_m above.
    trigger_radius_px = adaptive_trigger_radius_m * pixels_per_meter
    seed_positions_px = seed_positions_px or []

    # Ring buffer of recently-decoded-but-not-yet-inferred raw frames --
    # lets a trigger "rewind" onto frames that were already decoded
    # and skipped, without re-reading the video (which the seeking-
    # cost note above rules out). Sized generously for the worst case:
    # the rewind target is adaptive_rewind_samples MEASURED frames
    # back, each up to `step` raw frames apart.
    ring_buffer_size = adaptive_rewind_samples * step + step + 1
    raw_frame_buffer: "deque[Tuple[int, object]]" = deque(maxlen=ring_buffer_size)

    # Frame indices of the last (adaptive_rewind_samples + 1) MEASURED
    # frames (the +1 is so index [0] is exactly adaptive_rewind_samples
    # measured-samples before the current one, not adaptive_rewind_samples - 1).
    sparse_history: "deque[int]" = deque(maxlen=adaptive_rewind_samples + 1)

    dense_frames_remaining = 0
    for frame_index in range(total_frames):
        success, frame_bgr = capture.read()
        if not success or frame_bgr is None:
            decode_failed_by_frame[frame_index] = True
            reporter.update(frame_index)
            continue

        raw_frame_buffer.append((frame_index, frame_bgr))

        is_dense_burst = dense_frames_remaining > 0
        is_scheduled_sample = (frame_index % step) == 0
        should_measure = is_dense_burst or is_scheduled_sample

        if not should_measure:
            frame_skipped_by_frame[frame_index] = True
            reporter.update(frame_index)
            continue

        detections = detect_fn(frame_bgr)
        all_detections[frame_index] = detections
        sparse_history.append(frame_index)
        # Only detections near a known athlete's seed position ever
        # reach the trigger -- see _detections_near_seeds' docstring
        # and the 2026-08-24 update note above (fixes the trigger
        # firing on OTHER people's movement in a multi-person frame).
        trigger_detections = _detections_near_seeds(detections, seed_positions_px, trigger_radius_px)
        triggered = trigger_tracker.update(frame_index, trigger_detections)

        if is_dense_burst:
            dense_frames_remaining -= 1
        elif triggered:
            # Rewind onto whatever raw frames are still buffered from
            # adaptive_rewind_samples measured frames ago, running
            # detect_fn on the ones that were skipped -- then keep
            # measuring every frame for adaptive_dense_duration_s.
            rewind_target = sparse_history[0]
            buffered_by_index = dict(raw_frame_buffer)
            for backfill_index in range(rewind_target, frame_index):
                if frame_skipped_by_frame[backfill_index] and backfill_index in buffered_by_index:
                    all_detections[backfill_index] = detect_fn(buffered_by_index[backfill_index])
                    frame_skipped_by_frame[backfill_index] = False
                # A backfill_index missing from buffered_by_index fell
                # out of the ring buffer already (older than its size)
                # -- shouldn't normally happen given how it's sized
                # above, and is simply left skipped if it ever does.
            dense_frames_remaining = dense_frames_target

        reporter.update(frame_index)

    return all_detections, decode_failed_by_frame, frame_skipped_by_frame


def track_all_athletes(
    video_path: Path,
    detect_fn,
    assignment: FrameAssignment,
    max_match_speed_m_s: float = 25.0,
    pixels_per_meter: Optional[float] = None,
    fallback_max_match_distance_px: float = 120.0,
    detections_cache_path: Optional[Path] = None,
    max_frames_since_match: Optional[int] = None,
    adaptive_skip_frames: Optional[int] = None,
    adaptive_rewind_samples: int = 5,
    adaptive_dense_duration_s: float = 2.5,
    adaptive_trigger_speed_m_s: float = 2.0,
    adaptive_trigger_radius_m: float = 1.5,
    progress_callback: Optional[Callable[[str, float, float, Optional[float]], None]] = None,
) -> Dict[str, AthleteTrack]:
    """Track every assigned athlete's hip midpoint across a video.

    Runs pose detection over the whole video in one sequential pass
    (see _detect_all_frames for why -- per-frame seeking measured at
    ~1.3 frames/second on real footage, sequential reading is much
    faster). Then, starting at the /who/ reference frame, each
    athlete's user-clicked position is matched to the nearest actual
    detection there (falling back to the raw click if none is close
    enough), and the same greedy nearest-detection matching sweeps
    forward and backward through the cached per-frame detections --
    pure in-memory work at this point, no further video I/O.

    Args:
        video_path: Path to the video file (same one /who/ used).
        detect_fn: A callable frame_bgr -> List[PersonDetection] --
            renamed from a hardcoded MediaPipe `landmarker` 2026-08-10
            so this function works with any pose backend (added
            alongside jumpstart/pose/detector_yolo.py as a second
            backend option -- see jumpstart/workflow_demo.py's
            --pose-backend, which builds the actual closure for
            whichever backend was chosen). See _detect_all_frames for
            the exact calling convention.
        assignment: The FrameAssignment produced by /who/ for this
            video.
        max_match_speed_m_s: Maximum plausible REAL speed (metres per
            SECOND) an athlete's hip may move and still count as the
            same person, converted to a pixel-distance budget for
            each match via pixels_per_meter and however much real
            time has actually elapsed since that athlete's last
            successful match (added 2026-07-30, replacing a flat
            max_match_distance_m -- see the module docstring's
            2026-07-30 note for why a flat distance was still too
            generous for a normal single-frame step at high fps, and
            let through physically impossible matches on real
            1080p/240fps footage). 25.0 m/s (raised from an initial
            8.0 m/s -- see the module docstring's same-day CORRECTION
            note) is deliberately much higher than a real
            countermovement jump's peak hip velocity (~3.5-4 m/s):
            real MediaPipe position noise turned out to be a
            continuous, heavy-tailed distribution with no clean
            separation between ordinary jitter and a genuine
            mismatch, so a tight threshold rejected real (if noisy)
            data far more often than it caught real mismatches, which
            measurably made real-footage results worse, not better.
        pixels_per_meter: This video's real pixels-per-metre (see
            /who/'s calibration). If None (this resolution hasn't
            been calibrated yet), falls back to
            fallback_max_match_distance_px (scaled by frames elapsed
            since the last match) as a fixed pixel radius per frame
            -- same behaviour as before this fix, so tracking still
            runs (just with the old resolution-inconsistent
            tolerance) rather than failing outright.
        fallback_max_match_distance_px: Used only when
            pixels_per_meter is None -- see above.
        detections_cache_path: Added 2026-08-14 to speed up iterating
            on /tracking/'s matching logic alone, without re-running
            the (much slower) /pose/ detection pass every time: if
            set and the file already exists, this function loads
            (all_detections, decode_failed_by_frame) from it via
            pickle instead of calling _detect_all_frames at all. If
            set and the file does NOT exist yet, detection runs
            normally and the result is pickled there afterwards for
            next time. None (the default) disables caching entirely
            -- detection always runs fresh, matching the old
            behaviour.
        max_frames_since_match: Added 2026-08-15 (oplossing C5 uit
            Jumpstart_oplossingsrichtingen.docx). The per-athlete
            match budget in _allowed_distances grows UNBOUNDED with
            however many frames have elapsed since that athlete's
            last successful match (budget = speed * elapsed_time *
            pixels_per_meter) -- during a long missed streak this
            budget can grow large enough that almost any detection
            anywhere in frame falls "within budget", making the match
            effectively random rather than a genuine nearest-neighbour
            decision. If set, once frames_since_match for an athlete
            exceeds this value, that athlete's budget is clamped back
            down to a single frame's worth (no further growth) instead
            of continuing to expand -- still allows recovery once a
            real detection appears nearby, but stops the runaway
            growth. None (the default) preserves the old unbounded
            behaviour.
        adaptive_skip_frames: ADDED 2026-08-21, adaptive frame-skip
            feature (ported from the frame-skip-experiment-2026-08-21
            dated copy into this real module 2026-09-03, after real-
            footage validation on 2026-08-26 -- see TODO.md item 3's
            2026-08-26 update). Passed straight through to
            _detect_all_frames: None (the default) disables it
            entirely, giving byte-for-byte the old dense-detection
            behaviour. See _detect_all_frames's own docstring for the
            full sparse/trigger/rewind/dense mechanism.
        adaptive_rewind_samples: See _detect_all_frames.
        adaptive_dense_duration_s: See _detect_all_frames. Default
            changed 2026-08-24 from 8.0 to 2.5 -- see
            _detect_all_frames' "UPDATED 2026-08-24" docstring note.
            The webapp itself calls this with 4.5s -- see
            frontend_webapp/pipeline.run_analysis.
        adaptive_trigger_speed_m_s: See _detect_all_frames.
        adaptive_trigger_radius_m: See _detect_all_frames's
            _detections_near_seeds-based ROI filter (added
            2026-08-24). This function always passes
            assignment.click_positions as the seed positions --
            there's no separate parameter for that here, since
            track_all_athletes already has the assignment.

    Returns:
        A dict of participant_id -> AthleteTrack, one per athlete in
        the assignment.
    """
    capture = cv2.VideoCapture(str(video_path))
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        # Actual decoded frame size -- deliberately NOT the filename's
        # resolution label (e.g. "480p"), which can disagree with
        # what the file actually decodes to. /jumps/ uses this to
        # scale a single manually-calibrated pixels-per-metre value
        # to each video's real pixel density.
        frame_width_px = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height_px = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if pixels_per_meter is None:
            print(
                "    No pixels-per-metre calibration yet for "
                f"{frame_width_px}x{frame_height_px} -- matching "
                f"tolerance falls back to a fixed "
                f"{fallback_max_match_distance_px:.0f}px radius per "
                "frame gap (may behave inconsistently across "
                "resolutions, see the module docstring's 2026-07-29 "
                "note)."
            )

        cache_is_usable = False
        if detections_cache_path is not None and detections_cache_path.exists():
            print(f"    Cache gevonden, detecties laden uit {detections_cache_path} ...")
            # Vangnet (2026-09-17): een cache-bestand dat om welke
            # reden dan ook niet meer in te laden is, telt gewoon als
            # cache-miss -- detectie draait dan opnieuw en het bestand
            # wordt daarna overschreven met het nieuwe formaat. Vangt
            # twee bekende gevallen af die hiervoor de hele analyse
            # lieten crashen:
            #   - ModuleNotFoundError: caches van voor 2026-09-17
            #     bevatten gepickelde PersonDetection-objecten met het
            #     oude modulepad jumpstart.pose.detector (dat bestand
            #     is weg sinds de MediaPipe-backend verwijderd is; de
            #     dataclass staat nu in jumpstart/pose/detector_yolo.py).
            #   - ValueError bij het uitpakken: de gecachete tuple
            #     groeide 2026-08-21 naar drie elementen
            #     (frame_skipped_by_frame), dus een cache van voor
            #     2026-09-03 past niet in deze drie namen.
            try:
                with open(detections_cache_path, "rb") as cache_file:
                    all_detections, decode_failed_by_frame, frame_skipped_by_frame = pickle.load(cache_file)
            except Exception as cache_error:
                print(
                    f"    WAARSCHUWING: cache {detections_cache_path} kon "
                    f"niet ingeladen worden ({type(cache_error).__name__}: "
                    f"{cache_error}) -- cache wordt genegeerd, detectie "
                    "draait opnieuw en de cache wordt daarna vervangen."
                )
                all_detections = None
            # Toegevoegd 2026-09-08 (vangnet): een cache-bestand dat
            # niet bij DEZE video hoort -- bv. door een elders
            # gefixte cache-naamgevingsbug (frontend_webapp/
            # pipeline.py keyde tot 2026-09-08 op het per-run
            # volgnummer video_id i.p.v. de video-bestandsnaam, zie
            # _detections_cache_path_for's docstring daar) of een
            # handmatig hernoemd bestand -- gaf hiervoor verderop een
            # IndexError bij te weinig gecachete frames, of erger:
            # STILZWIJGEND de detecties van een andere video, als de
            # framecounts toevallig dicht genoeg bij elkaar lagen. Een
            # geldige cache heeft precies één entry per frame van DEZE
            # video.
            if all_detections is None:
                # Inladen is hierboven al mislukt en gemeld -- niet
                # nog een tweede waarschuwing erachteraan.
                pass
            elif len(all_detections) == total_frames:
                cache_is_usable = True
            else:
                print(
                    f"    WAARSCHUWING: cache {detections_cache_path} hoort "
                    f"niet bij deze video ({len(all_detections)} gecachete "
                    f"frames, maar deze video heeft er {total_frames}) -- "
                    "cache wordt genegeerd, detectie draait opnieuw."
                )

        if not cache_is_usable:
            print(
                f"    Video has {total_frames} frames at {fps:.1f}fps -- "
                "running pose detection once over the whole video "
                "(sequential read, no seeking)..."
            )
            all_detections, decode_failed_by_frame, frame_skipped_by_frame = _detect_all_frames(
                capture,
                detect_fn,
                total_frames,
                pixels_per_meter=pixels_per_meter,
                fps=fps,
                adaptive_skip_frames=adaptive_skip_frames,
                adaptive_rewind_samples=adaptive_rewind_samples,
                adaptive_dense_duration_s=adaptive_dense_duration_s,
                adaptive_trigger_speed_m_s=adaptive_trigger_speed_m_s,
                seed_positions_px=list(assignment.click_positions),
                adaptive_trigger_radius_m=adaptive_trigger_radius_m,
                progress_callback=progress_callback,
            )
            if detections_cache_path is not None:
                detections_cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(detections_cache_path, "wb") as cache_file:
                    pickle.dump((all_detections, decode_failed_by_frame, frame_skipped_by_frame), cache_file)
                print(f"    Detecties gecached naar {detections_cache_path}")
    finally:
        capture.release()

    # Frame-level (not athlete-specific) count of how many people
    # MediaPipe returned per frame -- added 2026-08-07, see
    # AthleteTrack.raw_detection_count -- computed once here and
    # handed to every athlete's finalize() call below so a low match
    # rate can be diagnosed as "MediaPipe saw nobody" (count 0) versus
    # "MediaPipe saw someone but /tracking/'s matching rejected every
    # candidate for this athlete" (count > 0, still interpolated).
    raw_detection_count_by_frame = [len(dets) for dets in all_detections]

    def _allowed_distances(
        frames_since_match: Dict[str, int]
    ) -> Dict[str, float]:
        """Per-athlete pixel-distance budget for this match attempt.

        Scales with however many frames have actually elapsed since
        each athlete's own last successful match, so a normal single-
        frame step gets a tight, physically sane budget while
        recovering from a real multi-frame gap still gets a
        proportionally larger one -- see the module docstring's
        2026-07-30 note.

        UPDATED 2026-08-15 (oplossing C5): if max_frames_since_match
        is set, elapsed_frames is clamped to that ceiling before
        computing the budget, so a very long missed streak no longer
        grows the budget without limit -- see this function's
        docstring in track_all_athletes.
        """
        distances = {}
        for pid, elapsed_frames in frames_since_match.items():
            capped_elapsed_frames = (
                min(elapsed_frames, max_frames_since_match)
                if max_frames_since_match is not None
                else elapsed_frames
            )
            if pixels_per_meter is not None:
                elapsed_s = capped_elapsed_frames / fps
                distances[pid] = max_match_speed_m_s * elapsed_s * pixels_per_meter
            else:
                distances[pid] = fallback_max_match_distance_px * capped_elapsed_frames
        return distances

    mutable_tracks = {
        pid: _MutableTrack(participant_id=pid)
        for pid in assignment.participant_ids
    }

    # Per-athlete, per-absolute-frame -- whether this athlete had at
    # least one in-budget candidate that frame, regardless of whether
    # it won the greedy match -- see AthleteTrack.had_eligible_candidate
    # and _greedy_match's eligible_pids return value (both added
    # 2026-08-14). Populated below in all three passes (seed, forward,
    # backward) so every visited frame has an entry.
    eligible_candidate_by_frame_by_pid: Dict[str, Dict[int, bool]] = {
        pid: {} for pid in assignment.participant_ids
    }

    seed_positions = dict(
        zip(assignment.participant_ids, assignment.click_positions)
    )
    seed_detections = all_detections[assignment.frame_index]
    seed_frames_since_match = {pid: 1 for pid in seed_positions}
    seed_positions, seed_matched, seed_visibility, seed_eligible = _greedy_match(
        seed_positions, seed_detections, _allowed_distances(seed_frames_since_match)
    )
    for pid in seed_positions:
        eligible_candidate_by_frame_by_pid[pid][assignment.frame_index] = pid in seed_eligible
    for pid, (x, y) in seed_positions.items():
        # NaN if this athlete's seed click never matched an actual
        # detection at the reference frame (raw click position used
        # as-is) -- see AthleteTrack.visibility.
        visibility = seed_visibility.get(pid, float("nan")) if pid in seed_matched else float("nan")
        mutable_tracks[pid].append(assignment.frame_index, x, y, visibility)

    forward_positions = dict(seed_positions)
    forward_frames_since_match = {pid: 1 for pid in forward_positions}
    for frame_index in range(assignment.frame_index + 1, total_frames):
        detections = all_detections[frame_index]
        forward_positions, matched, matched_visibility, eligible = _greedy_match(
            forward_positions, detections, _allowed_distances(forward_frames_since_match)
        )
        for pid in forward_frames_since_match:
            eligible_candidate_by_frame_by_pid[pid][frame_index] = pid in eligible
            forward_frames_since_match[pid] = 1 if pid in matched else forward_frames_since_match[pid] + 1
        for pid in matched:
            x, y = forward_positions[pid]
            mutable_tracks[pid].append(frame_index, x, y, matched_visibility.get(pid, float("nan")))

    backward_positions = dict(seed_positions)
    backward_frames_since_match = {pid: 1 for pid in backward_positions}
    for frame_index in range(assignment.frame_index - 1, -1, -1):
        detections = all_detections[frame_index]
        backward_positions, matched, matched_visibility, eligible = _greedy_match(
            backward_positions, detections, _allowed_distances(backward_frames_since_match)
        )
        for pid in backward_frames_since_match:
            eligible_candidate_by_frame_by_pid[pid][frame_index] = pid in eligible
            backward_frames_since_match[pid] = 1 if pid in matched else backward_frames_since_match[pid] + 1
        for pid in matched:
            x, y = backward_positions[pid]
            mutable_tracks[pid].append(frame_index, x, y, matched_visibility.get(pid, float("nan")))

    return {
        pid: track.finalize(
            video_id=assignment.video_id,
            fps=fps,
            frame_width_px=frame_width_px,
            frame_height_px=frame_height_px,
            raw_detection_count_by_frame=raw_detection_count_by_frame,
            decode_failed_by_frame=decode_failed_by_frame,
            had_eligible_candidate_by_frame=eligible_candidate_by_frame_by_pid.get(pid),
            frame_skipped_by_frame=frame_skipped_by_frame,
        )
        for pid, track in mutable_tracks.items()
    }
