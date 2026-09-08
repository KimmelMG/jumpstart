"""Coupling profiles to tracked objects for /identify/.

NEEDS REVIEW -- see the boundary marker in workflow_demo.py.

This is deliberately thin. In this implementation, identity is
established the moment /tracking/ matches each athlete's /who/ seed
position to a real detection -- there is no separate re-identification
pass afterwards, because the seed already carries the participant_id.
So /identify/ here is just the orchestration point that calls /pose/
and /tracking/ together for one video and hands back tracks keyed by
participant_id, matching the "couple profiles to objects by order in
frame" description in the workflow doc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from jumpstart.tracking.tracker import AthleteTrack, track_all_athletes
from jumpstart.who.assignment import FrameAssignment


def identify_athletes_in_video(
    video_path: Path,
    detect_fn,
    assignment: FrameAssignment,
    max_match_speed_m_s: float = 25.0,
    pixels_per_meter: Optional[float] = None,
    detections_cache_path: Optional[Path] = None,
    max_frames_since_match: Optional[int] = None,
    adaptive_skip_frames: Optional[int] = None,
    adaptive_rewind_samples: int = 5,
    adaptive_dense_duration_s: float = 2.5,
    adaptive_trigger_speed_m_s: float = 2.0,
    adaptive_trigger_radius_m: float = 1.5,
) -> Dict[str, AthleteTrack]:
    """Couple profiles to tracked hip positions for one video.

    Args:
        video_path: Path to the video file /who/ was run on.
        detect_fn: A callable frame_bgr -> List[PersonDetection],
            renamed from a hardcoded MediaPipe `landmarker` 2026-08-10
            so /pose/ backends are interchangeable (see
            jumpstart/pose/detector_yolo.py and
            jumpstart/workflow_demo.py's --pose-backend, which builds
            this closure) -- passed straight through to
            track_all_athletes, unchanged otherwise.
        assignment: The FrameAssignment produced by /who/ for this
            video.
        max_match_speed_m_s: Passed through to track_all_athletes
            (renamed from max_match_distance_m 2026-07-30 -- see that
            function's docstring and the module docstring's
            2026-07-30 note for why a flat distance let through
            physically impossible matches on real footage).
        pixels_per_meter: Passed through to track_all_athletes -- see
            that function's docstring and the module docstring's
            2026-07-29 note for why matching tolerance needs this to
            stay consistent across resolutions.
        detections_cache_path: Passed through to track_all_athletes
            (added 2026-08-14) -- see that function's docstring for
            the load-if-exists/save-if-missing pickle caching this
            enables.
        max_frames_since_match: Passed through to track_all_athletes
            (added 2026-08-15) -- see that function's docstring for
            why an unbounded match budget during a long missed streak
            is a robustness problem.
        adaptive_skip_frames: Passed through to track_all_athletes
            (added 2026-08-21, adaptive frame-skip feature -- ported
            from the frame-skip-experiment-2026-08-21 dated copy into
            this real module 2026-09-03, after real-footage
            validation on 2026-08-26). None (the default) disables
            it, giving identical behaviour to before this feature
            existed.
        adaptive_rewind_samples: Passed through to track_all_athletes.
        adaptive_dense_duration_s: Passed through to track_all_athletes.
            Default changed 2026-08-24 from 8.0 to 2.5 -- see
            tracker.py's _detect_all_frames "UPDATED 2026-08-24" note.
            The webapp itself calls this with 4.5s -- see
            jumpstart_webapp/pipeline.run_analysis.
        adaptive_trigger_speed_m_s: Passed through to track_all_athletes.
        adaptive_trigger_radius_m: Passed through to track_all_athletes
            (added 2026-08-24) -- restricts the adaptive-skip trigger
            to detections near a known athlete's /who/ seed position,
            fixing it firing on OTHER people's movement in a
            multi-person frame (see tracker.py's
            _detections_near_seeds).

    Returns:
        A dict of participant_id -> AthleteTrack for this video.
    """
    return track_all_athletes(
        video_path=video_path,
        detect_fn=detect_fn,
        assignment=assignment,
        max_match_speed_m_s=max_match_speed_m_s,
        pixels_per_meter=pixels_per_meter,
        detections_cache_path=detections_cache_path,
        max_frames_since_match=max_frames_since_match,
        adaptive_skip_frames=adaptive_skip_frames,
        adaptive_rewind_samples=adaptive_rewind_samples,
        adaptive_dense_duration_s=adaptive_dense_duration_s,
        adaptive_trigger_speed_m_s=adaptive_trigger_speed_m_s,
        adaptive_trigger_radius_m=adaptive_trigger_radius_m,
    )
