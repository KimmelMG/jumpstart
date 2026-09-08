"""Demonstrates the full Jumpstart workflow, end to end.

Run this script to walk all the way from raw video + participant
info to a spreadsheet of jump parameters:

    FIRST TIME ONLY:
        1. Download the participant info template.

    EVERY USE:
        2. Fill in the template, load it, and build athlete profiles.
        3. Register the batch of videos for this session.
        4. (Optional, --assign-who) Assign athlete order-in-frame per
           video: a reference frame partway through each video is
           shown, and for each athlete (left to right) you click hip
           (choosing the matching profile right after). The FIRST
           athlete assigned at a given resolution (in this video, or
           an earlier one this run) also needs two more clicks --
           top of head, then bottom of feet -- to calibrate real
           pixels-per-metre for that resolution; every other athlete
           at that same resolution reuses it automatically (see the
           pixel-to-metre calibration note below).

    =====================================================================
    REVIEWED BOUNDARY -- everything above this line (/profiles/, /io/,
    /who/) has been read, tested (including synthetic end-to-end runs),
    and reviewed carefully.

    Everything below this line (--run-analysis: /pose/ -> /tracking/ ->
    /identify/ -> /events/ -> /jumps/ -> /calc/ -> /export/) was built
    in one pass to get a working pipeline end to end. It has only been
    verified with synthetic data (fake landmark detections, a simulated
    jump-physics signal) -- never real Jumpstart footage. Known open
    issues to check before trusting the numbers:
        - Pixel-to-metre calibration (added 2026-07-29, confirmed on
          real footage): pixels-per-metre is a property of the
          camera setup (distance/zoom/resolution) at a given depth,
          not of any one athlete, so /who/ only needs a head+feet
          click (against that athlete's own known height) from the
          FIRST athlete assigned at each resolution -- every other
          athlete at that resolution, in this video and every later
          video this run, reuses the same value automatically (see
          jumpstart.who.assignment's 2026-07-29 docstring note and
          Session.pixels_per_meter_by_resolution). This replaces an
          earlier placeholder-constant approach that, confirmed on
          real footage, produced wildly inconsistent jump counts
          (some athletes 0 jumps, others 1-3, no consistent pattern)
          because the free-flight window in /events/ compares pixel-
          derived acceleration against a fixed physical tolerance
          around -9.81 m/s^2, so a wrong pixel scale can cause real
          jumps to be missed entirely or their flight window
          measured too narrowly -- this is not limited to
          countermovement_depth_px (still reported in PIXELS, not
          metres); flight-time-derived parameters are affected too,
          indirectly, through which flight window gets accepted in
          the first place. The calibration cache is NOT persisted
          between separate script runs -- restarting the script
          means the first athlete at each resolution needs the
          head/feet clicks again. If a resolution has no calibration
          yet when --run-analysis reaches it, every athlete at that
          resolution is skipped for that video with a printed
          warning rather than guessed at.
        - Event-detection thresholds (smoothing_window_s, tolerance_m_s2,
          max_gap_s, onset_k, onset_min_consecutive_s, onset_baseline_window_s
          in jumpstart/events/detector.py) are starting points, now
          expressed in seconds (converted to frames via each track's
          own fps) rather than raw frame counts, so behaviour is
          consistent across the study's 60fps/240fps conditions --
          see that module's 2026-07-28 docstring note. On a
          clean synthetic test signal they were within roughly
          15-20% of the true flight time -- expect to retune against
          real footage. Onset (start-of-movement), takeoff and
          landing follow the conventions documented at the top of
          jumpstart/events/detector.py (per-trial adaptive baseline
          + k*SD directional threshold + sustained-frame check for
          onset; last/first ground-contact frame for takeoff/
          landing) -- agreed 2026-07-15, not yet checked against real
          footage.
        - /tracking/'s seeded nearest-neighbour matching assumes a
          small number of well-separated athletes and a mostly static
          camera; it has not been tested against occlusion, crossing
          paths, or a moving camera. Its matching tolerance
          (--max-match-speed-m-s, added 2026-07-29, renamed and
          reworked 2026-07-30, confirmed on real footage) is now
          expressed as a max plausible real SPEED (metres/second),
          scaled by how much real time has elapsed since each
          athlete's last successful match and converted to pixels via
          this video's own calibrated pixels-per-metre -- a fixed
          pixel radius (2026-07-29) previously lost an athlete
          completely after the reference frame on a high-resolution
          video, while the same session's lower-resolution encode
          tracked them fine the whole way through, even though the
          athlete never left frame; a flat distance (whether metres
          or pixels) still let through physically impossible single-
          frame matches at high fps (1-8% of frames on real
          1080p/240fps footage implied 30-70 m/s hip movement) --
          see jumpstart.tracking.tracker's 2026-07-29 and 2026-07-30
          docstring notes.
        - Requires a downloaded MediaPipe pose landmarker model file
          (see jumpstart/pose/detector.py) that is not included here.
    If you are reviewing this codebase later and want to know exactly
    where "solid" work ends and "first draft" work begins, this
    comment block is that line.
    =====================================================================

    =====================================================================
    OPEN REQUIREMENTS -- not implemented yet, raise these again the next
    time this script gets touched:

        1. Video format support -- DONE 2026-08-28. SUPPORTED_EXTENSIONS
           in jumpstart/io/video_input.py now also accepts .avi, .mkv,
           .wmv, .3gp, .3g2, .webm, .mts, .m2ts, .flv, .mpg and .mpeg
           (on top of the original .mp4/.mov/.m4v). /io/ itself still
           only checks the extension (kept deliberately "dumb" -- see
           that module's docstring), so a recognized extension does
           NOT guarantee this machine's OpenCV build can actually
           decode that file's codec. That is now checked separately,
           once per video, right before /tracking/ opens it for real
           (see _check_video_decodable below and its call site in
           main()) -- a video that fails this check is skipped with a
           clear message instead of silently producing an
           empty/garbage track.

        2. Phone usability. The eventual HTML front-end itself is
           expected to work fine on a phone. What's NOT yet designed
           for phone use is the underlying functionality this script
           exposes: things like uploading/registering large phone-shot
           videos over a mobile connection, and how much of the
           /pose/-/tracking/-/events/ processing (if any) could
           realistically run on or be triggered comfortably from a
           phone versus needing a proper back-end. This needs concrete
           design thought once the front-end architecture decision
           (FastAPI+HTMX / React / Streamlit -- see the workflow doc)
           is made.
    =====================================================================

Usage (PowerShell, from the folder that contains the jumpstart/
package -- one level above jumpstart/, not inside it):
    python -m jumpstart.workflow_demo --first-time

    python -m jumpstart.workflow_demo --participants participants.xlsx `
        --videos video1.mp4 video2.mp4 --assign-who --run-analysis `
        --export-path results.xlsx

    python -m jumpstart.workflow_demo --participants participants.xlsx `
        --video-folder "C:\Videos\Session1" --assign-who --run-analysis

    python -m jumpstart.workflow_demo --participants participants.xlsx `
        --video-folder "C:\Videos\Session1" --no-recursive `
        --assign-who --run-analysis

--assign-who and --run-analysis are both needed to actually produce
jump parameters -- without them this only registers participants/
videos and prints a status summary (useful for a quick sanity check
before committing to a full run).
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

if __name__ == "__main__" and __package__ is None:
    # Allow running this file directly (python workflow_demo.py ...)
    # in addition to the recommended `python -m jumpstart.workflow_demo`.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jumpstart.io import (
    discover_videos_in_folder,
    get_recording_datetime,
    register_videos,
)
from jumpstart.profiles import build_profiles, load_participant_table
from jumpstart.profiles.manager import create_participant_template
from jumpstart.session import Session
from jumpstart.who import assign_athlete_order, extract_reference_frame

# ============================================================================
# REVIEWED BOUNDARY -- see the module docstring above for the full note.
# Everything imported below this line belongs to the --run-analysis path
# (/pose/ through /export/) and has NOT been checked against real footage.
# ============================================================================
from jumpstart.calc import (
    calculate_jump_parameters,
    check_calibration,
    implied_pixels_per_meter_for_jump,
)
from jumpstart.events import GRAVITY_M_S2, compute_kinematics
from jumpstart.export import export_results
from jumpstart.identify import identify_athletes_in_video
from jumpstart.jumps import build_jumps_for_athlete
from jumpstart.pose import (
    create_pose_landmarker,
    create_yolo_pose_model,
    detect_people,
    detect_people_yolo,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the workflow demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--first-time",
        action="store_true",
        help="Only create the participant template and exit.",
    )
    parser.add_argument(
        "--template-path",
        type=Path,
        default=Path("participant_template.xlsx"),
        help="Where to write/read the participant template.",
    )
    parser.add_argument(
        "--participants",
        type=Path,
        help="Path to a filled-in participant file (.xlsx or .csv).",
    )
    video_source = parser.add_mutually_exclusive_group()
    video_source.add_argument(
        "--videos",
        type=Path,
        nargs="+",
        help="Paths to individual video files to register.",
    )
    video_source.add_argument(
        "--video-folder",
        type=Path,
        help=(
            "Folder containing video files to register -- every "
            "supported file in it (and its subfolders, by default; "
            "see --no-recursive) is registered, in one call. Use "
            "this instead of --videos for a whole session's worth of "
            "footage at once."
        ),
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help=(
            "With --video-folder, only look directly inside that "
            "folder, not its subfolders."
        ),
    )
    parser.add_argument(
        "--assign-who",
        action="store_true",
        help=(
            "After registering videos, interactively assign athlete "
            "order-in-frame for each video (/who/ step)."
        ),
    )
    parser.add_argument(
        "--frame-fraction",
        type=float,
        default=0.5,
        help=(
            "Where in each video to grab the reference frame for "
            "/who/, as a fraction of total frames (default 0.5 = "
            "halfway, so athletes have had time to get in position)."
        ),
    )
    # -- everything below this point is the --run-analysis path; see
    # the REVIEWED BOUNDARY note in the module docstring. --
    parser.add_argument(
        "--run-analysis",
        action="store_true",
        help=(
            "After --assign-who, run the rest of the pipeline "
            "(/pose/ -> /tracking/ -> /identify/ -> /events/ -> "
            "/jumps/ -> /calc/ -> /export/) and write out jump "
            "parameters. NEEDS REVIEW -- see the module docstring."
        ),
    )
    parser.add_argument(
        "--pose-backend",
        choices=["mediapipe", "yolo"],
        default="mediapipe",
        help=(
            "Added 2026-08-10: which /pose/ model to run. 'mediapipe' "
            "(default) is the original backend -- lite and full "
            "variants both tested on real footage, no measurable "
            "difference. 'yolo' is a new, NOT YET VALIDATED "
            "alternative (jumpstart/pose/detector_yolo.py), added "
            "after real footage showed MediaPipe losing the athlete's "
            "hip position entirely (not just low-confidence) during "
            "the fast/blurry part of most jumps -- untested whether "
            "YOLO actually handles this better. Requires `pip install "
            "ultralytics` (not included/installed by this project)."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "Path to a pose model file, for either backend. For "
            "--pose-backend mediapipe: a .task pose landmarker file; "
            "leave unset to auto-download the default model to "
            "~/.jumpstart/models/ the first time it's needed (see "
            "jumpstart/pose/detector.py). For --pose-backend yolo: a "
            ".pt YOLO-pose weights file; leave unset to let "
            "Ultralytics auto-download --yolo-model-name to its own "
            "cache the first time it's needed (see "
            "jumpstart/pose/detector_yolo.py) -- only set this for a "
            "different model variant or an offline copy, same as the "
            "MediaPipe path."
        ),
    )
    parser.add_argument(
        "--num-poses",
        type=int,
        default=10,
        help=(
            "--pose-backend mediapipe only: max number of people "
            "MediaPipe should detect per frame. Set a bit higher than "
            "the number of athletes actually in frame."
        ),
    )
    parser.add_argument(
        "--min-pose-detection-confidence",
        type=float,
        default=0.5,
        help=(
            "--pose-backend mediapipe only: added 2026-08-15 "
            "(oplossing B1 uit Jumpstart_oplossingsrichtingen.docx). "
            "Previously never set explicitly, so MediaPipe silently "
            "used its own default (~0.5). Lower this (e.g. 0.2-0.3) "
            "to test whether MediaPipe was reporting 'nobody found' "
            "on frames where a partially-visible athlete is genuinely "
            "in frame -- compare raw_detection_count in the debug CSV "
            "before/after on the same video."
        ),
    )
    parser.add_argument(
        "--min-pose-presence-confidence",
        type=float,
        default=0.5,
        help=(
            "--pose-backend mediapipe only: added 2026-08-15, same "
            "reasoning as --min-pose-detection-confidence -- "
            "MediaPipe's separate threshold for whether a detected "
            "pose is confidently 'present' once found at all."
        ),
    )
    parser.add_argument(
        "--yolo-model-name",
        type=str,
        default="yolov8n-pose.pt",
        help=(
            "--pose-backend yolo only: which YOLO-pose variant to "
            "auto-download/use when --model-path is not set. "
            "'yolov8n-pose.pt' (nano, fastest/least accurate) through "
            "'yolov8x-pose.pt' (extra-large, slowest/most accurate), "
            "or a newer 'yolo11n-pose.pt'-style name -- see "
            "https://docs.ultralytics.com/tasks/pose/#models for the "
            "full list. Starting with the smallest (nano) variant is "
            "reasonable since MediaPipe's own lite-vs-full comparison "
            "found no accuracy difference on this project's footage, "
            "i.e. model capacity was probably not the bottleneck; "
            "raise this only if YOLO nano's own results look "
            "promising but not yet accurate enough."
        ),
    )
    parser.add_argument(
        "--yolo-conf-threshold",
        type=float,
        default=0.25,
        help=(
            "--pose-backend yolo only: minimum YOLO person-detection "
            "confidence to keep a detected person at all (YOLO's own "
            "default is 0.25). This is separate from the per-keypoint "
            "visibility score written to --debug-csv-dir's visibility "
            "column."
        ),
    )
    parser.add_argument(
        "--pose-device",
        type=str,
        default=None,
        help=(
            "--pose-backend yolo only: inference device to pass to "
            "Ultralytics, e.g. 'cpu' or 'cuda:0'. Leave unset to let "
            "Ultralytics auto-select (GPU if available, else CPU)."
        ),
    )
    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=480,
        help=(
            "--pose-backend yolo only: internal resolution (pixels, "
            "longest side) YOLO resizes each frame to before running "
            "inference. Ultralytics' own default is 640; inference "
            "cost scales roughly with the square of this number, so "
            "480 is noticeably faster than 640 on CPU-only machines "
            "(no dedicated GPU) at some cost to detecting small/far-"
            "away people. NEEDS REVIEW: this speed/accuracy tradeoff "
            "has not been checked against real footage -- raise this "
            "back toward 640 if YOLO starts missing athletes, lower "
            "it further (e.g. 320) if CPU-only speed is still not "
            "practical."
        ),
    )
    parser.add_argument(
        "--max-match-speed-m-s",
        type=float,
        default=25.0,
        help=(
            "Max plausible REAL speed (metres/second) an athlete's "
            "hip may move for /tracking/ to still count a detection "
            "as the same athlete. Renamed from --max-match-distance-m "
            "2026-07-30: the allowed pixel distance for a match is "
            "now this speed times however much real time has "
            "actually elapsed since that athlete's last successful "
            "match, converted to pixels via /who/-calibrated "
            "pixels-per-metre. Raised same-day from an initial 8.0 "
            "to 25.0 after a real full-pipeline test made results "
            "WORSE: real MediaPipe hip-position noise is a "
            "continuous, heavy-tailed distribution with no clean "
            "cutoff between ordinary jitter and a genuine mismatch, "
            "so 8.0 rejected a meaningful share of real (if noisy) "
            "detections, not just genuine mismatches -- see "
            "jumpstart/tracking/tracker.py's 2026-07-30 CORRECTION "
            "docstring note for the full analysis."
        ),
    )
    parser.add_argument(
        "--max-frames-since-match",
        type=int,
        default=None,
        help=(
            "Added 2026-08-15 (oplossing C5 uit "
            "Jumpstart_oplossingsrichtingen.docx). The match budget "
            "for a missing athlete grows UNBOUNDED with how many "
            "frames have passed since their last match (budget = "
            "speed * elapsed_time). If set, caps how many frames of "
            "growth are allowed before the budget stops expanding "
            "further, so a very long missed streak doesn't turn "
            "matching into an effectively random pick. None (default) "
            "keeps the old unbounded behaviour -- try e.g. 30-60 "
            "(frames, not seconds) once you have a feel for how long "
            "real occlusions/misses last on this footage."
        ),
    )
    parser.add_argument(
        "--tolerance-m-s2",
        type=float,
        default=3.0,
        help=(
            "How far (m/s^2) computed acceleration may drift from "
            "-9.81 and still count as free flight, in "
            "jumpstart/events/detector.py. Previously not exposed "
            "here at all (always the hardcoded default) -- now "
            "tunable so you can experiment on real footage without "
            "needing a code change each time. See that module's "
            "2026-07-29 docstring note for why this value interacts "
            "with real-footage noise levels."
        ),
    )
    parser.add_argument(
        "--smoothing-window-s",
        type=float,
        default=7 / 120,
        help=(
            "Savitzky-Golay smoothing/derivative window length "
            "(seconds) in jumpstart/events/detector.py, floored at "
            "MIN_DERIVATIVE_WINDOW_SAMPLES actual samples regardless "
            "of this value (see that module's 2026-07-29 docstring "
            "note). Larger = smoother but softer/later takeoff-"
            "landing edges; smaller = sharper edges but noisier."
        ),
    )
    parser.add_argument(
        "--min-flight-duration-s",
        type=float,
        default=0.13,
        help=(
            "Minimum flight-window length (seconds) in "
            "jumpstart/events/detector.py's _find_flight_windows to "
            "count as a real jump rather than noise. Raised "
            "2026-07-30 from an implicit ~0.1s after real footage "
            "still showed far too many false-positive jumps -- see "
            "that module's 2026-07-30 docstring note. On this "
            "project's own real debug-CSV data, 0.13s cut total "
            "false-positive deviation across 16 real athlete/video "
            "combinations from roughly 107-184 down to 24 -- real "
            "progress, not a complete fix."
        ),
    )
    parser.add_argument(
        "--no-refine-edges",
        action="store_true",
        help=(
            "Turn OFF stage 2 of the two-stage jump-edge detection "
            "(jumpstart/events/detector.py's _refine_flight_edges, "
            "added 2026-09-03). Stage 2 is ON by default: after the "
            "wide, noise-robust window has located a jump, it places "
            "takeoff/landing on the velocity maximum/minimum inside "
            "that stretch. Measured against the full manual "
            "reference (81 jumps) this removed a -0.108m jump-height "
            "bias (to +0.022m) and lifted CCC from 0.21 to 0.78. Use "
            "this flag only to reproduce a pre-2026-09-03 run."
        ),
    )
    parser.add_argument(
        "--export-path",
        type=Path,
        default=Path("jump_parameters.xlsx"),
        help="Where to write the final jump parameters (.xlsx or .csv).",
    )
    parser.add_argument(
        "--debug-csv-dir",
        type=Path,
        default=None,
        help=(
            "Added 2026-07-30 to diagnose jump-detection accuracy on "
            "real footage (see the module docstring): if set, for "
            "every athlete/video this also writes a "
            "<video_id>_<participant_id>.csv into this folder with "
            "the per-frame height/velocity/acceleration signal that "
            "/events/ actually ran free-flight detection on, plus "
            "whether each frame fell inside the tolerance_m_s2 band "
            "around -9.81 m/s^2. Share the CSV for an athlete/video "
            "that found the wrong number of jumps to retune "
            "tolerance_m_s2/smoothing_window_s against real noise "
            "instead of only synthetic tests."
        ),
    )
    parser.add_argument(
        "--detections-cache-dir",
        type=Path,
        default=None,
        help=(
            "Added 2026-08-14 to speed up iterating on /tracking/'s "
            "matching logic without re-running the much slower "
            "/pose/ detection pass every time: if set, for every "
            "video this caches the raw per-frame pose detections to "
            "<this dir>/<video filename stem>.pkl the first time, and "
            "loads from that file on later runs instead of "
            "re-detecting (keyed on the filename, not video_id, since "
            "video_id is only a per-invocation sequential label and "
            "would otherwise collide across separate single-video "
            "runs -- see the 2026-08-17 fix note at this flag's call "
            "site). Delete the relevant .pkl (or the whole folder) if "
            "you change --pose-backend, the model, or anything else "
            "that would change what /pose/ itself returns -- the "
            "cache has no way to know its contents are stale."
        ),
    )
    parser.add_argument(
        "--disk-space-margin-gb",
        type=float,
        default=2.0,
        help=(
            "Added 2026-08-07 for OneDrive-backed workspaces where "
            "videos are too large to keep locally and get downloaded "
            "on demand: extra free disk space (GB), on top of each "
            "video's own file size, required before /pose/ starts "
            "processing it. If there isn't enough, the run PAUSES "
            "with a message asking you to free up space and press "
            "Enter, instead of failing partway through a video with "
            "an unclear error (see _wait_for_disk_space's docstring "
            "for why this has to be checked before opening the video, "
            "not caught as an error during it)."
        ),
    )
    return parser.parse_args()


def _wait_for_disk_space(video_path: Path, min_extra_bytes: int) -> None:
    """Block until there's enough free disk space to safely process video_path.

    Added 2026-08-07 after a real run on a OneDrive-backed workspace
    (videos too large to keep locally, so OneDrive downloads each one
    on demand) ran into disk-space pressure partway through a long
    multi-video batch. cv2.VideoCapture (see /tracking/'s
    _detect_all_frames) needs the WHOLE video file downloaded locally
    before it can decode frames -- OneDrive's Files On-Demand fetches
    a cloud-only file in full on first real read, it does not stream
    frame-by-frame -- and OpenCV gives no clean, catchable "disk full"
    signal when that download can't complete: it just silently fails
    to read frames, indistinguishable from any other read failure.
    Checking free space against the video's own file size BEFORE ever
    opening it sidesteps that blind spot entirely, since a video that
    can't yet fit is never opened in the first place.

    This only guards the READ side (/pose/ downloading+decoding a
    video). Write-side disk-full errors (debug CSVs, the results
    export) raise a normal, catchable OSError instead -- see
    _run_with_disk_pause for that path.

    Args:
        video_path: The video about to be processed. Its file size
            (video_path.stat().st_size) is what needs to fit -- this
            works even for a not-yet-downloaded OneDrive placeholder,
            since OneDrive reports the real logical file size via
            stat() regardless of local hydration state, not some
            small placeholder size.
        min_extra_bytes: Extra headroom required on top of the
            video's own size (see --disk-space-margin-gb) -- covers
            that video's own debug-CSV output (a few MB per athlete
            in real runs, tiny next to video file sizes) plus a
            margin for whatever temporary space OneDrive's own
            download process needs.
    """
    needed_bytes = video_path.stat().st_size + min_extra_bytes
    first_attempt = True
    while True:
        free_bytes = shutil.disk_usage(str(video_path.parent)).free
        if free_bytes >= needed_bytes:
            if not first_attempt:
                print("    Genoeg ruimte vrij -- ga verder.")
            return
        first_attempt = False
        needed_gb = needed_bytes / 1024 ** 3
        free_gb = free_bytes / 1024 ** 3
        print(
            f"    Te weinig vrije schijfruimte om {video_path.name} te "
            f"verwerken: {free_gb:.1f} GB vrij, ~{needed_gb:.1f} GB "
            "nodig (videogrootte + marge). Maak ruimte vrij (bijv. via "
            "OneDrive 'Vrij ruimte op' op al verwerkte video's) en druk "
            "op Enter om het opnieuw te proberen..."
        )
        input()


def _check_video_decodable(video_path: Path) -> Optional[str]:
    """Verify OpenCV can actually open and decode this video's first frame.

    Added 2026-08-28 alongside the widened SUPPORTED_EXTENSIONS (see
    jumpstart/io/video_input.py and this module's OPEN REQUIREMENTS
    item 1). A recognized extension only means /io/ was willing to
    register the file -- it says nothing about whether this machine's
    OpenCV build has a working codec for it. Without this check,
    jumpstart/tracking/tracker.py's track_all_athletes opens the file
    with cv2.VideoCapture and, if that silently fails, just carries on
    with total_frames == 0 / fps == 0 and no detections at all --
    surfacing hours later as an empty results file with no clear
    reason why, instead of a clear, immediate, per-video message here.

    Deliberately NOT done in jumpstart/io/video_input.py itself --
    that module is kept extension-only on purpose (see its own
    docstring) so a whole team's batch of videos can be registered
    quickly without opening every file. This runs once per video,
    right before --run-analysis actually opens it, and only AFTER
    _wait_for_disk_space has confirmed the (possibly not-yet-
    downloaded OneDrive) file can be read at all -- same ordering
    reason as that function's own docstring: reading a frame here
    triggers the same full-file download as /tracking/ would.

    Args:
        video_path: The video about to be processed.

    Returns:
        None if the video opened and its first frame decoded
        successfully. Otherwise a short, human-readable reason to
        show the user.
    """
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return (
                "OpenCV kon dit bestand niet openen -- vermoedelijk "
                f"ontbreekt er codec-/containerondersteuning voor "
                f"{video_path.suffix!r}-bestanden op deze machine."
            )
        frame_read_ok, _frame = capture.read()
        if not frame_read_ok:
            return (
                "OpenCV kon het bestand wel openen, maar geen eerste "
                "frame decoderen -- vermoedelijk een codec die deze "
                "OpenCV-installatie niet ondersteunt, ook al opent de "
                "container zelf wel."
            )
        return None
    finally:
        capture.release()


def _run_with_disk_pause(write_fn, description: str):
    """Run write_fn(), pausing and retrying on a disk-related OSError
    instead of letting it crash the whole run.

    Added 2026-08-07 alongside _wait_for_disk_space -- that function
    guards video READS (see its docstring for why cv2 can't be
    trusted to fail cleanly there); this one guards WRITES (debug
    CSVs, the results export), where Python's own file I/O DOES raise
    a normal, catchable OSError when the disk is full. As a side
    effect this also catches a locked/in-use file (e.g. the export
    .xlsx still open in Excel) -- not the primary goal, but a welcome
    one, since that's another common reason a long run used to crash.

    Args:
        write_fn: A zero-argument callable that performs the write and
            returns whatever the caller wants back.
        description: Short human-readable description of what's being
            written, for the pause message.

    Returns:
        Whatever write_fn() returns, once it succeeds.
    """
    while True:
        try:
            return write_fn()
        except OSError as exc:
            print(
                f"    Kon {description} niet wegschrijven ({exc}). Dit "
                "is vaak schijfruimte (maak ruimte vrij) of een "
                "bestand dat nog open staat in een ander programma "
                "(bijv. Excel -- sluit het). Druk op Enter om het "
                "opnieuw te proberen..."
            )
            input()


def _write_debug_csv(
    output_dir: Path,
    track,
    pixels_per_meter: float,
    tolerance_m_s2: float,
    smoothing_window_s: float,
    jumps,
    video_name: str,
) -> Path:
    """Dump one athlete/video's kinematics signal to a CSV for
    manual inspection (added 2026-07-30 -- see --debug-csv-dir help).

    Recomputes exactly what build_jumps_for_athlete's
    detect_jumps_from_track ran free-flight detection on (same
    compute_kinematics call, same arguments), so this is not an
    approximation of the real signal -- it is the real signal.

    UPDATED 2026-08-07: now also exports was_interpolated and
    visibility per frame (both already tracked internally on
    AthleteTrack -- see jumpstart/tracking/tracker.py -- but never
    written out before). Added after comparing real debug-CSV data
    against manually-verified ground truth showed that a large chunk
    of detected/missed jumps trace back to the raw hip signal itself
    being unreliable right around takeoff/landing (either frozen
    because MediaPipe missed the detection and the position was
    linearly interpolated, or jittery because MediaPipe's confidence
    genuinely dropped during the fast, blurry part of the jump) --
    these two columns make that visible directly in the CSV instead
    of having to infer it from the shape of the height/velocity/
    acceleration curves.

    UPDATED 2026-08-07 (same day, follow-up): also exports
    raw_detection_count -- added after was_interpolated/visibility on
    real footage showed something bigger than expected: 85-96% of ALL
    frames interpolated, video-wide, not just around jumps, while
    visibility stayed consistently high (>0.99) whenever a match DID
    happen. That combination means the problem usually isn't "MediaPipe
    detected this person with low confidence" (which visibility alone
    would already show) -- it's "no accepted detection at all for this
    athlete that frame." raw_detection_count tells apart WHY: 0 means
    MediaPipe found nobody in the whole frame (a /pose/ or model-choice
    problem); a track still marked was_interpolated on a frame where
    raw_detection_count > 0 means MediaPipe DID see someone, but
    /tracking/'s greedy matching (distance budget, or another athlete's
    claim winning first) rejected every candidate for this athlete (a
    /tracking/ problem instead). See jumpstart/tracking/tracker.py's
    AthleteTrack.raw_detection_count docstring for the full reasoning.

    UPDATED 2026-08-17: the output filename and two new leading CSV
    columns (video_id, video_filename) now identify the video by its
    actual filename, not just track.video_id ("V001" etc) -- video_id
    is only a per-script-invocation sequential label (see
    jumpstart/io/video_input.py's register_videos), so a folder full
    of debug CSVs from several separate test runs was previously
    impossible to tell apart by filename alone.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_name}_{track.participant_id}.csv"

    height_m, velocity_m_s, acceleration_m_s2 = compute_kinematics(
        track.hip_y,
        track.fps,
        pixels_per_meter,
        smoothing_window_s=smoothing_window_s,
    )
    in_tolerance_band = np.abs(acceleration_m_s2 - (-GRAVITY_M_S2)) <= tolerance_m_s2

    # Label each frame with which detected jump (if any) it falls
    # inside, and which phase, so it's obvious in the CSV where
    # /events/ did (or didn't) find a jump relative to the raw signal.
    phase_by_local_index = {}
    for jump_number, jump in enumerate(jumps, start=1):
        start_local = list(track.frame_indices).index(jump.start_frame)
        takeoff_local = list(track.frame_indices).index(jump.takeoff_frame)
        landing_local = list(track.frame_indices).index(jump.landing_frame)
        for local_index in range(start_local, takeoff_local + 1):
            phase_by_local_index[local_index] = f"J{jump_number:02d}_countermovement"
        for local_index in range(takeoff_local + 1, landing_local):
            phase_by_local_index[local_index] = f"J{jump_number:02d}_flight"
        phase_by_local_index[landing_local] = f"J{jump_number:02d}_landing"

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "video_id",
                "video_filename",
                "pixels_per_meter",
                "frame_index",
                "hip_y_px",
                "height_m",
                "velocity_m_s",
                "acceleration_m_s2",
                "in_tolerance_band",
                "detected_jump_phase",
                "was_interpolated",
                "visibility",
                "raw_detection_count",
                "decode_failed",
                "had_eligible_candidate",
            ]
        )
        for local_index, frame_index in enumerate(track.frame_indices):
            visibility = track.visibility[local_index]
            raw_count = track.raw_detection_count[local_index]
            writer.writerow(
                [
                    track.video_id,
                    video_name,
                    pixels_per_meter,
                    frame_index,
                    track.hip_y[local_index],
                    height_m[local_index],
                    velocity_m_s[local_index],
                    acceleration_m_s2[local_index],
                    bool(in_tolerance_band[local_index]),
                    phase_by_local_index.get(local_index, ""),
                    bool(track.was_interpolated[local_index]),
                    # Written as "" rather than "nan" for a missing
                    # score (no real detection behind this sample --
                    # see AthleteTrack.visibility), so it's blank in
                    # Excel/pandas instead of the string "nan".
                    "" if np.isnan(visibility) else visibility,
                    # -1 means "unknown" (finalize() called without
                    # raw_detection_count_by_frame -- shouldn't happen
                    # via this workflow, only via direct/test calls),
                    # written as "" for the same reason as visibility.
                    "" if raw_count < 0 else int(raw_count),
                    # Added 2026-08-14 -- see AthleteTrack.decode_failed
                    # and AthleteTrack.had_eligible_candidate.
                    bool(track.decode_failed[local_index]),
                    bool(track.had_eligible_candidate[local_index]),
                ]
            )
    return output_path


def main() -> None:
    """Run the full workflow end to end."""
    args = parse_args()

    if args.first_time:
        template_path = create_participant_template(args.template_path)
        print(f"Template created at: {template_path}")
        return

    if args.participants is None or (
        args.videos is None and args.video_folder is None
    ):
        raise SystemExit(
            "--participants and either --videos or --video-folder "
            "are required unless --first-time is set."
        )

    if args.video_folder is not None:
        video_paths = discover_videos_in_folder(
            args.video_folder, recursive=not args.no_recursive
        )
        scope = "this folder only" if args.no_recursive else "including subfolders"
        print(
            f"Found {len(video_paths)} video(s) in {args.video_folder} "
            f"({scope})"
        )
    else:
        video_paths = args.videos

    participant_table = load_participant_table(args.participants)
    participants = build_profiles(participant_table)

    session = Session()
    session.participants = participants
    session.videos = register_videos(video_paths, session.session_id)

    if args.assign_who:
        for video in session.videos:
            print(f"\n--- /who/: {video.video_id} ({video.path.name}) ---")
            frame, frame_index = extract_reference_frame(
                video.path, fraction=args.frame_fraction
            )
            session.who_assignments[video.video_id] = assign_athlete_order(
                frame=frame,
                video_id=video.video_id,
                frame_index=frame_index,
                profiles=session.participants,
                pixels_per_meter_cache=session.pixels_per_meter_by_resolution,
            )

    print()
    print(session.summary())

    # =========================================================================
    # REVIEWED BOUNDARY -- everything above this point in main() has been
    # reviewed and tested. Everything from here on (--run-analysis) is a
    # first draft; see the module docstring for the full list of caveats.
    # =========================================================================
    if not args.run_analysis:
        return

    if not session.who_assignments:
        raise SystemExit(
            "--run-analysis needs at least one video with an "
            "athlete order-in-frame assignment -- pass --assign-who "
            "too."
        )

    participants_by_id = {p.participant_id: p for p in session.participants}

    # detect_fn is a plain callable frame_bgr -> List[PersonDetection]
    # -- added 2026-08-10 alongside --pose-backend so /tracking/ (see
    # jumpstart/tracking/tracker.py's detect_fn parameter) does not
    # need to know which backend produced it. Each backend's own
    # colour-space requirement (MediaPipe needs RGB, YOLO/Ultralytics
    # accepts the raw BGR frame cv2 already hands back) is handled
    # right here, once, rather than inside /tracking/.
    if args.pose_backend == "yolo":
        print(
            "\n--- /pose/: loading YOLO-pose model "
            f"({args.yolo_model_name if args.model_path is None else args.model_path}) "
            "-- NOT YET VALIDATED against real footage, see "
            "jumpstart/pose/detector_yolo.py's module docstring ---"
        )
        yolo_model = create_yolo_pose_model(
            model_path=args.model_path,
            model_name=args.yolo_model_name,
            device=args.pose_device,
        )

        def detect_fn(
            frame_bgr, _model=yolo_model, _conf=args.yolo_conf_threshold, _imgsz=args.yolo_imgsz
        ):
            return detect_people_yolo(_model, frame_bgr, conf_threshold=_conf, imgsz=_imgsz)

    else:
        print("\n--- /pose/: loading MediaPipe pose landmarker ---")
        landmarker = create_pose_landmarker(
            args.model_path,
            num_poses=args.num_poses,
            min_pose_detection_confidence=args.min_pose_detection_confidence,
            min_pose_presence_confidence=args.min_pose_presence_confidence,
        )

        def detect_fn(frame_bgr, _landmarker=landmarker):
            # MediaPipe needs RGB; cv2 (and this whole pipeline
            # upstream of here) works in BGR -- see
            # jumpstart/pose/detector.py's detect_people docstring.
            return detect_people(_landmarker, frame_bgr[:, :, ::-1])

    all_parameters = []
    for video in session.videos:
        assignment = session.who_assignments.get(video.video_id)
        if assignment is None:
            print(
                f"Skipping {video.video_id} ({video.path.name}): "
                "no /who/ assignment."
            )
            continue

        print(
            f"\n--- {video.video_id} ({video.path.name}): "
            "/tracking/ + /identify/ -- following "
            f"{len(assignment.participant_ids)} athlete(s) through "
            "the whole video ---"
        )
        # Checked BEFORE opening the video (not after a failed read) --
        # see _wait_for_disk_space's docstring for why this has to be
        # a proactive check rather than catching an error from cv2.
        _wait_for_disk_space(
            video.path, min_extra_bytes=int(args.disk_space_margin_gb * 1024 ** 3)
        )
        # Added 2026-08-28 alongside the widened video-format support
        # (see _check_video_decodable's docstring): confirms this
        # machine's OpenCV build can actually decode the file before
        # /tracking/ spends any time on it.
        decode_error = _check_video_decodable(video.path)
        if decode_error is not None:
            print(f"    {video.video_id}: {decode_error} -- video wordt overgeslagen.")
            continue
        # This video's real pixels-per-metre, if /who/ has calibrated
        # its resolution yet -- used to keep /tracking/'s matching
        # tolerance consistent in real-world terms (see
        # jumpstart.tracking.tracker's 2026-07-29 docstring note).
        pixels_per_meter_for_tracking = session.pixels_per_meter_by_resolution.get(
            (assignment.frame_width_px, assignment.frame_height_px)
        )
        # None (no caching) unless --detections-cache-dir was passed --
        # see that flag's help for what this does.
        # Keyed on the filename stem, not video.video_id -- video_id
        # is only a per-invocation sequential label ("V001" for
        # whichever video is first in *this* --videos call), so two
        # separate single-video runs reusing the same
        # --detections-cache-dir would otherwise silently load each
        # other's cached detections (found 2026-08-17: crashed with
        # an IndexError when a shorter cached video's detections were
        # loaded for a longer one; would have loaded WRONG data
        # without any error if the frame counts had been close
        # enough not to crash).
        detections_cache_path = (
            args.detections_cache_dir / f"{video.path.stem}.pkl"
            if args.detections_cache_dir is not None
            else None
        )
        tracks = identify_athletes_in_video(
            video_path=video.path,
            detect_fn=detect_fn,
            assignment=assignment,
            max_match_speed_m_s=args.max_match_speed_m_s,
            pixels_per_meter=pixels_per_meter_for_tracking,
            detections_cache_path=detections_cache_path,
            max_frames_since_match=args.max_frames_since_match,
        )

        recorded_at = get_recording_datetime(video.path)
        if recorded_at is None:
            print(
                f"    {video.video_id}: could not read a recording "
                "date/time from the video file -- recorded_at will "
                "be empty for this video's rows."
            )

        for participant_id, track in tracks.items():
            # Real pixels-per-metre for this athlete's actual decoded
            # resolution -- calibrated once (by whichever athlete was
            # first assigned at this resolution, in this video or an
            # earlier one) and shared across the whole run. See
            # jumpstart.who.assignment's 2026-07-29 docstring note.
            pixels_per_meter = session.pixels_per_meter_by_resolution.get(
                (track.frame_width_px, track.frame_height_px)
            )
            if pixels_per_meter is None:
                print(
                    f"    {video.video_id}/{participant_id}: no "
                    f"calibration yet for resolution "
                    f"{track.frame_width_px}x{track.frame_height_px} "
                    "-- skipping jump detection (0 jumps) rather than "
                    "guessing a pixel scale."
                )
                continue

            print(
                f"    {video.video_id}/{participant_id}: "
                "/events/ + /jumps/ -- detecting jumps..."
            )
            jumps = build_jumps_for_athlete(
                track,
                pixels_per_meter=pixels_per_meter,
                tolerance_m_s2=args.tolerance_m_s2,
                smoothing_window_s=args.smoothing_window_s,
                min_flight_duration_s=args.min_flight_duration_s,
                refine_edges=not args.no_refine_edges,
            )
            print(
                f"    {video.video_id}/{participant_id}: "
                f"found {len(jumps)} jump(s), running /calc/..."
            )
            # Added 2026-09-03: calibration net -- see
            # jumpstart/calc/calibration_check.py.
            implied_by_jump = {
                jump.jump_number: implied_pixels_per_meter_for_jump(track, jump)
                for jump in jumps
            }
            calibration = check_calibration(track, jumps, pixels_per_meter)
            if not calibration.within_tolerance:
                print(
                    f"    WARNING {video.video_id}/{participant_id}: "
                    f"calibration used {calibration.used_pixels_per_meter:.1f} px/m, "
                    f"but the flight phase of {calibration.jumps_used} jump(s) "
                    f"implies {calibration.implied_pixels_per_meter:.1f} px/m "
                    f"({(calibration.ratio - 1) * 100:+.0f}%). Re-check the "
                    "head+feet click for this video -- is the camera distance "
                    "still the one it was calibrated at?"
                )
            if args.debug_csv_dir is not None:
                debug_path = _run_with_disk_pause(
                    lambda: _write_debug_csv(
                        args.debug_csv_dir,
                        track,
                        pixels_per_meter=pixels_per_meter,
                        tolerance_m_s2=args.tolerance_m_s2,
                        smoothing_window_s=args.smoothing_window_s,
                        jumps=jumps,
                        video_name=video.path.stem,
                    ),
                    f"debug CSV voor {video.video_id}/{participant_id}",
                )
                print(f"    {video.video_id}/{participant_id}: debug CSV -> {debug_path}")
            participant = participants_by_id[participant_id]
            for jump in jumps:
                all_parameters.append(
                    calculate_jump_parameters(
                        jump,
                        track,
                        participant,
                        video_filename=video.path.name,
                        recorded_at=recorded_at,
                        pixels_per_meter=pixels_per_meter,
                        pixels_per_meter_implied=implied_by_jump.get(jump.jump_number),
                    )
                )

        # Written after EVERY video (added 2026-08-07), not just once
        # at the very end -- a multi-video run can take hours (see
        # /pose/'s per-frame inference cost), and previously a crash
        # on video N (disk space or anything else) lost every result
        # already computed for videos 1..N-1 too, since they only
        # existed in the in-memory all_parameters list until the loop
        # fully finished. Re-writing the whole (growing) results file
        # after each video is simple and safe -- export_results()
        # always writes the complete all_parameters list, so this
        # file is never missing anything already computed, and never
        # more than one video's worth of work behind if something
        # does still go wrong.
        output_path = _run_with_disk_pause(
            lambda: export_results(all_parameters, args.export_path),
            f"tussentijds resultaat na {video.video_id}",
        )
        print(
            f"    {video.video_id}: tussentijds resultaat opgeslagen "
            f"({len(all_parameters)} sprong(en) totaal tot nu toe) -> "
            f"{output_path}"
        )

    print(f"\n--- /export/: {len(all_parameters)} jump result(s) totaal ---")
    print(f"Resultaten staan in: {args.export_path}")


if __name__ == "__main__":
    main()
