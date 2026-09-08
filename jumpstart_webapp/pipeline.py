"""Runs the /pose/ -> /export/ analysis in a background thread.

Mirrors jumpstart/workflow_demo.py's main() analysis loop (the part
below its "REVIEWED BOUNDARY") as closely as possible, calling the
exact same jumpstart.* functions the CLI calls, so a run started from
this web front-end produces identical results to the equivalent CLI
command. The only intentional differences:

  - Progress goes to Job.log_lines (polled by the browser) instead of
    stdout.
  - The CLI's low-disk-space pause (workflow_demo._wait_for_disk_space)
    blocks on input(), which has no meaning in a background thread with
    no terminal attached -- here a low-disk-space warning is logged
    instead of pausing. If this matters for your setup (e.g. OneDrive-
    backed video folders), keep an eye on the log and free up space
    proactively.
  - The other fine-tuning flags (tolerance_m_s2, smoothing_window_s,
    min_flight_duration_s, max_match_speed_m_s, num_poses) are not
    exposed in the web UI (kept at the same defaults workflow_demo.py
    uses) -- use the CLI directly if you need to tune those.
  - debug-CSV export and the MediaPipe model variant (lite/full) ARE
    exposed (added 2026-08-14) -- see analysis_start's debug_csv and
    mediapipe_variant form fields in app.py. Debug CSVs, when enabled,
    are written to jumpstart_webapp/_debug_csv/ (see state.py's
    DEBUG_CSV_DIR) using the exact same _write_debug_csv function the
    CLI uses (imported directly from workflow_demo.py rather than
    duplicated here), so the CSV format is identical either way.
  - Adaptive frame-skip (added 2026-09-03, ported from the
    frame-skip-experiment-2026-08-21 dated copy into the real
    jumpstart/tracking/tracker.py and jumpstart/identify/couple.py)
    IS exposed -- see analysis_start's frame_skip checkbox in app.py,
    which maps to run_analysis's adaptive_skip_frames/
    adaptive_dense_duration_s below. Fixed at 3 skipped frames / 4.5s
    dense duration per Mathijs' own tested values (2026-08-26 manual-
    analysis comparison) rather than exposed as separate tunable form
    fields. Requires a pixels-per-meter calibration for a video's
    resolution to work -- falls back to normal per-frame detection
    (with a printed warning, not shown in the web UI's log panel
    since it goes to the server's own stdout like the rest of
    /tracking/'s progress output) if no calibration exists yet.
"""

from __future__ import annotations

import shutil
import traceback
from pathlib import Path
from typing import Optional

from jumpstart.calc import (
    calculate_jump_parameters,
    check_calibration,
    implied_pixels_per_meter_for_jump,
)
from jumpstart.export import export_results
from jumpstart.identify import identify_athletes_in_video
from jumpstart.io import get_recording_datetime
from jumpstart.jumps import build_jumps_for_athlete
from jumpstart.pose import (
    create_pose_landmarker,
    create_yolo_pose_model,
    detect_people,
    detect_people_yolo,
)
from jumpstart.workflow_demo import _write_debug_csv

from jumpstart_webapp.state import DETECTIONS_CACHE_DIR, Job

# Same defaults as jumpstart/workflow_demo.py's parse_args().
DEFAULTS = dict(
    max_match_speed_m_s=25.0,
    tolerance_m_s2=3.0,
    smoothing_window_s=7 / 120,
    min_flight_duration_s=0.13,
    # Stage 2 of the two-stage jump-edge detection (added 2026-09-03,
    # see jumpstart/events/detector.py's docstring). On by default:
    # removes a measured -0.108m jump-height bias and lifts CCC
    # against the manual reference from 0.21 to 0.78.
    refine_edges=True,
    num_poses=10,
    disk_space_margin_gb=2.0,
)

# Same URL-pattern trick noted in jumpstart/pose/detector.py's own
# docstring: MediaPipe's lite/full/heavy pose landmarker variants share
# the same download URL, differing only in this one path segment.
MEDIAPIPE_FULL_MODEL_PATH = Path.home() / ".jumpstart" / "models" / "pose_landmarker_full.task"
MEDIAPIPE_FULL_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)


def _check_disk_space(job: Job, video_path: Path, min_extra_bytes: int) -> None:
    """Non-blocking version of workflow_demo._wait_for_disk_space.

    The CLI pauses with input() when space is tight; a background
    thread has no terminal to read from, so this only logs a warning
    and continues -- see this module's docstring.
    """
    try:
        needed_bytes = video_path.stat().st_size + min_extra_bytes
        free_bytes = shutil.disk_usage(str(video_path.parent)).free
        if free_bytes < needed_bytes:
            job.log(
                f"    WAARSCHUWING: mogelijk te weinig vrije schijfruimte "
                f"voor {video_path.name} ({free_bytes / 1024**3:.1f} GB "
                f"vrij, ~{needed_bytes / 1024**3:.1f} GB nodig). Het "
                "script gaat toch door -- maak zo nodig ruimte vrij."
            )
    except OSError:
        pass

def _detections_cache_path_for(
    video_path: Path,
    pose_backend: str,
    yolo_model_name: str,
    mediapipe_variant: str,
    adaptive_skip_frames: Optional[int],
) -> Path:
    """Build the cache filename for one video + pose/frame-skip setting.

    All the settings that change what /pose/+/tracking/ would actually
    produce are baked into the filename (added 2026-09-04) so a run
    with a DIFFERENT setting than an earlier cached run just misses
    the cache and detects fresh, rather than silently reusing
    detections from an incompatible configuration.

    FIXED 2026-09-08: this used to key on video.video_id ("V001",
    "V002", ...), which is only a video's sequential position WITHIN
    one single run (see jumpstart/io/video_input.py's
    register_videos), not a stable per-video identifier. Two runs
    with videos in a different order silently read/wrote each other's
    cache files under the same name -- exactly the same bug already
    fixed in the CLI (workflow_demo.py) on 2026-08-17, which this
    route was not updated to match at the time. Confirmed to have
    caused a failed run on 2026-09-08 (wrong cached detections loaded
    for 4 videos, a crash on a 5th). Now keyed on video_path.stem (the
    filename without extension, e.g.
    "20260611_S02_ipadH_C02_240fps_1080p"), which is unique and stable
    regardless of video order or which videos are included in a run.
    """
    if pose_backend == "yolo":
        backend_key = f"yolo-{yolo_model_name}"
    else:
        backend_key = f"mediapipe-{mediapipe_variant}"
    skip_key = f"skip{adaptive_skip_frames}" if adaptive_skip_frames else "noskip"
    safe_backend_key = backend_key.replace("/", "_")
    return DETECTIONS_CACHE_DIR / f"{video_path.stem}__{safe_backend_key}__{skip_key}.pkl"

def run_analysis(
    job: Job,
    pose_backend: str,
    export_path: Path,
    yolo_model_name: str,
    mediapipe_variant: str = "lite",
    debug_csv_dir: Optional[Path] = None,
    adaptive_skip_frames: Optional[int] = None,
    adaptive_dense_duration_s: float = 4.5,
    use_cache: bool = False,
) -> None:
    """Background-thread entry point: run /pose/ through /export/.

    Args:
        job: The global Job (see state.py) -- session, who_assignments
            and participants must already be filled in.
        pose_backend: "mediapipe" or "yolo".
        export_path: Where to write the results spreadsheet.
        yolo_model_name: Only used when pose_backend == "yolo".
        mediapipe_variant: "lite" (default, tested) or "full" (heavier,
            more accurate per MediaPipe's own claims -- NOT confirmed
            to matter on this project's footage, see
            jumpstart/pose/detector.py's module docstring). Only used
            when pose_backend == "mediapipe".
        debug_csv_dir: If given, write one CSV per athlete/video here
            (same format/columns as the CLI's --debug-csv-dir, via
            workflow_demo._write_debug_csv). None (default) writes no
            CSVs, same as before this option existed.
        adaptive_skip_frames: Passed straight through to
            identify_athletes_in_video/track_all_athletes (added
            2026-09-03, see jumpstart/tracking/tracker.py). None
            (default) disables frame-skipping entirely -- every frame
            is measured, identical to before this option existed. 3
            (the value analysis_start's frame_skip checkbox sends when
            checked) means 1 in every 4 frames is measured while
            "calm", switching to full-density measurement once a fast
            movement is detected near a known athlete.
        adaptive_dense_duration_s: How long (seconds) to measure every
            frame after a frame-skip trigger fires, before returning
            to sparse sampling. Only meaningful when
            adaptive_skip_frames is set. Default 4.5s matches the
            value used in the 2026-08-26 manual-analysis comparison
            run.
        use_cache: If True (added 2026-09-04, analysis_start's
            use_cache checkbox), each video's raw pose detections are
            loaded from -- and, if missing, saved to -- a pickle file
            under DETECTIONS_CACHE_DIR, keyed by video id, pose
            backend/model and frame-skip setting (see
            _detections_cache_path_for below). A later run with an
            identical setting for the same video reuses the cached
            detections and skips /pose/ entirely; a changed setting
            just misses the cache. False (default) never reads or
            writes a cache file, identical to before this option
            existed.
    """
    try:
        session = job.session
        assert session is not None
        participants_by_id = {p.participant_id: p for p in session.participants}

        if adaptive_skip_frames:
            job.log(
                f"Frame-skip actief: {adaptive_skip_frames} frame(s) overslaan, "
                f"{adaptive_dense_duration_s}s dense meten na een snelheidstrigger "
                "(vereist een pixels-per-meter-kalibratie per videoresolutie -- "
                "zonder kalibratie meet een video gewoon elk frame, zonder "
                "melding in dit logpaneel)."
            )
        if use_cache:
            job.log(
                "Detectie-cache actief: per video worden ruwe /pose/-detecties "
                f"opgeslagen in {DETECTIONS_CACHE_DIR} (en hergebruikt als een "
                "eerdere cache voor dezelfde video + hetzelfde pose-model + "
                "dezelfde frame-skip-instelling bestaat -- een andere "
                "instelling mist de cache gewoon en detecteert opnieuw)."
            )
        if pose_backend == "yolo":
            job.log(
                "\n--- /pose/: laden YOLO-pose model "
                f"({yolo_model_name}) -- NOG NIET GEVALIDEERD tegen "
                "echte beelden, zie jumpstart/pose/detector_yolo.py ---"
            )
            yolo_model = create_yolo_pose_model(model_name=yolo_model_name)

            def detect_fn(frame_bgr, _model=yolo_model):
                # imgsz=480 (vs Ultralytics' own default 640) trades a little
                # small/far-away-person accuracy for noticeably faster CPU
                # inference -- see jumpstart/pose/detector_yolo.py's
                # detect_people_yolo docstring. NEEDS REVIEW: not yet
                # checked against real footage.
                return detect_people_yolo(_model, frame_bgr, conf_threshold=0.25, imgsz=480)

        elif mediapipe_variant == "full":
            job.log(
                "\n--- /pose/: laden MediaPipe pose landmarker "
                "(full-variant, nauwkeuriger maar trager dan lite) ---"
            )
            landmarker = create_pose_landmarker(
                MEDIAPIPE_FULL_MODEL_PATH,
                num_poses=DEFAULTS["num_poses"],
                model_url=MEDIAPIPE_FULL_MODEL_URL,
            )

            def detect_fn(frame_bgr, _landmarker=landmarker):
                return detect_people(_landmarker, frame_bgr[:, :, ::-1])

        else:
            job.log("\n--- /pose/: laden MediaPipe pose landmarker (lite, standaard) ---")
            landmarker = create_pose_landmarker(None, num_poses=DEFAULTS["num_poses"])

            def detect_fn(frame_bgr, _landmarker=landmarker):
                return detect_people(_landmarker, frame_bgr[:, :, ::-1])

        all_parameters = []
        for video in session.videos:
            assignment = session.who_assignments.get(video.video_id)
            if assignment is None:
                job.log(f"Overslaan {video.video_id} ({video.path.name}): geen /who/-toewijzing.")
                continue

            job.log(
                f"\n--- {video.video_id} ({video.path.name}): /tracking/ + "
                f"/identify/ -- {len(assignment.participant_ids)} atleet(en) "
                "volgen door de hele video ---"
            )
            _check_disk_space(
                job, video.path,
                min_extra_bytes=int(DEFAULTS["disk_space_margin_gb"] * 1024 ** 3),
            )
            # Gewijzigd 2026-09-03: neem de px/m die bij DEZE video is
            # vastgelegd. Alleen als die ontbreekt (CLI-route)
            # terugvallen op de gedeelde resolutie-cache, zoals eerder.
            pixels_per_meter_for_tracking = (
                assignment.pixels_per_meter
                if assignment.pixels_per_meter is not None
                else session.pixels_per_meter_by_resolution.get(
                    (assignment.frame_width_px, assignment.frame_height_px)
                )
            )
            if pixels_per_meter_for_tracking is not None:
                job.log(
                    f"    {video.video_id}: kalibratie "
                    f"{pixels_per_meter_for_tracking:.1f} px/m"
                )
            detections_cache_path = (
                _detections_cache_path_for(
                    video.path,
                    pose_backend,
                    yolo_model_name,
                    mediapipe_variant,
                    adaptive_skip_frames,
                )
                if use_cache
                else None
            )
            tracks = identify_athletes_in_video(
                video_path=video.path,
                detect_fn=detect_fn,
                assignment=assignment,
                max_match_speed_m_s=DEFAULTS["max_match_speed_m_s"],
                pixels_per_meter=pixels_per_meter_for_tracking,
                detections_cache_path=detections_cache_path,
                adaptive_skip_frames=adaptive_skip_frames,
                adaptive_dense_duration_s=adaptive_dense_duration_s,
            )

            recorded_at = get_recording_datetime(video.path)
            if recorded_at is None:
                job.log(
                    f"    {video.video_id}: kon geen opnamedatum/-tijd "
                    "lezen -- recorded_at blijft leeg voor deze video."
                )

            for participant_id, track in tracks.items():
                # Gewijzigd 2026-09-03: zelfde bron als hierboven.
                pixels_per_meter = (
                    assignment.pixels_per_meter
                    if assignment.pixels_per_meter is not None
                    else session.pixels_per_meter_by_resolution.get(
                        (track.frame_width_px, track.frame_height_px)
                    )
                )
                if pixels_per_meter is None:
                    job.log(
                        f"    {video.video_id}/{participant_id}: nog geen "
                        f"kalibratie voor resolutie "
                        f"{track.frame_width_px}x{track.frame_height_px} -- "
                        "sprongdetectie overgeslagen (0 sprongen)."
                    )
                    continue

                job.log(f"    {video.video_id}/{participant_id}: /events/ + /jumps/ -- sprongen zoeken...")
                jumps = build_jumps_for_athlete(
                    track,
                    pixels_per_meter=pixels_per_meter,
                    tolerance_m_s2=DEFAULTS["tolerance_m_s2"],
                    smoothing_window_s=DEFAULTS["smoothing_window_s"],
                    min_flight_duration_s=DEFAULTS["min_flight_duration_s"],
                    refine_edges=DEFAULTS["refine_edges"],
                )
                job.log(f"    {video.video_id}/{participant_id}: {len(jumps)} sprong(en) gevonden, /calc/...")
                # Toegevoegd 2026-09-03: vangnet op de kalibratie --
                # reken uit de vluchtfase zelf terug welke px/m eruit
                # zou moeten komen en waarschuw bij een grote afwijking.
                # Zie jumpstart/calc/calibration_check.py.
                implied_by_jump = {
                    jump.jump_number: implied_pixels_per_meter_for_jump(track, jump)
                    for jump in jumps
                }
                calibration = check_calibration(track, jumps, pixels_per_meter)
                if not calibration.within_tolerance:
                    job.log(
                        f"    WAARSCHUWING {video.video_id}/{participant_id}: "
                        f"gebruikte kalibratie {calibration.used_pixels_per_meter:.1f} px/m, "
                        f"maar de valbeweging van {calibration.jumps_used} sprong(en) "
                        f"impliceert {calibration.implied_pixels_per_meter:.1f} px/m "
                        f"({(calibration.ratio - 1) * 100:+.0f}%). Controleer de "
                        "kop+voeten-klik voor deze video -- klopt de camera-afstand nog?"
                    )
                if debug_csv_dir is not None:
                    # Own try/except (rather than letting a write failure
                    # abort the whole run via the outer try/except) -- one
                    # bad CSV write (e.g. disk full) shouldn't lose an
                    # otherwise-successful analysis run.
                    try:
                        debug_path = _write_debug_csv(
                            debug_csv_dir,
                            track,
                            pixels_per_meter=pixels_per_meter,
                            tolerance_m_s2=DEFAULTS["tolerance_m_s2"],
                            smoothing_window_s=DEFAULTS["smoothing_window_s"],
                            jumps=jumps,
                            video_name=video.path.name,
                        )
                        job.log(f"    {video.video_id}/{participant_id}: debug CSV -> {debug_path}")
                    except OSError as exc:
                        job.log(
                            f"    WAARSCHUWING: debug CSV voor {video.video_id}/"
                            f"{participant_id} kon niet worden weggeschreven: {exc}"
                        )
                participant = participants_by_id[participant_id]
                for jump in jumps:
                    all_parameters.append(
                        calculate_jump_parameters(
                            jump, track, participant,
                            video_filename=video.path.name,
                            recorded_at=recorded_at,
                            pixels_per_meter=pixels_per_meter,
                            pixels_per_meter_implied=implied_by_jump.get(jump.jump_number),
                        )
                    )
                    with job.lock:
                        job.per_participant_counts[participant_id] = (
                            job.per_participant_counts.get(participant_id, 0) + 1
                        )

            output_path = export_results(all_parameters, export_path)
            job.log(
                f"    {video.video_id}: tussentijds resultaat opgeslagen "
                f"({len(all_parameters)} sprong(en) totaal tot nu toe) -> {output_path.name}"
            )
            with job.lock:
                job.export_path = output_path
                job.result_count = len(all_parameters)

        job.log(f"\n--- /export/: {len(all_parameters)} sprong(en) totaal ---")
        job.log(f"Resultaten staan in: {export_path}")
        with job.lock:
            job.status = "done"

    except Exception as exc:  # noqa: BLE001 -- surface any failure to the UI
        job.log(f"\nFOUT: {exc}")
        job.log(traceback.format_exc())
        with job.lock:
            job.status = "error"
            job.error_message = str(exc)
