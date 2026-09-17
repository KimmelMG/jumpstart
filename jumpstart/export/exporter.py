"""Result export for /export/.

Collects every computed JumpParameters record into one flat results
table and writes it to disk. Deliberately simple: no figures, no
per-condition statistics (Bland-Altman, ICC, CV%, RMSE) -- those
belong with the validation-analysis work already tracked separately
in Jumpstart_data_entry_v2.xlsx, not in this script.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from jumpstart.calc.parameters import JumpParameters
from jumpstart.events import GRAVITY_M_S2, compute_kinematics


def parameters_to_dataframe(
    all_parameters: List[JumpParameters],
) -> pd.DataFrame:
    """Collect JumpParameters records into a flat results table."""
    return pd.DataFrame([vars(p) for p in all_parameters])


def _excel_safe(table: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of table with timezone-aware datetime columns
    made naive, so it can be written with .to_excel().

    recorded_at (see jumpstart.io.video_metadata.get_recording_datetime)
    is timezone-aware (UTC) by construction, but Excel's datetime type
    has no timezone concept -- pandas raises ValueError otherwise.
    The CSV path doesn't need this: to_csv() just writes the ISO
    string, tzinfo and all.
    """
    table = table.copy()
    for column in table.columns:
        if pd.api.types.is_datetime64tz_dtype(table[column]):
            table[column] = table[column].dt.tz_localize(None)
    return table


def export_results(
    all_parameters: List[JumpParameters], output_path: Path
) -> Path:
    """Export all computed jump parameters to a spreadsheet.

    Args:
        all_parameters: One JumpParameters per detected jump, across
            every athlete and video in the session.
        output_path: Where to write the results (.xlsx or .csv).

    Returns:
        The path results were written to.
    """
    output_path = Path(output_path)
    table = parameters_to_dataframe(all_parameters)
    if output_path.suffix.lower() == ".csv":
        table.to_csv(output_path, index=False)
    else:
        _excel_safe(table).to_excel(output_path, index=False)
    return output_path

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