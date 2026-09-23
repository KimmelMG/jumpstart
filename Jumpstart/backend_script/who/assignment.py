"""Interactive athlete order-in-frame assignment (/who/).

Shows a single reference frame to the user and lets them click on
athletes left to right, one at a time. Every athlete needs a HIP
click (used by /tracking/ to seed its matching). On top of that,
pixels-per-metre calibration needs a head+feet click from ONE athlete
per resolution -- see the 2026-07-29 note below for why it's just
one, not every athlete in every video. The result is an ordered list
of participant IDs (left to right) plus the frame that was used --
exactly what /identify/ needs later to couple tracked objects to
profiles.

This is a deliberately low-tech, matplotlib-based interaction. It
does not depend on the eventual front-end decision (FastAPI+HTMX vs.
React vs. Streamlit) -- it only needs a Python environment with a
display, so it can be built and used right now, and swapped for a
web widget later without changing any other module (assign_order
still returns the same FrameAssignment either way).

IMPORTANT (added 2026-07-29): pixels-per-metre calibration replaces
the earlier "reference pixels-per-metre + resolution scaling"
approach in jumpstart/jumps/segmenter.py, which only corrected for
resolution differences and still relied on a manually-guessed
constant never measured against anything real.

Pixels-per-metre is a property of the CAMERA SETUP (distance, zoom,
resolution) at a given depth, not of any one athlete -- a taller
athlete simply covers more pixels at the same real distance, but the
pixels-per-metre CONVERSION FACTOR at that distance is the same for
everyone standing there. So this module only asks for a head+feet
click (measured against that one athlete's own known height) from
the FIRST athlete assigned at a given resolution; the resulting
pixels-per-metre value is cached (by (frame_width_px, frame_height_px)
-- the video's ACTUAL decoded resolution, not a filename label) and
reused automatically for every other athlete at that same resolution,
in this video and in every later video processed in the same run, with
no further head/feet clicks needed. This assumes every athlete jumps
at roughly the same distance from the camera (side-on, same physical
spot) -- true for this project's fixed-camera setup (see
jumpstart/tracking/tracker.py's module docstring) but worth
reconsidering if that ever changes. The cache is NOT persisted
between separate script runs -- see workflow_demo.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from backend_script.profiles.models import Participant

# (frame_width_px, frame_height_px) -> pixels_per_meter, shared across
# every video processed in one run. See the module docstring's
# 2026-07-29 note.
PixelsPerMeterCache = Dict[Tuple[int, int], float]


@dataclass(frozen=True)
class FrameAssignment:
    """Result of assigning athlete order-in-frame for one video.

    Attributes:
        video_id: Identifier of the video this assignment belongs
            to.
        frame_index: Index of the frame that was shown to the user.
        frame_width_px: ACTUAL decoded width (pixels) of the
            reference frame shown to the user -- same value
            /tracking/ reads for this video, used as (part of) the
            pixels-per-metre cache key.
        frame_height_px: ACTUAL decoded height (pixels) -- see
            frame_width_px.
        participant_ids: Participant IDs in left-to-right order, as
            assigned by the user. This is the order /identify/ will
            use to couple tracked objects to profiles.
        click_positions: Pixel (x, y) HIP position of each click, in
            the same order as participant_ids. This is the position
            /tracking/ seeds its matching from.
        head_positions: Pixel (x, y) TOP-OF-HEAD position of each
            athlete, same order as participant_ids, or None where
            this athlete's calibration was skipped because this
            resolution was already calibrated by an earlier athlete
            (in this video or an earlier one) -- see the module
            docstring's 2026-07-29 note. Not used by /tracking/.
        feet_positions: Pixel (x, y) BOTTOM-OF-FEET position, or
            None -- see head_positions.
        pixels_per_meter: The pixels-per-metre that applies to THIS
            video, snapshotted when /who/ for this video was
            finished. None where it was not recorded (CLI route) --
            callers then fall back to
            Session.pixels_per_meter_by_resolution, as before.
    """

    video_id: str
    frame_index: int
    frame_width_px: int
    frame_height_px: int
    participant_ids: List[str]
    click_positions: List[Tuple[float, float]] = field(
        default_factory=list
    )
    head_positions: List[Optional[Tuple[float, float]]] = field(
        default_factory=list
    )
    feet_positions: List[Optional[Tuple[float, float]]] = field(
        default_factory=list
    )
    # Toegevoegd 2026-09-03: de px/m die voor DEZE video geldt.
    # Voorheen zocht de analyse altijd de LAATSTE kalibratie van die
    # resolutie op, waardoor een herkalibratie halverwege met
    # terugwerkende kracht ook al afgeronde video's raakte -- en
    # video's op een andere camera-afstand een px/m kregen die daar
    # 25-30% naast zat. None = niet vastgelegd (CLI-route).
    pixels_per_meter: Optional[float] = None


def assign_athlete_order(
    frame: np.ndarray,
    video_id: str,
    frame_index: int,
    profiles: Sequence[Participant],
    pixels_per_meter_cache: Optional[PixelsPerMeterCache] = None,
) -> FrameAssignment:
    """Let the user click athletes left-to-right and assign profiles.

    Opens an interactive matplotlib window showing ``frame``. Every
    athlete needs a HIP click -- /pose/ detects and tracks the hip
    midpoint specifically (see
    jumpstart.pose.detector.PersonDetection), and /tracking/ matches
    this click to the nearest detection within a fixed pixel
    tolerance (see tracker._greedy_match's max_match_distance), so a
    click far from the actual hip can fail to match and leave that
    athlete stuck at the click position instead of being tracked.
    Right after the hip click, the remaining (not-yet-assigned)
    profiles are printed in the terminal as a numbered list and the
    user types the number of the profile that athlete belongs to.

    On top of the hip click, the FIRST athlete assigned at a
    resolution not yet in pixels_per_meter_cache also needs two more
    clicks -- TOP OF HEAD (kruin), then BOTTOM OF FEET -- used to
    calibrate real pixels-per-metre for that resolution from this one
    athlete's own known height (see the module docstring's 2026-07-29
    note for why one athlete is enough). Every other athlete, in this
    video and in later videos of the same run, only needs the hip
    click once this resolution is cached.

    The user finishes by pressing Enter (with the plot window
    focused, and only between athletes -- not mid-way through a
    pending calibration) once every athlete visible in frame has
    been assigned, or earlier if fewer than all profiles are in this
    particular video (e.g. 8 of 20 team members in frame). Pressing
    'u' undoes the last click: if it's mid-way through a calibration
    (head or feet click still pending), it cancels that athlete and
    returns to the hip-click step; if the last athlete was already
    fully assigned, it undoes that whole athlete instead -- and if
    that athlete's click had written to the calibration cache
    (established a brand-new resolution, or force-recalibrated one --
    see 'c' below), that write is rolled back too: a brand-new
    calibration is removed entirely, a forced recalibration is
    restored to whatever value it overwrote (safe to do because undo
    is last-in-first-out: no later athlete can already depend on a
    calibration that's still eligible to be undone).

    Pressing 'c' (only while waiting for the next hip click, not
    mid-calibration) arms a one-off "force recalibrate" for the NEXT
    athlete clicked: even if this resolution already has a cached
    value (e.g. reused from an earlier video, or from a camera
    position you now suspect changed), that next athlete's hip click
    will be followed by the head/feet clicks again, and the result
    OVERWRITES the cached value for every athlete/video after this
    point. Press 'c' again to disarm it without using it. Use this
    when the automatic reuse doesn't seem right for a particular
    video (e.g. you think the camera moved between sessions even
    though the resolution didn't change).

    Args:
        frame: BGR image array, e.g. from extract_reference_frame.
        video_id: Identifier of the video this frame was taken from.
        frame_index: Index of the frame within that video.
        profiles: All available participant profiles to choose
            from.
        pixels_per_meter_cache: Shared dict of
            (frame_width_px, frame_height_px) -> pixels_per_meter,
            mutated in place as new resolutions get calibrated. Pass
            the same dict across every video in a run (e.g.
            session.pixels_per_meter_by_resolution) so calibration
            carries over between videos. If None, a fresh empty dict
            is used for this call only (no reuse across videos --
            useful for standalone testing).

    Returns:
        A FrameAssignment with participant IDs in the order they
        were clicked (left to right, as intended by the user), plus
        the calibrating athlete's head/feet pixel positions (None
        for every other athlete -- see FrameAssignment).
    """
    if pixels_per_meter_cache is None:
        pixels_per_meter_cache = {}

    frame_height_px, frame_width_px = frame.shape[0], frame.shape[1]
    resolution_key = (frame_width_px, frame_height_px)

    remaining: List[Participant] = list(profiles)
    assigned: List[str] = []
    click_positions: List[Tuple[float, float]] = []
    head_positions: List[Optional[Tuple[float, float]]] = []
    feet_positions: List[Optional[Tuple[float, float]]] = []
    # Parallel to the lists above: None if this athlete reused an
    # existing calibration (or no calibration exists yet and this
    # athlete didn't establish one -- shouldn't happen). Otherwise
    # (resolution_key, previous_value) -- previous_value is None if
    # this athlete established a brand-new resolution's calibration,
    # or the old cached value if this athlete force-recalibrated an
    # already-cached resolution (see 'c'). Undo uses previous_value
    # to either delete the cache entry or restore it.
    established_resolution: List[
        Optional[Tuple[Tuple[int, int], Optional[float]]]
    ] = []

    # Tiny per-athlete state machine: "hip" -- waiting for the next
    # athlete's hip click (and profile choice); "head"/"feet" --
    # waiting for the calibration clicks of the athlete currently
    # in `pending` (entered if this resolution isn't cached yet, or
    # if 'c' armed a forced recalibration).
    state = "hip"
    pending: dict = {}
    # One-off flag armed by pressing 'c' while in "hip" state; forces
    # the NEXT athlete through head/feet clicks even if this
    # resolution is already cached, overwriting the cached value.
    # Consumed (reset to False) as soon as it's used.
    force_recalibrate = False

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(frame[:, :, ::-1])  # BGR -> RGB for correct display
    ax.axis("off")

    def _status_label() -> str:
        if resolution_key in pixels_per_meter_cache:
            calibrated = f"ja ({pixels_per_meter_cache[resolution_key]:.1f} px/m)"
        else:
            calibrated = "nog niet"
        armed = " | HERKALIBRATIE GEWAPEND" if force_recalibrate else ""
        return (
            f"Toegewezen: {len(assigned)} | Resterend: {len(remaining)} | "
            f"Resolutie {frame_width_px}x{frame_height_px} gekalibreerd: "
            f"{calibrated}{armed}"
        )

    def _current_instruction() -> str:
        if state == "hip":
            return (
                "Klik op het MIDDEN VAN DE HEUPEN van de volgende "
                "atleet, van links naar rechts."
            )
        if state == "head":
            return (
                f"Klik op de KRUIN (bovenkant hoofd) van "
                f"{pending.get('participant_id', '')} (kalibratie "
                "voor deze resolutie)."
            )
        return (
            f"Klik op de VOETEN (onderkant) van "
            f"{pending.get('participant_id', '')}."
        )

    def _update_title() -> None:
        ax.set_title(
            _current_instruction() + "\n"
            "'u' = ongedaan maken, 'c' = forceer herkalibratie voor "
            "volgende atleet, 'enter' = klaar (alleen tussen atleten)."
        )
        ax.set_xlabel(_status_label())
        fig.canvas.draw_idle()

    def _prompt_for_profile() -> Optional[Participant]:
        if not remaining:
            print("Geen profielen meer over om toe te wijzen.")
            return None
        print("\nWelk profiel hoort bij dit punt?")
        for idx, participant in enumerate(remaining, start=1):
            print(
                f"  {idx}. {participant.participant_id} - "
                f"{participant.name}"
            )
        print("  0. Annuleer deze klik")
        while True:
            raw = input("Kies nummer: ").strip()
            if raw == "0":
                return None
            if raw.isdigit() and 1 <= int(raw) <= len(remaining):
                return remaining[int(raw) - 1]
            print("Ongeldige keuze, probeer opnieuw.")

    def _draw_marker(
        order_number: int,
        participant_id: str,
        hip_position: Tuple[float, float],
        head_position: Optional[Tuple[float, float]],
        feet_position: Optional[Tuple[float, float]],
    ) -> None:
        ax.plot(*hip_position, "o", color="lime", markersize=10)
        ax.annotate(
            f"{order_number}: {participant_id}",
            hip_position,
            textcoords="offset points",
            xytext=(8, 8),
            color="lime",
            fontsize=9,
            fontweight="bold",
        )
        if head_position is not None and feet_position is not None:
            ax.plot(*head_position, "^", color="cyan", markersize=7)
            ax.plot(*feet_position, "v", color="yellow", markersize=7)
            ax.plot(
                [head_position[0], feet_position[0]],
                [head_position[1], feet_position[1]],
                "--", color="white", linewidth=1,
            )

    def _redraw_all_markers() -> None:
        ax.clear()
        ax.imshow(frame[:, :, ::-1])
        ax.axis("off")
        for i, (pid, hip_pos, head_pos, feet_pos) in enumerate(
            zip(assigned, click_positions, head_positions, feet_positions),
            start=1,
        ):
            _draw_marker(i, pid, hip_pos, head_pos, feet_pos)
        _update_title()

    def _finalize_athlete(
        participant_id: str,
        hip_position: Tuple[float, float],
        head_position: Optional[Tuple[float, float]],
        feet_position: Optional[Tuple[float, float]],
        calibration_write: Optional[Tuple[Tuple[int, int], Optional[float]]],
    ) -> None:
        order_number = len(assigned) + 1
        assigned.append(participant_id)
        click_positions.append(hip_position)
        head_positions.append(head_position)
        feet_positions.append(feet_position)
        established_resolution.append(calibration_write)
        _draw_marker(order_number, participant_id, hip_position, head_position, feet_position)

    def _on_click(event) -> None:
        nonlocal state, pending, force_recalibrate
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        position = (event.xdata, event.ydata)

        if state == "hip":
            if click_positions and position[0] < click_positions[-1][0]:
                print(
                    "Let op: dit punt ligt links van het vorige punt "
                    "-- weet je zeker dat dit klopt qua links-naar-"
                    "rechts volgorde?"
                )
            chosen = _prompt_for_profile()
            if chosen is None:
                return
            remaining.remove(chosen)
            if resolution_key in pixels_per_meter_cache and not force_recalibrate:
                # Already calibrated (by an earlier athlete, in this
                # video or an earlier one) -- no head/feet clicks
                # needed for this athlete.
                ppm = pixels_per_meter_cache[resolution_key]
                print(
                    f"{chosen.participant_id} toegewezen (kalibratie "
                    f"hergebruikt: {ppm:.1f} px/m op "
                    f"{frame_width_px}x{frame_height_px})."
                )
                _finalize_athlete(
                    chosen.participant_id, position, None, None, None
                )
                _update_title()
            else:
                previous_value = pixels_per_meter_cache.get(resolution_key)
                pending = {
                    "participant_id": chosen.participant_id,
                    "hip": position,
                    "previous_value": previous_value,
                }
                force_recalibrate = False  # consumed
                state = "head"
                _update_title()
        elif state == "head":
            pending["head"] = position
            state = "feet"
            _update_title()
        else:  # state == "feet"
            head_position = pending["head"]
            feet_position = position
            pixel_height = pixel_height_for_athlete(head_position, feet_position)
            participant = next(
                p for p in profiles
                if p.participant_id == pending["participant_id"]
            )
            ppm = pixels_per_meter_from_click(
                head_position, feet_position, participant.height_cm
            )
            if ppm is None:
                print(
                    f"Kop/voeten-klik voor {pending['participant_id']} "
                    f"te dicht bij elkaar ({pixel_height:.1f}px) -- "
                    "waarschijnlijk een misklik. Opnieuw beginnen met "
                    "de heup-klik voor deze atleet."
                )
                remaining.append(participant)
                pending = {}
                state = "hip"
                _update_title()
                return
            previous_value = pending["previous_value"]
            pixels_per_meter_cache[resolution_key] = ppm
            if previous_value is None:
                print(
                    f"Resolutie {frame_width_px}x{frame_height_px} "
                    f"gekalibreerd via {pending['participant_id']}: "
                    f"{ppm:.1f} px/m -- andere atleten en video's op "
                    "deze resolutie hoeven niet opnieuw gemeten te "
                    "worden."
                )
            else:
                print(
                    f"Resolutie {frame_width_px}x{frame_height_px} "
                    f"HERkalibreerd via {pending['participant_id']}: "
                    f"{ppm:.1f} px/m (was {previous_value:.1f} px/m) "
                    "-- geldt vanaf nu voor deze resolutie."
                )
            _finalize_athlete(
                pending["participant_id"],
                pending["hip"],
                head_position,
                feet_position,
                (resolution_key, previous_value),
            )
            pending = {}
            state = "hip"
            _update_title()

    def _on_key(event) -> None:
        nonlocal state, pending, force_recalibrate
        if event.key == "u":
            if state != "hip" and pending:
                removed_id = pending["participant_id"]
                removed_participant = next(
                    p for p in profiles if p.participant_id == removed_id
                )
                remaining.append(removed_participant)
                pending = {}
                state = "hip"
                print(
                    f"Bezig met {removed_id} geannuleerd -- begin "
                    "opnieuw met de heup-klik."
                )
                _update_title()
            elif assigned:
                removed_id = assigned.pop()
                click_positions.pop()
                head_positions.pop()
                feet_positions.pop()
                removed_write = established_resolution.pop()
                if removed_write is not None:
                    (removed_res, previous_value) = removed_write
                    if previous_value is None:
                        pixels_per_meter_cache.pop(removed_res, None)
                        print(
                            f"Kalibratie voor resolutie "
                            f"{removed_res[0]}x{removed_res[1]} ook "
                            "verwijderd (was net vastgelegd door deze "
                            "atleet)."
                        )
                    else:
                        pixels_per_meter_cache[removed_res] = previous_value
                        print(
                            f"Kalibratie voor resolutie "
                            f"{removed_res[0]}x{removed_res[1]} "
                            f"teruggezet naar {previous_value:.1f} px/m "
                            "(herkalibratie door deze atleet ongedaan "
                            "gemaakt)."
                        )
                removed_participant = next(
                    p for p in profiles if p.participant_id == removed_id
                )
                remaining.append(removed_participant)
                print(f"Laatste toewijzing ({removed_id}) ongedaan gemaakt.")
                _redraw_all_markers()
        elif event.key == "c":
            if state != "hip":
                print(
                    "Nog niet klaar met de huidige atleet -- 'c' "
                    "genegeerd."
                )
                return
            force_recalibrate = not force_recalibrate
            if force_recalibrate:
                current = pixels_per_meter_cache.get(resolution_key)
                current_text = (
                    f"{current:.1f} px/m" if current is not None
                    else "nog niet gekalibreerd"
                )
                print(
                    f"Herkalibratie gewapend voor resolutie "
                    f"{frame_width_px}x{frame_height_px} (huidige "
                    f"waarde: {current_text}) -- de volgende atleet "
                    "krijgt de kop/voeten-klik, ongeacht cache. Druk "
                    "'c' nogmaals om te annuleren."
                )
            else:
                print("Herkalibratie geannuleerd.")
            _update_title()
        elif event.key == "enter":
            if state != "hip":
                print(
                    "Nog niet klaar met de huidige atleet (kruin/"
                    "voeten-klik nog nodig) -- 'enter' genegeerd."
                )
                return
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", _on_click)
    fig.canvas.mpl_connect("key_press_event", _on_key)
    _update_title()
    plt.show()

    return FrameAssignment(
        video_id=video_id,
        frame_index=frame_index,
        frame_width_px=frame_width_px,
        frame_height_px=frame_height_px,
        participant_ids=assigned,
        click_positions=click_positions,
        head_positions=head_positions,
        feet_positions=feet_positions,
    )


def pixel_height_for_athlete(
    head_position: Tuple[float, float],
    feet_position: Tuple[float, float],
) -> float:
    """Return the pixel distance between a head click and a feet click."""
    (head_x, head_y) = head_position
    (feet_x, feet_y) = feet_position
    return float(np.hypot(feet_x - head_x, feet_y - head_y))


def pixels_per_meter_from_click(
    head_position: Tuple[float, float],
    feet_position: Tuple[float, float],
    height_cm: float,
) -> Optional[float]:
    """Return real pixels-per-metre from one head/feet click + height.

    Args:
        head_position: Pixel (x, y) of the top-of-head click.
        feet_position: Pixel (x, y) of the bottom-of-feet click.
        height_cm: This athlete's known standing height, in cm.

    Returns:
        pixel_height / (height_cm / 100), or None if the head/feet
        pixel distance is degenerate (<=1px, e.g. a misclick landed
        on the same spot twice) -- callers should treat None as "this
        click could not calibrate anything" and ask the user to
        retry, rather than caching a nonsensical value.
    """
    pixel_height = pixel_height_for_athlete(head_position, feet_position)
    if pixel_height <= 1.0:
        return None
    return pixel_height / (height_cm / 100.0)
