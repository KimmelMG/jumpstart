"""FastAPI + HTMX web front-end for the Jumpstart pipeline.

Run locally (same machine that has the `jumpstart` package and its
dependencies installed -- see the Stappenplan):

    cd "...\\Phase 3 - Script development 13-7"
    .\\venv\\Scripts\\Activate.ps1
    pip install fastapi "uvicorn[standard]" jinja2 python-multipart
    uvicorn jumpstart_webapp.app:app --reload

Then open http://127.0.0.1:8000 in a browser -- full-screen it (F11)
on whatever screen/monitor is connected (HDMI or otherwise) when
presenting.

This module only orchestrates HTTP requests -> the same jumpstart.*
functions the CLI (jumpstart/workflow_demo.py) calls. It does not
reimplement any analysis logic.

UPDATE 2026-09-04 (sessie-isolatie + FIFO-wachtrij, TODO.md item 12):
elke browser/sessie krijgt nu zijn EIGEN Job (deelnemers, video's,
/who/-voortgang, analysestatus/log/resultaat) i.p.v. één gedeeld
globaal `JOB`-object -- zie state.py's `JobManager`/`get_current_job`
hieronder. De zware pose-analysestap blijft bewust sequentieel: alle
sessies delen één FIFO-wachtrij (`state.ANALYSIS_QUEUE`) zodat er
nooit twee analyses tegelijk om dezelfde CPU/GPU concurreren,
ongeacht hoeveel mensen tegelijk de webapp gebruiken.
"""

from __future__ import annotations

import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from jumpstart.io import discover_videos_in_folder, register_videos
from jumpstart.profiles import build_profiles, load_participant_table
from jumpstart.profiles.manager import create_participant_template
from jumpstart.session import Session
from jumpstart.who.assignment import (
    FrameAssignment,
    pixel_height_for_athlete,
    pixels_per_meter_from_click,
)
from jumpstart.who.frame_picker import extract_reference_frame

from jumpstart_webapp import pipeline
from jumpstart_webapp.state import ANALYSIS_QUEUE, DOWNLOADS_DIR, JOB_MANAGER, Job, WhoClickState

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Jumpstart")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
templates.env.filters["id"] = id  # cache-busting for the reference-frame <img> tag
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ---------------------------------------------------------------------------
# Sessie-isolatie (2026-09-04): elke browser krijgt via een cookie een
# eigen Job. Zie state.py's JobManager voor de details.
# ---------------------------------------------------------------------------

SESSION_COOKIE_NAME = "jumpstart_session"
SESSION_COOKIE_MAX_AGE_S = 60 * 60 * 24 * 7  # 7 dagen


@app.middleware("http")
async def ensure_session_cookie(request: Request, call_next):
    """Resolve (of maak) de sessie-cookie vóór de route draait, en zet 'm
    op de ECHTE uitgaande response na afloop.

    BUGFIX 2026-09-08: de vorige aanpak (een `response: Response`
    dependency-parameter waarop `set_cookie` werd aangeroepen, zie de
    oude `get_current_job` hieronder) werkt NIET betrouwbaar zodra een
    route zelf een eigen Response-object teruggeeft -- en dat doet hier
    elke route (`templates.TemplateResponse(...)`). FastAPI voegt de
    headers van een dependency-`Response` dan NIET automatisch samen met
    de response die de route teruggeeft. Gevolg: de cookie werd in de
    praktijk nooit naar de browser gestuurd, dus kreeg elke request een
    NIEUWE, lege sessie (`job.session is None`) -- precies het
    "/who/start geeft altijd 400 Bad Request"-symptoom dat hiermee werd
    waargenomen. Een middleware werkt wel altijd: die past de ECHTE
    uitgaande response aan, ongeacht wat de route teruggeeft.
    """
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    is_new = session_id is None
    if is_new:
        session_id = uuid.uuid4().hex
    request.state.session_id = session_id

    response = await call_next(request)

    if is_new:
        response.set_cookie(
            SESSION_COOKIE_NAME,
            session_id,
            max_age=SESSION_COOKIE_MAX_AGE_S,
            httponly=True,
            samesite="lax",
        )
    return response


def get_current_job(request: Request) -> Job:
    """FastAPI dependency: geeft de Job van de huidige sessie terug.

    De cookie zelf wordt door de `ensure_session_cookie`-middleware
    hierboven gelezen/aangemaakt en op `request.state.session_id` gezet
    (en, bij een nieuwe sessie, op de uitgaande response geplakt) --
    deze dependency hoeft alleen nog de bijbehorende Job op te zoeken.
    """
    return JOB_MANAGER.get_or_create(request.state.session_id)


# ---------------------------------------------------------------------------
# Page shell
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request, job: Job = Depends(get_current_job)) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/template")
def download_template() -> FileResponse:
    path = create_participant_template(DOWNLOADS_DIR / "participant_template.xlsx")
    return FileResponse(path, filename="participant_template.xlsx")


# ---------------------------------------------------------------------------
# Step 1: participants + videos
# ---------------------------------------------------------------------------

@app.post("/setup", response_class=HTMLResponse)
async def setup(
    request: Request,
    participants_file: UploadFile,
    video_mode: str = Form(...),
    video_folder: str = Form(""),
    video_paths_text: str = Form(""),
    recursive: Optional[str] = Form(None),
    video_files: List[UploadFile] = File(default=[]),
    job: Job = Depends(get_current_job),
) -> HTMLResponse:
    job.reset()
    try:
        participants_path = job.uploads_dir / participants_file.filename
        with open(participants_path, "wb") as fh:
            shutil.copyfileobj(participants_file.file, fh)

        table = load_participant_table(participants_path)
        participants = build_profiles(table)
        if not participants:
            raise ValueError("Het deelnemersbestand bevat geen rijen.")

        if video_mode == "folder":
            folder = Path(video_folder.strip().strip('"'))
            if not folder.is_dir():
                raise ValueError(f"Videomap niet gevonden: {folder}")
            video_paths = discover_videos_in_folder(folder, recursive=bool(recursive))
            if not video_paths:
                raise ValueError(f"Geen ondersteunde video's gevonden in {folder}")
        elif video_mode == "upload":
            # Videos uploaded straight from the browser (e.g. from a phone) --
            # save each to this session's own uploads dir first, then hand
            # the resulting local paths to register_videos exactly like the
            # other two modes do.
            video_paths = []
            for upload in video_files:
                if not upload.filename:
                    continue
                dest = job.uploads_dir / upload.filename
                with open(dest, "wb") as fh:
                    shutil.copyfileobj(upload.file, fh)
                video_paths.append(dest)
            if not video_paths:
                raise ValueError("Upload minstens één video.")
        else:
            raw_lines = [line.strip().strip('"') for line in video_paths_text.splitlines()]
            video_paths = [Path(line) for line in raw_lines if line]
            if not video_paths:
                raise ValueError("Geef minstens één videopad op.")

        session = Session()
        session.participants = participants
        session.videos = register_videos(video_paths, session.session_id)

        job.participants = participants
        job.session = session
        job.videos_needing_who = list(session.videos)
    except Exception as exc:  # noqa: BLE001
        job.setup_error = str(exc)
        return templates.TemplateResponse(
            request, "partials/videos_panel.html", {"job": job}
        )

    return templates.TemplateResponse(request, "partials/videos_panel.html", {"job": job})


# ---------------------------------------------------------------------------
# Step 2: /who/ -- athlete order-in-frame, done via clicks in the browser
# ---------------------------------------------------------------------------

def _who_context(job: Job) -> dict:
    markers = []
    who = job.who
    if who is not None:
        for pid, hip, head, feet in zip(
            who.assigned_ids, who.hip_positions, who.head_positions, who.feet_positions
        ):
            markers.append({"pid": pid, "hip": hip, "head": head, "feet": feet})
    return {"job": job, "who": who, "markers": markers}


def _load_video_for_who(job: Job, video, is_redo: bool = False) -> None:
    """Build a fresh WhoClickState for the given video and make it current."""
    frame, frame_index = extract_reference_frame(video.path, fraction=0.5)
    frame_path = job.frames_dir / f"{video.video_id}.jpg"
    import cv2

    cv2.imwrite(str(frame_path), frame)
    resolution_key = (frame.shape[1], frame.shape[0])
    # Welke px/m geldt voor deze video? Bij het opnieuw doen van een al
    # afgeronde video: wat er toen voor DIE video is vastgelegd. Anders:
    # de laatste kalibratie voor deze resolutie (doorgeven).
    previous = job.session.who_assignments.get(video.video_id)
    if is_redo and previous is not None and previous.pixels_per_meter is not None:
        ppm = previous.pixels_per_meter
        ppm_source = job.calibration_source_by_video.get(video.video_id)
    else:
        ppm = job.session.pixels_per_meter_by_resolution.get(resolution_key)
        ppm_source = job.calibration_source_by_resolution.get(resolution_key)
    job.who = WhoClickState(
        video=video,
        frame_path=frame_path,
        frame_width_px=frame.shape[1],
        frame_height_px=frame.shape[0],
        remaining=list(job.session.participants),
        pixels_per_meter=ppm,
        pixels_per_meter_source=ppm_source,
        is_redo=is_redo,
    )


def _advance_to_next_video(job: Job) -> None:
    """Pop the next video needing /who/ and extract its reference frame."""
    if not job.videos_needing_who:
        job.who = None
        return
    video = job.videos_needing_who.pop(0)
    _load_video_for_who(job, video)


@app.get("/who/start", response_class=HTMLResponse)
def who_start(request: Request, job: Job = Depends(get_current_job)) -> HTMLResponse:
    if job.session is None:
        return HTMLResponse("Rond eerst stap 1 af.", status_code=400)
    _advance_to_next_video(job)
    return templates.TemplateResponse(request, "partials/who_panel.html", _who_context(job))


@app.get("/frames/{video_id}.jpg")
def frame_image(video_id: str, job: Job = Depends(get_current_job)) -> FileResponse:
    path = job.frames_dir / f"{video_id}.jpg"
    return FileResponse(path)


@app.post("/who/click", response_class=HTMLResponse)
async def who_click(request: Request, job: Job = Depends(get_current_job)) -> HTMLResponse:
    body = await request.json()
    who = job.who
    if who is None:
        return templates.TemplateResponse(request, "partials/who_panel.html", _who_context(job))

    x, y = float(body["x"]), float(body["y"])
    resolution_key = (who.frame_width_px, who.frame_height_px)
    cache = job.session.pixels_per_meter_by_resolution

    if who.step == "hip":
        participant_id = body.get("participant_id")
        chosen = next((p for p in who.remaining if p.participant_id == participant_id), None)
        if chosen is None:
            who.message = "Kies eerst een profiel voordat je klikt."
            return templates.TemplateResponse(request, "partials/who_panel.html", _who_context(job))
        who.remaining = [p for p in who.remaining if p.participant_id != participant_id]
        force_recalibrate = bool(body.get("force_recalibrate"))
        # Gewijzigd 2026-09-03: kijk naar de px/m die voor DEZE video
        # geldt, niet rechtstreeks naar de gedeelde resolutie-cache.
        if who.pixels_per_meter is not None and not force_recalibrate:
            ppm = who.pixels_per_meter
            who.assigned_ids.append(chosen.participant_id)
            who.hip_positions.append((x, y))
            who.head_positions.append(None)
            who.feet_positions.append(None)
            who.calibration_writes.append(None)
            who.message = (
                f"{chosen.participant_id} toegewezen (kalibratie hergebruikt: {ppm:.1f} px/m)."
            )
        else:
            who.pending_participant_id = chosen.participant_id
            who.pending_hip = (x, y)
            who.pending_previous_ppm = who.pixels_per_meter
            who.step = "head"
            who.message = f"Klik nu op de KRUIN (bovenkant hoofd) van {chosen.participant_id}."
    elif who.step == "head":
        who.pending_head = (x, y)
        who.step = "feet"
        who.message = f"Klik nu op de VOETEN van {who.pending_participant_id}."
    else:  # feet
        participant = next(
            p for p in job.session.participants if p.participant_id == who.pending_participant_id
        )
        ppm = pixels_per_meter_from_click(who.pending_head, (x, y), participant.height_cm)
        if ppm is None:
            who.message = (
                "Kop/voeten-klik te dicht bij elkaar (misklik) -- begin opnieuw met de heup-klik "
                f"voor {who.pending_participant_id}."
            )
            who.remaining.append(participant)
            who.step = "hip"
            who.pending_participant_id = None
            who.pending_head = None
        else:
            # Gewijzigd 2026-09-03: de kalibratie geldt altijd voor DEZE
            # video. Alleen als dit geen herhaling is van een al
            # afgeronde video ("Vorige video"), wordt hij doorgegeven
            # aan de nog niet gedane video's van deze resolutie.
            who.pixels_per_meter = ppm
            who.pixels_per_meter_source = who.video.video_id
            if not who.is_redo:
                cache[resolution_key] = ppm
                job.calibration_source_by_resolution[resolution_key] = who.video.video_id
            who.assigned_ids.append(who.pending_participant_id)
            who.hip_positions.append(who.pending_hip)
            who.head_positions.append(who.pending_head)
            who.feet_positions.append((x, y))
            who.calibration_writes.append((resolution_key, who.pending_previous_ppm))
            who.message = (
                f"{who.video.video_id} gekalibreerd: {ppm:.1f} px/m"
                + ("." if who.is_redo
                   else " -- geldt ook voor de volgende video's van deze resolutie.")
            )
            who.step = "hip"
            who.pending_participant_id = None
            who.pending_head = None

    return templates.TemplateResponse(request, "partials/who_panel.html", _who_context(job))


@app.post("/who/undo", response_class=HTMLResponse)
def who_undo(request: Request, job: Job = Depends(get_current_job)) -> HTMLResponse:
    who = job.who
    if who is None:
        return templates.TemplateResponse(request, "partials/who_panel.html", _who_context(job))
    cache = job.session.pixels_per_meter_by_resolution
    if who.step != "hip" and who.pending_participant_id is not None:
        participant = next(p for p in job.session.participants if p.participant_id == who.pending_participant_id)
        who.remaining.append(participant)
        who.step = "hip"
        who.pending_participant_id = None
        who.pending_head = None
        who.message = "Bezig met deze atleet geannuleerd."
    elif who.assigned_ids:
        removed_id = who.assigned_ids.pop()
        who.hip_positions.pop()
        who.head_positions.pop()
        who.feet_positions.pop()
        write = who.calibration_writes.pop()
        if write is not None:
            res_key, previous = write
            # Gewijzigd 2026-09-03: ook de waarde die voor DEZE video
            # geldt terugdraaien, niet alleen de gedeelde cache.
            who.pixels_per_meter = previous
            who.pixels_per_meter_source = None
            if not who.is_redo:
                if previous is None:
                    cache.pop(res_key, None)
                    job.calibration_source_by_resolution.pop(res_key, None)
                else:
                    cache[res_key] = previous
        participant = next(p for p in job.session.participants if p.participant_id == removed_id)
        who.remaining.append(participant)
        who.message = f"Laatste toewijzing ({removed_id}) ongedaan gemaakt."
    return templates.TemplateResponse(request, "partials/who_panel.html", _who_context(job))


@app.post("/who/finish", response_class=HTMLResponse)
def who_finish(request: Request, job: Job = Depends(get_current_job)) -> HTMLResponse:
    who = job.who
    if who is not None:
        assignment = FrameAssignment(
            video_id=who.video.video_id,
            frame_index=0,
            frame_width_px=who.frame_width_px,
            frame_height_px=who.frame_height_px,
            participant_ids=who.assigned_ids,
            click_positions=who.hip_positions,
            head_positions=who.head_positions,
            feet_positions=who.feet_positions,
            # Toegevoegd 2026-09-03: leg de px/m die voor deze video
            # geldt hier vast, zodat een latere (her)kalibratie deze
            # video niet meer raakt.
            pixels_per_meter=who.pixels_per_meter,
        )
        job.session.who_assignments[who.video.video_id] = assignment
        if who.pixels_per_meter_source is not None:
            job.calibration_source_by_video[who.video.video_id] = who.pixels_per_meter_source
        job.completed_videos.append(who.video)
    _advance_to_next_video(job)
    return templates.TemplateResponse(request, "partials/who_panel.html", _who_context(job))


@app.post("/who/skip", response_class=HTMLResponse)
def who_skip(request: Request, job: Job = Depends(get_current_job)) -> HTMLResponse:
    who = job.who
    if who is not None:
        job.completed_videos.append(who.video)
    _advance_to_next_video(job)
    return templates.TemplateResponse(request, "partials/who_panel.html", _who_context(job))


@app.post("/who/previous", response_class=HTMLResponse)
def who_previous(request: Request, job: Job = Depends(get_current_job)) -> HTMLResponse:
    """Step back to the most recently finished/skipped video and redo it.

    The video currently in progress (if any) is put back at the front
    of the queue so it isn't lost, and any earlier /who/-assignment for
    the video we're going back to is discarded so it can be redone from
    scratch. The pixels-per-metre recorded for THIS video is kept and
    reused (changed 2026-09-03) -- recalibrating here changes only this
    video, and does NOT change what videos further down the queue
    inherit.
    """
    if not job.completed_videos:
        if job.who is not None:
            job.who.message = "Er is geen eerdere video om naar terug te gaan."
        return templates.TemplateResponse(request, "partials/who_panel.html", _who_context(job))

    previous_video = job.completed_videos.pop()
    if job.who is not None:
        job.videos_needing_who.insert(0, job.who.video)
    # Gewijzigd 2026-09-03: eerst laden (leest de eerder vastgelegde
    # px/m van deze video), dan pas de oude assignment weggooien.
    _load_video_for_who(job, previous_video, is_redo=True)
    job.session.who_assignments.pop(previous_video.video_id, None)
    job.who.message = f"Terug naar {previous_video.video_id} -- wijs de atleten opnieuw toe."
    return templates.TemplateResponse(request, "partials/who_panel.html", _who_context(job))


# ---------------------------------------------------------------------------
# Step 3: analysis (/pose/ -> /export/), run via the shared FIFO-wachtrij
# ---------------------------------------------------------------------------

@app.post("/analysis/start", response_class=HTMLResponse)
def analysis_start(
    request: Request,
    pose_backend: str = Form("yolo"),
    yolo_model_name: str = Form("yolov8n-pose.pt"),
    mediapipe_variant: str = Form("lite"),
    export_filename: str = Form("jump_parameters.xlsx"),
    debug_csv: Optional[str] = Form(None),
    frame_skip: Optional[str] = Form(None),
    use_cache: Optional[str] = Form(None),
    job: Job = Depends(get_current_job),
) -> HTMLResponse:
    if job.session is None or not job.session.who_assignments:
        return HTMLResponse(
            "Minstens één video moet een /who/-toewijzing hebben voor je de analyse start.",
            status_code=400,
        )
    job.log_lines = []
    job.export_path = None
    job.result_count = 0
    job.per_participant_counts = {}
    job.error_message = None

    # Datum/tijdstempel altijd in de bestandsnaam (2026-09-04, TODO.md
    # item 12): zonder dit zouden twee sessies die allebei de
    # standaardnaam "jump_parameters.xlsx" gebruiken (of dezelfde
    # eigen naam intypen) elkaars resultaten alsnog overschrijven,
    # ook al hebben ze nu allebei hun eigen downloads-map.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    export_path = Path(export_filename).expanduser()
    stem = export_path.stem or "jump_parameters"
    suffix = export_path.suffix or ".xlsx"
    stamped_name = f"{stem}_{timestamp}{suffix}"
    if export_path.is_absolute():
        export_path = export_path.parent / stamped_name
    else:
        export_path = job.downloads_dir / stamped_name

    debug_csv_dir = job.debug_csv_dir if debug_csv else None

    # Frame-skip (added 2026-09-03): checkbox in the UI maps to fixed,
    # user-confirmed values (3 frames overslaan, 4.5s dense na trigger)
    # rather than exposing all of track_all_athletes' adaptive_* knobs
    # in the form -- see pipeline.run_analysis for what these actually
    # do and their fallback behaviour when no pixels-per-meter
    # calibration exists yet for a video's resolution.
    adaptive_skip_frames = 3 if frame_skip else None
    adaptive_dense_duration_s = 4.5

    def _task(
        _job=job,
        _pose_backend=pose_backend,
        _export_path=export_path,
        _yolo_model_name=yolo_model_name,
        _mediapipe_variant=mediapipe_variant,
        _debug_csv_dir=debug_csv_dir,
        _adaptive_skip_frames=adaptive_skip_frames,
        _adaptive_dense_duration_s=adaptive_dense_duration_s,
        _use_cache=bool(use_cache),
    ) -> None:
        pipeline.run_analysis(
            _job,
            _pose_backend,
            _export_path,
            _yolo_model_name,
            _mediapipe_variant,
            _debug_csv_dir,
            adaptive_skip_frames=_adaptive_skip_frames,
            adaptive_dense_duration_s=_adaptive_dense_duration_s,
            use_cache=_use_cache,
        )

    # Sequentiële FIFO-wachtrij (2026-09-04, TODO.md item 12) i.p.v. een
    # eigen thread per sessie: job.status wordt "queued" totdat de
    # gedeelde worker-thread aan deze taak toekomt, zodat er nooit twee
    # analyses tegelijk om dezelfde CPU/GPU concurreren.
    ANALYSIS_QUEUE.enqueue(job, _task)
    time.sleep(0.1)  # let the queued status land before the first render
    return templates.TemplateResponse(request, "partials/status.html", {"job": job})


@app.get("/status", response_class=HTMLResponse)
def status(request: Request, job: Job = Depends(get_current_job)) -> HTMLResponse:
    return templates.TemplateResponse(request, "partials/status.html", {"job": job})


@app.get("/download")
def download_results(job: Job = Depends(get_current_job)):
    if job.export_path is None or not job.export_path.exists():
        return JSONResponse({"error": "Nog geen resultaten."}, status_code=404)
    return FileResponse(job.export_path, filename=job.export_path.name)
