"""Per-session in-memory state for the Jumpstart web front-end.

UPDATE 2026-09-04 (sessie-isolatie + FIFO-wachtrij, TODO.md item 12):
dit bestand was tot nu toe een enkel globaal `JOB`-object -- expliciet
NIET multi-user, zie de oude module-docstring hieronder. Dat is nu
vervangen door een `JobManager` die per browser/sessie (via een cookie,
zie app.py) een eigen `Job` bijhoudt, zodat meerdere mensen
tegelijkertijd onafhankelijk van elkaar door stap 1/2 (deelnemers,
video's, /who/) kunnen lopen zonder elkaars voortgang te overschrijven.

De zware pose-analysestap (/pose/+/tracking/+/events/) blijft
bewust SEQUENTIEEL draaien -- zie `AnalysisQueue` onderaan dit bestand.
De hardware van een toekomstige gebruiker is niet vooraf bekend (zie
TODO.md item 12), dus een simpele, altijd-werkende FIFO-wachtrij is de
veilige default: elke sessie kan zijn analyse inplannen, en ze worden
één voor één afgehandeld door een enkele achtergrond-worker-thread.
Echte parallelle analyses (met een dynamisch aantal workers, afgestemd
op de beschikbare hardware) is een losse, latere uitbreiding -- niet in
deze wijziging.

Oorspronkelijke (nu achterhaalde) module-docstring, ter documentatie:
    "In-memory state for the (single-user, local) Jumpstart web
    front-end. This is deliberately NOT a database and NOT multi-user
    -- the whole point of this tool is 'one researcher, one laptop,
    one HDMI cable to a projector/monitor while presenting or running
    a session'. A single global Job holds everything..."
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from jumpstart.io.video_input import VideoFile
from jumpstart.profiles.models import Participant
from jumpstart.session import Session
from jumpstart.who.assignment import FrameAssignment

BASE_DIR = Path(__file__).resolve().parent

# Gedeelde, sessie-onafhankelijke mappen. DOWNLOADS_DIR blijft hier
# alleen nog voor het (stateloze) deelnemers-sjabloon
# (download_template in app.py) -- de daadwerkelijke resultaten per
# sessie staan voortaan onder elke sessie se eigen map, zie
# Job._SESSIONS_DIR/<session_id>/downloads hieronder.
DOWNLOADS_DIR = BASE_DIR / "_downloads"
# Detections-cache (2026-09-04): BEWUST gedeeld tussen alle sessies,
# niet per sessie -- het is een pure prestatie-cache, keyed op
# video-id + pose-model + frame-skip-instelling (zie
# pipeline._detections_cache_path_for), dus hergebruik tussen twee
# sessies die toevallig dezelfde video analyseren is alleen maar
# winst, geen correctheidsrisico.
DETECTIONS_CACHE_DIR = BASE_DIR / "_detections_cache"
# Sessie-specifieke data (uploads, referentieframes, resultaten,
# debug-CSV's) komt onder deze map, één submap per sessie-id -- zie
# Job.__post_init__ hieronder. Dit voorkomt dat twee gelijktijdige
# sessies elkaars geuploade bestanden, referentieframes (die anders
# allebei "V001.jpg" zouden heten) of resultaatbestanden overschrijven.
SESSIONS_DIR = BASE_DIR / "_sessions"

for _dir in (DOWNLOADS_DIR, DETECTIONS_CACHE_DIR, SESSIONS_DIR):
    _dir.mkdir(exist_ok=True)


@dataclass
class WhoClickState:
    """In-progress /who/ assignment state for ONE video.

    Mirrors the state machine in jumpstart/who/assignment.py's
    assign_athlete_order (hip -> [head -> feet] -> hip -> ...), but
    driven by HTTP requests from the browser instead of matplotlib
    mouse/key events.
    """

    video: VideoFile
    frame_path: Path
    frame_width_px: int
    frame_height_px: int
    remaining: List[Participant]
    assigned_ids: List[str] = field(default_factory=list)
    hip_positions: List[Tuple[float, float]] = field(default_factory=list)
    head_positions: List[Optional[Tuple[float, float]]] = field(default_factory=list)
    feet_positions: List[Optional[Tuple[float, float]]] = field(default_factory=list)
    # None = no calibration write happened for this athlete; otherwise
    # (resolution_key, previous_ppm_or_None) for undo, same convention
    # as assignment.py's established_resolution.
    calibration_writes: List[Optional[Tuple[Tuple[int, int], Optional[float]]]] = field(
        default_factory=list
    )
    # Toegevoegd 2026-09-03: de px/m die voor DEZE video geldt. Bij een
    # nieuwe video overgenomen van de laatste kalibratie voor deze
    # resolutie; bij het opnieuw doen van een al afgeronde video
    # ("Vorige video") overgenomen van wat er toen voor die video is
    # vastgelegd. Wordt bij /who/finish op de FrameAssignment gezet.
    pixels_per_meter: Optional[float] = None
    # video_id waar deze px/m gekalibreerd is, alleen voor de weergave.
    pixels_per_meter_source: Optional[str] = None
    # True als deze video via "Vorige video" opnieuw wordt gedaan --
    # dan geldt een herkalibratie ALLEEN voor deze video.
    is_redo: bool = False
    step: str = "hip"  # "hip" | "head" | "feet"
    pending_participant_id: Optional[str] = None
    pending_hip: Optional[Tuple[float, float]] = None
    pending_head: Optional[Tuple[float, float]] = None
    pending_previous_ppm: Optional[float] = None
    message: str = ""


class Job:
    """One session's state: loaded participants, registered videos, the
    /who/ assignment in progress, and the analysis run's live log +
    results.

    UPDATE 2026-09-04: voorheen was er precies één van deze objecten
    (het globale `JOB`), nu heeft elke browser-sessie (zie app.py's
    sessie-cookie) zijn eigen `Job`, aangemaakt/opgehaald via
    `JOB_MANAGER.get_or_create(session_id)` onderaan dit bestand.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.lock = threading.Lock()
        # Eigen mappen voor deze sessie, zodat gelijktijdige sessies
        # elkaars uploads/referentieframes/resultaten/debug-CSV's nooit
        # kunnen overschrijven. Een korte prefix (eerste 8 tekens van
        # de sessie-id) is genoeg -- de volledige id is 32 tekens (hex
        # uuid4) en zou paden onnodig lang maken.
        session_dir = SESSIONS_DIR / session_id[:8]
        self.uploads_dir = session_dir / "uploads"
        self.frames_dir = session_dir / "frames"
        self.downloads_dir = session_dir / "downloads"
        self.debug_csv_dir = session_dir / "debug_csv"
        for _dir in (self.uploads_dir, self.frames_dir, self.downloads_dir, self.debug_csv_dir):
            _dir.mkdir(parents=True, exist_ok=True)
        self.reset()

    def reset(self) -> None:
        self.participants: List[Participant] = []
        self.session: Optional[Session] = None
        self.videos_needing_who: List[VideoFile] = []
        self.who: Optional[WhoClickState] = None
        self.setup_error: Optional[str] = None
        # Videos for which /who/ was already finished or skipped, in the
        # order that happened -- lets "Vorige video" step back and redo
        # an earlier video's assignment.
        self.completed_videos: List[VideoFile] = []
        # Toegevoegd 2026-09-03, alleen voor de weergave in het
        # /who/-paneel: bij welke video is de px/m gekalibreerd die nu
        # voor een resolutie geldt (doorgeefwaarde), resp. die voor een
        # al afgeronde video is vastgelegd.
        self.calibration_source_by_resolution: Dict[Tuple[int, int], str] = {}
        self.calibration_source_by_video: Dict[str, str] = {}

        # idle | queued | running | done | error -- "queued" en de
        # bijbehorende queue_position zijn nieuw sinds de FIFO-wachtrij
        # (2026-09-04, zie AnalysisQueue onderaan dit bestand).
        self.status: str = "idle"
        self.queue_position: int = 0
        self.log_lines: List[str] = []
        self.export_path: Optional[Path] = None
        self.result_count: int = 0
        self.per_participant_counts: Dict[str, int] = {}
        self.error_message: Optional[str] = None

    def log(self, text: str) -> None:
        with self.lock:
            for line in str(text).splitlines() or [""]:
                self.log_lines.append(line)


class JobManager:
    """Keeps one Job per browser session, keyed by an opaque session id.

    Thread-safe: get_or_create is called from FastAPI's request-handling
    threads and can race between two near-simultaneous first requests
    from the same brand-new session id (rare, but the lock makes it
    safe either way).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, Job] = {}

    def get_or_create(self, session_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(session_id)
            if job is None:
                job = Job(session_id)
                self._jobs[session_id] = job
            return job

    def count(self) -> int:
        with self._lock:
            return len(self._jobs)


JOB_MANAGER = JobManager()


# ---------------------------------------------------------------------------
# FIFO-wachtrij voor de zware analysestap (2026-09-04)
# ---------------------------------------------------------------------------

class AnalysisQueue:
    """Runs at most one analysis (/pose/+/tracking/+/events/) at a time,
    across ALL sessions, in first-in-first-out order.

    Waarom sequentieel: de hardware van een toekomstige gebruiker is
    niet vooraf bekend (zie TODO.md item 12) -- twee zware
    pose-analyses tegelijk kunnen op onbekende/lichte hardware alleen
    maar allebei trager worden, zonder garantie dat dat sneller is dan
    gewoon wachten. Een enkele achtergrond-worker-thread haalt taken
    één voor één van de wachtrij en voert ze uit; elke sessie ziet zijn
    eigen status ("queued" met een postie-nummer, dan "running", dan
    "done"/"error") via zijn eigen Job-object.
    """

    def __init__(self) -> None:
        self._queue: "queue.Queue[Tuple[Job, Callable[[], None]]]" = queue.Queue()
        self._pending: List[Job] = []  # zelfde volgorde als de queue, voor positienummers
        self._pending_lock = threading.Lock()
        self._worker_started = False
        self._worker_lock = threading.Lock()

    def _ensure_worker(self) -> None:
        # Lazy-gestart (i.p.v. bij module-import) zodat een testimport
        # van dit bestand niet meteen een achtergrond-thread opstart.
        with self._worker_lock:
            if not self._worker_started:
                threading.Thread(target=self._worker_loop, daemon=True).start()
                self._worker_started = True

    def enqueue(self, job: Job, task: Callable[[], None]) -> None:
        """Add one analysis run to the queue and mark the job "queued"."""
        self._ensure_worker()
        with self._pending_lock:
            self._pending.append(job)
            self._renumber_locked()
        with job.lock:
            job.status = "queued"
        self._queue.put((job, task))

    def _renumber_locked(self) -> None:
        # Moet met self._pending_lock vast worden aangeroepen.
        for position, pending_job in enumerate(self._pending, start=1):
            with pending_job.lock:
                pending_job.queue_position = position

    def _worker_loop(self) -> None:
        while True:
            job, task = self._queue.get()
            with self._pending_lock:
                if job in self._pending:
                    self._pending.remove(job)
                self._renumber_locked()
            with job.lock:
                job.status = "running"
                job.queue_position = 0
            try:
                task()
            except Exception as exc:  # noqa: BLE001 -- nooit de worker zelf laten crashen
                job.log(f"\nFOUT (wachtrij-worker): {exc}")
                with job.lock:
                    job.status = "error"
                    job.error_message = str(exc)
            finally:
                self._queue.task_done()


ANALYSIS_QUEUE = AnalysisQueue()
