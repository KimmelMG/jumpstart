"""Startcontrole voor Jumpstart.

Wordt aangeroepen door run_server.main() VOORDAT de server start. Controleert
in een keer alles wat bij het opstarten mis kan gaan, en geeft bij een
probleem een duidelijke Nederlandse melding met de oplossing -- in plaats van
een Python-foutmelding halverwege het opstarten.

Bewust alleen standaard-Python (geen fastapi/uvicorn/numpy importeren): deze
controle moet ook werken als juist die packages ontbreken.
"""

from __future__ import annotations

import asyncio
import importlib.util
import socket
import sys
import uuid
from pathlib import Path

# De map "Jumpstart" (bevat backend_script/ en frontend_webapp/).
APP_DIR = Path(__file__).resolve().parent.parent

# Moet overeenkomen met requires-python in pyproject.toml.
REQUIRED_PYTHON = (3, 12)

# Importnaam -> packagenaam (zoals in pyproject.toml).
REQUIRED_MODULES = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "jinja2": "jinja2",
    "python_multipart": "python-multipart",
    "qrcode": "qrcode",
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "cv2": "opencv-python",
    "matplotlib": "matplotlib",
    "openpyxl": "openpyxl",
    "ultralytics": "ultralytics",
    "torch": "torch",
}

MODEL_FILE = "yolov8n-pose.pt"

START_HINT = "Start Jumpstart door Jumpstart_starten.bat te dubbelklikken."

LINE = "=" * 66


def _inside_running_event_loop() -> bool:
    """True als er al een asyncio event loop draait (Spyder, Jupyter)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _port_in_use(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError:
        return True
    finally:
        sock.close()
    return False


def _can_write(folder: Path) -> bool:
    test_file = folder / f".schrijftest_{uuid.uuid4().hex}"
    try:
        test_file.write_text("ok", encoding="utf-8")
    except OSError:
        return False
    try:
        test_file.unlink()
    except OSError:
        pass  # schrijven lukte; opruimen is niet essentieel
    return True


def run_checks(port: int) -> list[str]:
    """Geeft een lijst met problemen terug (leeg = alles in orde)."""

    # 1. Spyder/Jupyter: dan kan de server niet starten, de rest is dan
    #    niet relevant.
    if _inside_running_event_loop():
        return [
            "Jumpstart kan niet vanuit Spyder of Jupyter gestart worden (die\n"
            "  draaien zelf al een event loop). " + START_HINT
        ]

    problems: list[str] = []

    # 2. Python-versie.
    if sys.version_info[:2] != REQUIRED_PYTHON:
        wanted = ".".join(map(str, REQUIRED_PYTHON))
        found = ".".join(map(str, sys.version_info[:3]))
        problems.append(
            f"Verkeerde Python-versie: {found} (nodig: {wanted}).\n"
            "  Jumpstart is waarschijnlijk met een andere Python gestart dan de\n"
            "  eigen installatie. " + START_HINT
        )

    # 3. Mapstructuur.
    for needed in ("backend_script/__init__.py", "frontend_webapp/app.py"):
        if not (APP_DIR / needed).exists():
            problems.append(
                f"Onderdeel ontbreekt: {needed} (in {APP_DIR}).\n"
                "  Download de hele map opnieuw als zip en pak die volledig uit."
            )

    # 4. Packages.
    missing = [
        package
        for module, package in REQUIRED_MODULES.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        problems.append(
            "Deze packages ontbreken: " + ", ".join(missing) + ".\n"
            "  De installatie is niet (helemaal) gelukt of Jumpstart is met een\n"
            "  andere Python gestart. " + START_HINT + "\n"
            "  Die installeert ontbrekende packages automatisch."
        )

    # 5. Pose-model.
    if not (APP_DIR / MODEL_FILE).exists():
        problems.append(
            f"Het pose-model {MODEL_FILE} ontbreekt in {APP_DIR}.\n"
            "  Download de hele map opnieuw als zip en pak die volledig uit."
        )

    # 6. Schrijfrechten (resultaten en tijdelijke bestanden komen in
    #    frontend_webapp/).
    if not _can_write(APP_DIR / "frontend_webapp"):
        problems.append(
            f"Jumpstart kan niet schrijven in {APP_DIR / 'frontend_webapp'}.\n"
            "  Zet de map Jumpstart op een plek waar je bestanden mag opslaan,\n"
            "  bijvoorbeeld in Documenten, en start daar opnieuw."
        )

    # 7. Poort vrij.
    if _port_in_use(port):
        problems.append(
            f"Poort {port} is al in gebruik. Waarschijnlijk draait Jumpstart al\n"
            "  in een ander venster: open dan gewoon http://127.0.0.1:"
            f"{port} in de\n"
            "  browser, of sluit dat venster en start opnieuw."
        )

    return problems


def check_or_exit(port: int) -> None:
    """Voert alle controles uit; stopt Jumpstart netjes bij een probleem."""
    problems = run_checks(port)
    if not problems:
        return
    print()
    print(LINE)
    print("  Jumpstart kan niet starten:")
    for number, problem in enumerate(problems, start=1):
        print()
        # Vervolgregels netjes onder de tekst na het nummer laten beginnen.
        text = problem.replace("\n  ", "\n     ")
        print(f"  {number}. {text}")
    print(LINE)
    print()
    raise SystemExit(1)
