"""Eenmalige migratie van oude detectie-cachebestanden.

Waarom dit nodig is
-------------------
De dataclass PersonDetection stond tot 2026-09-17 in
jumpstart/pose/detector.py (de MediaPipe-backend). Dat bestand is
verwijderd bij het opschonen van MediaPipe, en de dataclass staat nu
in jumpstart/pose/detector_yolo.py.

In een pickle-bestand wordt per object opgeslagen UIT WELKE MODULE de
class kwam. Alle cachebestanden die voor die verhuizing zijn gemaakt
verwijzen dus naar 'jumpstart.pose.detector', een module die niet meer
bestaat. Inladen geeft daardoor:

    ModuleNotFoundError: No module named 'jumpstart.pose.detector'

De opgeslagen gegevens zelf mankeren niets -- het zijn per detectie
gewoon drie floats (x, y, visibility). Alleen het modulelabel klopt
niet meer. Dit script laadt elk cachebestand eenmalig in met een
tijdelijke omleiding van de oude modulenaam naar de nieuwe, en schrijft
het daarna terug met het juiste label. Daarna laden de caches gewoon
native en is dit script niet meer nodig.

Gebruik
-------
Draai dit vanuit de hoofdmap van het project (de map waar de mappen
jumpstart/ en jumpstart_webapp/ in staan), met de venv van Jumpstart
actief:

    python migrate_detections_cache.py

Of met een expliciete cachemap:

    python migrate_detections_cache.py pad/naar/_detections_cache

Het script maakt van elk bestand eerst een .bak-kopie, controleert na
het wegschrijven of het nieuwe bestand echt zonder omleiding inlaadt,
en zet het origineel terug als er iets misgaat. Bestanden die al goed
zijn worden overgeslagen.
"""

from __future__ import annotations

import pickle
import shutil
import sys
import types
from pathlib import Path

OLD_MODULE = "jumpstart.pose.detector"
NEW_MODULE = "jumpstart.pose.detector_yolo"


def find_cache_dir(argv: list[str]) -> Path:
    """Bepaal welke map met cachebestanden gemigreerd moet worden."""
    if len(argv) > 1:
        cache_dir = Path(argv[1])
    else:
        cache_dir = Path("jumpstart_webapp") / "_detections_cache"
    if not cache_dir.is_dir():
        sys.exit(
            f"Cachemap niet gevonden: {cache_dir}\n"
            "Draai dit script vanuit de hoofdmap van het project, of geef "
            "het pad naar _detections_cache mee als argument."
        )
    return cache_dir


def install_alias() -> None:
    """Laat de oude modulenaam tijdelijk naar de nieuwe class wijzen.

    Alleen actief binnen dit script; er wordt niets aan het project
    zelf gewijzigd.
    """
    from jumpstart.pose.detector_yolo import PersonDetection

    if PersonDetection.__module__ != NEW_MODULE:
        sys.exit(
            f"Onverwacht: PersonDetection zegt uit "
            f"'{PersonDetection.__module__}' te komen in plaats van "
            f"'{NEW_MODULE}'. Migratie afgebroken."
        )

    shim = types.ModuleType(OLD_MODULE)
    shim.PersonDetection = PersonDetection
    sys.modules[OLD_MODULE] = shim


def remove_alias() -> None:
    sys.modules.pop(OLD_MODULE, None)


def loads_without_alias(path: Path) -> bool:
    """Controleer of een bestand inlaadt zonder de omleiding."""
    remove_alias()
    try:
        with open(path, "rb") as handle:
            pickle.load(handle)
        return True
    except Exception:
        return False
    finally:
        install_alias()


def migrate_file(path: Path) -> str:
    """Migreer een enkel cachebestand. Geeft een statuswoord terug."""
    if loads_without_alias(path):
        return "overgeslagen (al goed)"

    try:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
    except Exception as error:
        return f"MISLUKT bij inladen ({type(error).__name__}: {error})"

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)

    try:
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)
    except Exception as error:
        shutil.copy2(backup, path)
        backup.unlink(missing_ok=True)
        return f"MISLUKT bij wegschrijven, origineel teruggezet ({error})"

    if not loads_without_alias(path):
        shutil.copy2(backup, path)
        backup.unlink(missing_ok=True)
        return "MISLUKT bij controle, origineel teruggezet"

    backup.unlink(missing_ok=True)
    return "gemigreerd"


def main() -> None:
    sys.path.insert(0, str(Path.cwd()))
    cache_dir = find_cache_dir(sys.argv)
    install_alias()

    files = sorted(p for p in cache_dir.glob("*.pkl") if p.is_file())
    if not files:
        print(f"Geen .pkl-bestanden gevonden in {cache_dir}.")
        return

    print(f"{len(files)} cachebestand(en) gevonden in {cache_dir}\n")

    counts: dict[str, int] = {}
    for path in files:
        status = migrate_file(path)
        key = status.split(" ")[0]
        counts[key] = counts.get(key, 0) + 1
        print(f"  {path.name}: {status}")

    remove_alias()

    print("\nKlaar.")
    for key, count in sorted(counts.items()):
        print(f"  {key}: {count}")
    if counts.get("MISLUKT"):
        print(
            "\nLet op: mislukte bestanden zijn onveranderd gebleven. Die "
            "worden bij de eerstvolgende analyse gewoon genegeerd (het "
            "vangnet in tracker.py) en opnieuw gedetecteerd."
        )


if __name__ == "__main__":
    main()
