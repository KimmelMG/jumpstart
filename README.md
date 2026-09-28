# Jumpstart

Python-tool voor markerless sprongdetectie en -analyse (countermovement
jumps) uit video bij meerdere personen in beeld: personen-detectie (YOLO-pose), tracking, sprongdetectie
en het berekenen van sprongmaten (hoogte, vluchttijd, RSI, etc.), met
zowel een command-line-route als een lokale webinterface. 


> [!TIP]
> **Uitlegvideo:** [Bekijk de video](https://github.com/KimmelMG/jumpstart/releases/download/v1.0/Jumpstart_uitlegvideo_take2.mp4)  
> **Pilotvalidatie en methode:** [Open het Word-document](https://github.com/KimmelMG/jumpstart/releases/download/v1.0/Jumpstart_pilotvalidatie_methode_NL.docx)


## Snel starten (Windows)

1. **Download de hele repository, niet losse bestanden.** Groene knop
   **Code -> Download ZIP**.
2. **Pak de zip uit:** rechtermuisknop op de zip -> **Alles uitpakken...**.
   Dubbelklikken op de .bat *binnen* de zip werkt niet.
3. **Dubbelklik `Jumpstart_starten.bat`** in de uitgepakte map. Naast de
   .bat hoort de map `Jumpstart` te staan. Verschijnt er een blauw
   Windows-venster ("Windows heeft uw pc beveiligd"), kies dan
   **Meer informatie -> Toch uitvoeren**.
4. De eerste keer installeert Jumpstart een eigen Python 3.12 en alle
   onderdelen (een paar minuten, internet nodig, ongeveer 3 GB in
   `%USERPROFILE%\.jumpstart`). Je hoeft zelf geen Python te installeren;
   een Python die al op je computer staat (Microsoft Store, Anaconda, ...)
   wordt bewust niet gebruikt. Daarna opent de browser vanzelf op
   `http://127.0.0.1:8000`. Laat het zwarte venster open; sluiten stopt
   Jumpstart.

Daarna start Jumpstart binnen enkele seconden, ook zonder internet. Je kunt
ook de snelkoppeling gebruiken die op je bureaublad is aangemaakt.

**Let op**

- Start Jumpstart alleen via `Jumpstart_starten.bat` (of de snelkoppeling),
  niet vanuit Spyder, Jupyter of een andere editor.
- Windows vraagt eenmalig om beheerdersrechten voor de firewall. Kies **Ja**
  als je een telefoon of tablet wilt gebruiken; bij **Nee** werkt Jumpstart
  alleen op deze computer.
- Kan Jumpstart niet starten, dan staat in het zwarte venster wat er mis is
  en wat je kunt doen.

## Voor ontwikkelaars

De packages en hun exacte versies staan in `Jumpstart/pyproject.toml` en
`Jumpstart/uv.lock`, en worden geinstalleerd met het meegeleverde
`Jumpstart/tools/uv.exe` ([uv](https://github.com/astral-sh/uv), licentie
in `Jumpstart/tools/`). Een versie aanpassen: wijzig `pyproject.toml`, draai
in de map `Jumpstart` het commando `tools\uv.exe lock` en commit beide
bestanden samen. Meer achtergrond:
[`Jumpstart/frontend_webapp/README.md`](Jumpstart/frontend_webapp/README.md).
  
## Structuur

```
jumpstart/                 analyse-package (personen-detectie, tracking,
                            sprongdetectie, berekeningen, CLI)
  workflow_demo.py           command-line-ingang
jumpstart_webapp/          lokale webinterface (FastAPI + HTMX)
  README.md                 volledige uitleg: installeren, starten,
                             telefoon/tablet-toegang, structuur
Jumpstart_starten.bat      dubbelklikken om te starten (installeert
                            zo nodig eerst de venv/packages)
participant_template.xlsx  leeg deelnemersbestand om in te vullen
  tools/uv.exe                 installeert Python + packages (niet zelf starten)
  pyproject.toml, uv.lock      vastgelegde Python-versie en packageversies
```

## Deelnemersbestand

Vul `participant_template.xlsx` in met de deelnemers van je sessie en
gebruik dat bestand als upload in stap 1 van de webinterface (of als
`--participants` op de command line).

## Video's

Video's horen niet in deze repository -- lever ze rechtstreeks aan bij de
analyse (via een videomap, losse bestanden, of upload in de
webinterface). Modelgewichten (YOLO-pose) worden automatisch gedownload
bij het eerste gebruik.

## Meer weten

Voor installatie-details, netwerk-/firewallconfiguratie en
telefoon/tablet-toegang op hetzelfde wifi-netwerk: zie
`jumpstart_webapp/README.md`.
