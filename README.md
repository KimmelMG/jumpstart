# Jumpstart

Python-tool voor markerless sprongdetectie en -analyse (countermovement
jumps) uit video: personen-detectie (YOLO-pose), tracking, sprongdetectie
en het berekenen van sprongmaten (hoogte, vluchttijd, RSI, etc.), met
zowel een command-line-route als een lokale webinterface.

## Snel starten

Dubbelklik op `Jumpstart_starten.bat`. Dat regelt bij de eerste keer
automatisch een virtuele omgeving + alle benodigde packages, en start
daarna de webinterface (opent zelf een browsertab).

Wil je zelf handmatig installeren of vanaf de command line werken, zie
de instructies in `jumpstart_webapp/README.md`.

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
requirements-analysis.txt  Python-dependencies voor de analyse
requirements-web.txt       Python-dependencies voor de webinterface
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
