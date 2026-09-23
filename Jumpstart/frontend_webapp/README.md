Jumpstart -- webinterface
Een lokale webinterface (FastAPI + HTMX) rond het bestaande `backend_script/`-script.
Verandert niets aan `backend_script/` zelf -- roept precies dezelfde functies aan
als `python -m backend_script.workflow_demo`, dus resultaten zijn identiek aan de
command-line-route.
Waarom FastAPI + HTMX (en niet React/Streamlit)?
FastAPI is een Python-webframework: het draait gewoon in hetzelfde
venv als de rest van het script en kan `jumpstart.*` direct importeren en
aanroepen -- geen aparte JavaScript-app die los van je Python-code moet
praten met een eigen API-laag.
HTMX is een klein JavaScript-bibliotheekje waarmee gewone HTML-knoppen
en -formulieren stukjes van de pagina kunnen verversen (bv. het logpaneel
elke seconde) zonder dat je zelf React/Vue hoeft te leren. Je schrijft
vrijwel alleen HTML + Python.
Dit was ook al de eigen conclusie in "User flow and script structure.docx"
("de zoete plek" voor dit soort praktijkproduct) -- minder werk dan een
React-app, blijft dicht bij je Python, en ziet er met Tailwind netjes uit.
Streamlit is sneller te bouwen maar minder maatwerk mogelijk voor een
echt gepolijste presentatie-uitstraling; React geeft meer controle maar
kost veel meer tijd. Voor "moet er professioneel uitzien en soepel werken
tijdens een presentatie" is FastAPI+HTMX de beste prijs/kwaliteit.
Installeren (eenmalig, in hetzelfde venv als het stappenplan)
```powershell
cd "C:\Users\kimme\OneDrive\Documenten\School\Sport Sciences\Jumpstart\Phase 3 - Script development 13-7"
.\venv\Scripts\Activate.ps1
pip install fastapi "uvicorn[standard]" jinja2 python-multipart qrcode
```
(De rest -- numpy, opencv-python, mediapipe, etc. -- staat al in je
Stappenplan.)
Starten
```powershell
cd "C:\Users\kimme\OneDrive\Documenten\School\Sport Sciences\Jumpstart\Phase 3 - Script development 13-7"
.\venv\Scripts\Activate.ps1
python -m frontend_webapp.run_server
```
Dit start dezelfde server als `uvicorn frontend_webapp.app:app --host 0.0.0.0 --port 8000` (die vlaggen zitten al in `run_server.py`), maar print
daarnaast automatisch het adres waarmee een telefoon/tablet op hetzelfde
wifi-netwerk verbinding kan maken, inclusief een scanbare QR-code -- zie
hieronder. Werkt `run_server` om wat voor reden dan ook niet (bijv. `qrcode`
nog niet geinstalleerd), dan is `uvicorn frontend_webapp.app:app --host 0.0.0.0 --port 8000` het exacte alternatief zonder de extra printjes.
Open daarna `http://127.0.0.1:8000` in een browser op dezelfde laptop. Voor
een presentatie: zet de browser fullscreen (F11) op het scherm dat via HDMI
is aangesloten.
Telefoon/tablet op hetzelfde wifi-netwerk laten meekijken
Start de server met `python -m frontend_webapp.run_server` (zie
hierboven) -- in de terminal verschijnt meteen het adres voor de
telefoon/tablet plus een QR-code, zonder dat je zelf `ipconfig` hoeft te
draaien.
Windows Firewall vraagt de eerste keer dat de server op dit
host-adres start om toegang toe te staan voor Python/uvicorn --
accepteer dat. Dit is eenmalig per laptop: eenmaal geaccepteerd, blijft
die regel staan voor elke volgende sessie, ook voor andere gebruikers van
diezelfde laptop.
Scan de QR-code met de camera van de telefoon/tablet (zelfde
wifi-netwerk als de laptop), of typ het geprinte adres over,
bijvoorbeeld `http://192.168.1.42:8000`.
De pose-detectie blijft altijd op de laptop draaien -- de telefoon is alleen
een extern scherm + bediening, nooit de machine die de analyse uitvoert.
Dit werkt alleen binnen hetzelfde lokale netwerk, niet over het internet.
Wat de webinterface doet
Deelnemers & video's -- upload het ingevulde deelnemersbestand
(.xlsx/.csv, template te downloaden rechtsboven) en geef aan hoe je de
video's aanlevert: een videomap (net als `--video-folder`), losse
videopaden (net als `--videos`), of ze rechtstreeks uploaden vanuit de
browser -- handig als je vanaf een telefoon/tablet werkt en de video's nog
niet op de servermachine staan.
Atleten toewijzen (`/who/`) -- per video zie je het referentiebeeld
in de browser. Kies een profiel, klik op de heup; bij de eerste atleet
per resolutie volgen ook een kruin- en voetenklik voor de
pixel-naar-meter-kalibratie, exact dezelfde logica als de bestaande
matplotlib-versie in `jumpstart/who/assignment.py` (die blijft ook
gewoon werken vanaf de command line).
Analyse & resultaten -- kies MediaPipe (standaard) of YOLO-pose
(nieuw, nog niet gevalideerd -- zie het Stappenplan), en start de
analyse. De voortgang die het script al printte (per video/atleet,
aantal sprongen, etc.) verschijnt nu live in het logpaneel in de
browser in plaats van in een command-line-venster. Aan het einde staat
er een overzicht per deelnemer plus een downloadknop voor het
resultatenbestand.
Bewuste vereenvoudigingen (v1)
De losse tuning-vlaggen (`--tolerance-m-s2`, `--smoothing-window-s`,
`--min-flight-duration-s`, `--max-match-speed-m-s`, `--debug-csv-dir`,
...) staan niet in de webinterface -- die blijven op dezelfde
standaardwaarden als het script. Gebruik de command line als je die wilt
aanpassen.
De "forceer herkalibratie" (`c`-toets in de matplotlib-versie) kan in de
webversie via het vinkje "Herkalibreer deze resolutie" dat verschijnt
zodra een resolutie al gekalibreerd is.
"Ongedaan maken" verwijdert in de webversie steeds de laatst toegewezen
atleet (inclusief kalibratie-rollback als die net was vastgelegd) -- dit
is hetzelfde eindresultaat als de 'u'-toets, alleen zonder de
tussenstap-voor-tussenstap variant tijdens een lopende kalibratieklik.
Bij weinig schijfruimte pauzeert het script niet meer (dat kan niet
zonder terminal) -- er verschijnt een waarschuwing in het logpaneel in
plaats daarvan. Maak dus proactief ruimte vrij bij grote videomappen.
Eén sessie tegelijk (geen meerdere gebruikers/tabs die tegelijk aan het
toewijzen zijn) -- past bij het gebruik: één laptop, één presentatie.
Video-upload (video_mode "upload") heeft geen eigen voortgangsbalk -- de
browser laat de normale uploadvoortgang zien, maar de webinterface zelf
toont pas iets zodra de upload klaar is. Bij een bestand met dezelfde
naam als een eerdere upload wordt die overschreven (net als bij het
deelnemersbestand).
Offline gebruik (geen CDN, geen internet nodig)
Tailwind en HTMX worden niet meer van een CDN geladen (`cdn.tailwindcss.com`
/ `unpkg.com`) -- beide staan nu lokaal in `static/`:
`static/tailwind.min.css` -- een vooraf gebouwde, geminificeerde Tailwind-
build (via de officiele Tailwind CLI), gescand tegen alle huidige
`templates/*.html`-bestanden. Alleen classes die daadwerkelijk in de
templates voorkomen zitten erin (kleiner bestand dan de volledige CDN-
build).
`static/htmx.min.js` -- exact dezelfde HTMX-versie (1.9.12) die eerder via
unpkg werd geladen, nu lokaal.
Let op bij nieuwe HTML/Tailwind-classes: omdat `tailwind.min.css` gebouwd
is op basis van de huidige templates, worden Tailwind-classes die je later
toevoegt in nieuwe HTML (bijv. bij een toekomstige uitbreiding) pas zichtbaar
na een nieuwe build van dat bestand -- anders dan met de oude CDN-versie, die
alles live compileerde. Vraag bij een volgende HTML-aanpassing gewoon om een
nieuwe `tailwind.min.css`, dan wordt die meegeleverd.
Structuur
```
frontend_webapp/
  app.py          FastAPI-routes
  run_server.py   opstart-helper: print LAN-adres + QR-code, start dan uvicorn
  pipeline.py     analyse-run (/pose/ t/m /export/) in een achtergrondthread
  state.py        gedeelde staat (deelnemers, video's, /who/-voortgang, log)
  templates/       HTML (Jinja2 + HTMX + Tailwind, beide lokaal geladen)
  static/          eigen CSS + lokale Tailwind-build en HTMX (geen CDN)
  _uploads/ _frames/ _downloads/   runtime-mappen (mag je negeren in git)
```