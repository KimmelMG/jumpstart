"""Startpunt van de Jumpstart-webinterface.

Jumpstart_starten.bat roept dit aan met:
    python -m frontend_webapp.run_server

Wat main() doet, in volgorde:
1. Startcontrole (preflight.py): Spyder/Jupyter, Python-versie, packages,
   model, schrijfrechten, poort vrij. Bij een probleem: duidelijke melding
   en netjes stoppen, in plaats van een Python-foutmelding.
2. Laat zien hoe de webinterface te bereiken is (adres + QR-code voor een
   telefoon). Bewust neutraal: we kunnen niet betrouwbaar zien of deze
   computer op wifi of kabel zit (sinds Windows 11 24H2 heeft `netsh wlan`
   locatietoestemming nodig) en ook niet of het netwerk verkeer tussen
   apparaten toestaat. Dus geen conclusie als "vaste verbinding" -- dat
   bracht een supervisor op 2026-09-23 op het verkeerde spoor.
3. Zet de werkmap op de map Jumpstart (zodat yolov8n-pose.pt altijd lokaal
   gevonden wordt), opent de browser en start uvicorn.

Werkt ook als het bestand direct gestart wordt (python pad/naar/run_server.py),
vanuit elke map: de map Jumpstart wordt zelf aan het zoekpad toegevoegd.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

# De map "Jumpstart" (bevat backend_script/ en frontend_webapp/). Wordt aan
# het zoekpad toegevoegd, zodat dit bestand ook direct gestart kan worden.
APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from frontend_webapp import preflight  # noqa: E402 -- na de sys.path-regel

try:
    import qrcode
except ImportError:  # pragma: no cover -- qrcode is optional, see docstring
    qrcode = None

HOST = "0.0.0.0"
PORT = 8000
LINE = "=" * 66

# Name of the inbound firewall rule that Jumpstart_starten.bat creates on
# first run. Kept in sync with that file -- if you rename it in one place,
# rename it in the other, or this check silently stops finding it.
FIREWALL_RULE_NAME = "Jumpstart"
FIREWALL_COMMAND = (
    "netsh advfirewall firewall add rule name=Jumpstart dir=in action=allow"
    " protocol=TCP localport=8000 profile=any remoteip=localsubnet"
)


def _get_lan_ip() -> str:
    """Best-effort local network IP address.

    Standard trick: open a UDP "connection" to a public address (no
    packet is actually sent for UDP connect()) purely to ask the OS
    which local interface/IP it would route through. Falls back to
    127.0.0.1 if that fails (e.g. no network connection at all) --
    in that case only this machine itself can reach the interface.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def _get_wifi_ssid() -> str | None:
    """Name of the wifi network this machine is on, or None.

    None covers every "not on wifi as far as we can tell" case: no
    wireless adapter (typical for a desktop PC), a wireless adapter
    that isn't connected, not running on Windows, or netsh being
    unavailable. Only used to phrase the advice -- never to block
    startup, hence the broad except.

    Parsing note: Windows localises netsh's output, so the value of
    the "State"/"Status" line differs per language. The "SSID" label
    itself is not translated, so we match on that instead. The regex
    is anchored at the start of the line so it does not also match
    the "BSSID" line, and only matches when there is an actual value
    (a disconnected adapter prints no SSID line at all).
    """
    if not sys.platform.startswith("win"):
        return None
    try:
        result = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:  # noqa: BLE001 -- never let a check break startup
        return None
    match = re.search(r"^\s*SSID\s*:\s*(\S.*?)\s*$", result.stdout, re.MULTILINE)
    return match.group(1) if match else None


def _firewall_rule_exists() -> bool | None:
    """Whether the inbound firewall rule for this port is present.

    Returns True/False on Windows when the check ran, and None when it
    could not run (not Windows, netsh unavailable). Reading firewall
    rules does not need administrator rights, so this works on a normal
    double-click start.

    Deliberately a hint, not a verdict: this only looks for the rule
    Jumpstart_starten.bat creates by name. A machine can also be
    reachable through some other rule the user added, or with the
    firewall switched off entirely -- so False means "most likely
    explanation if a phone cannot connect", never "this is definitely
    blocked".
    """
    if not sys.platform.startswith("win"):
        return None
    try:
        result = subprocess.run(
            ["netsh", "advfirewall", "firewall", "show", "rule", f"name={FIREWALL_RULE_NAME}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:  # noqa: BLE001 -- never let a check break startup
        return None
    return result.returncode == 0


def _print_firewall_warning() -> None:
    print("  LET OP: de firewallregel van Jumpstart ontbreekt op deze computer.")
    print("  Zonder dit is het niet mogelijk om een verbinding met een telefoon")
    print("  te maken. Wilt u dit oplossen? Sluit dit venster, start")
    print("  Jumpstart_starten.bat opnieuw en kies JA bij het venster dat om")
    print("  beheerdersrechten vraagt. Of voer dit eenmalig uit in PowerShell")
    print("  als administrator:")
    print(f"    {FIREWALL_COMMAND}")


def _print_qr(url: str) -> None:
    if qrcode is None:
        print("  (Installeer 'qrcode' -- pip install qrcode -- voor een scanbare")
        print("   QR-code hier, in plaats van het adres over te typen.)")
        return
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make()
    qr.print_ascii(invert=True)


def _print_access_info() -> None:
    lan_ip = _get_lan_ip()
    local_url = f"http://127.0.0.1:{PORT}"

    print()
    print(LINE)
    print(f"  Op deze computer:  {local_url}")

    # --- Case 1: no network at all -------------------------------------
    if lan_ip == "127.0.0.1":
        print(LINE)
        print("  Geen netwerkverbinding gevonden. Alleen deze computer zelf kan")
        print("  de interface openen -- telefoon/tablet werkt nu niet.")
        print(LINE)
        print()
        return

    lan_url = f"http://{lan_ip}:{PORT}"
    subnet = lan_ip.rsplit(".", 1)[0]
    ssid = _get_wifi_ssid()

    print(f"  Telefoon/tablet:   {lan_url}")
    print(LINE)

    if _firewall_rule_exists() is False:
        _print_firewall_warning()
        print()

    # Bewust neutraal: geen conclusie over wifi of kabel (zie docstring).
    if ssid:
        print(f"  Wifi-netwerk van deze computer: {ssid}")
    print("  Telefoon/tablet gebruiken? Zet het apparaat op hetzelfde netwerk als")
    print("  deze computer, zet mobiele data even uit en scan deze QR-code:")
    print()
    _print_qr(lan_url)
    print("  Laadt de pagina op de telefoon niet? Kijk op de telefoon bij het")
    print(f"  IP-adres (instellingen -> wifi -> het netwerk). Dat hoort ook met {subnet}.")
    print("  te beginnen. Zo niet, dan zit de telefoon op een ander netwerk. Zo wel,")
    print("  dan laat dit netwerk geen verkeer tussen apparaten toe (gebruikelijk op")
    print("  bedrijfs-, ziekenhuis- en gastnetwerken); gebruik dan een hotspot van")
    print("  de telefoon en zet de computer ook op die hotspot.")
    print()
    print("  Dit is informatie, geen foutmelding. De server start hieronder; laat")
    print("  dit venster open zolang je Jumpstart gebruikt.")
    print(LINE)
    print()


def main() -> None:
    # 1. Startcontrole -- stopt hier met een duidelijke melding als er iets
    #    niet klopt.
    preflight.check_or_exit(PORT)

    # Pas na de controle importeren: als uvicorn ontbreekt, meldt de
    # controle dat al netjes.
    import uvicorn

    # 2. Adres + QR-code.
    _print_access_info()

    # 3. Werkmap = map Jumpstart, browser openen, server starten.
    os.chdir(APP_DIR)
    threading.Timer(
        3.0, webbrowser.open, args=(f"http://127.0.0.1:{PORT}",)
    ).start()
    uvicorn.run(
        "frontend_webapp.app:app",
        host=HOST,
        port=PORT,
        app_dir=str(APP_DIR),
    )

if __name__ == "__main__":
    main()
