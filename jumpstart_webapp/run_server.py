"""Convenience launcher for the Jumpstart web interface.

Wraps `uvicorn jumpstart_webapp.app:app --host 0.0.0.0 --port 8000` and,
before starting, works out how this machine can be reached and prints
that -- including a scannable QR code -- so starting the server is the
only step. No manually running `ipconfig` and typing an IP on a phone.

Run with:
    python -m jumpstart_webapp.run_server

This has the exact same effect as the plain uvicorn command documented
in the README/handleiding -- it just prints the access info up front.
If anything here fails (e.g. qrcode isn't installed), the plain
`uvicorn jumpstart_webapp.app:app --host 0.0.0.0 --port 8000` command
still works as a fallback.

Why the three different messages below
--------------------------------------
Whether a phone can actually reach this machine depends on the network,
and that is only partly knowable from here:

* We CAN see whether this machine is on wifi and which network, and
  which local address the server is reachable on.
* We CANNOT see whether the network allows devices to talk to each
  other at all. Corporate/hospital and guest networks routinely put
  wired and wireless clients on separate segments, and some routers
  isolate wifi clients from one another. Nothing local reveals that --
  only an actual connection attempt from a phone does.

So this deliberately does not promise that phone access works. On wifi
it shows the QR code plus how to check the phone if it doesn't load;
on a wired-only machine it says up front that a phone often cannot
reach a wired PC (but still shows the address, because on a simple
home network it usually does work).
"""

from __future__ import annotations

import re
import socket
import subprocess
import sys

import uvicorn

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
    print("  Dat is verreweg de meest voorkomende reden dat de pagina op een")
    print("  telefoon eindeloos blijft laden: Windows laat de verbinding dan")
    print("  stilletjes vallen, zonder foutmelding.")
    print("  Oplossing: sluit dit venster, start Jumpstart_starten.bat opnieuw")
    print("  en kies JA bij het venster dat om beheerdersrechten vraagt. Of voer")
    print("  dit eenmalig uit in PowerShell als administrator:")
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

    # --- Case 2: on wifi -- phone access is plausible -------------------
    if ssid:
        print(f"  Deze computer zit op wifi-netwerk: {ssid}")
        print("  Zorg dat de telefoon op datzelfde netwerk zit en zet mobiele")
        print("  data even uit, anders loopt het verkeer daarlangs.")
        print()
        _print_qr(lan_url)
        print("  Laadt de pagina niet? Kijk op de telefoon bij het IP-adres")
        print(f"  (instellingen -> wifi -> het netwerk): dat hoort ook met {subnet}.")
        print("  te beginnen. Doet het dat niet, dan zit de telefoon op een ander")
        print("  netwerk. Klopt het wel, dan staat dit netwerk onderling verkeer")
        print("  tussen apparaten niet toe -- gebruikelijk op bedrijfs- en")
        print("  gastnetwerken, en niet vanaf deze kant op te lossen.")
        print(LINE)
        print()
        return

    # --- Case 3: wired only -- phone access often does not work ---------
    print("  LET OP: deze computer zit niet op wifi maar op een vaste (bekabelde)")
    print("  verbinding. Een telefoon kan een bekabelde computer vaak niet")
    print("  bereiken -- zeker niet op bedrijfs- of instellingsnetwerken, waar")
    print("  bekabeld en draadloos bewust gescheiden zijn. Reken er dus niet op:")
    print("  gebruik de interface op deze computer zelf.")
    print()
    print("  Op een eenvoudig thuisnetwerk werkt het meestal wél. Wil je het")
    print(f"  proberen: het IP-adres van de telefoon moet dan ook met {subnet}.")
    print("  beginnen (instellingen -> wifi -> het netwerk). Zo ja, scan deze:")
    print()
    _print_qr(lan_url)
    print(LINE)
    print()


def main() -> None:
    _print_access_info()
    uvicorn.run("jumpstart_webapp.app:app", host=HOST, port=PORT)


if __name__ == "__main__":
    main()
