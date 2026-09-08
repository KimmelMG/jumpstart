@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "VENVDIR=%USERPROFILE%\.jumpstart\venv"
set "INSTALL_MARKER=%VENVDIR%\.install_ok"

if exist "%INSTALL_MARKER%" if exist "%VENVDIR%\Scripts\python.exe" goto :run

echo.
echo [Jumpstart] Geen complete virtuele omgeving gevonden - installatie wordt (opnieuw) gestart.
echo [Jumpstart] Dit duurt de eerste keer een paar minuten ^(mediapipe/opencv/ultralytics zijn grote packages^).
echo [Jumpstart] De virtuele omgeving komt te staan in:
echo [Jumpstart]   %VENVDIR%
echo [Jumpstart] ^(buiten je OneDrive-map en buiten de map van de Microsoft Store-Python,
echo [Jumpstart]  om synchronisatie- en padproblemen te vermijden^)
echo.

call :find_python
if not defined PYEXE call :install_python_via_winget
if not defined PYEXE goto :no_python

echo [Jumpstart] Python gevonden: %PYDESC%
if defined STORE_PYTHON_ONLY (
    echo.
    echo [Jumpstart] LET OP: alleen de Microsoft Store-versie van Python is gevonden.
    echo [Jumpstart] Dat kan werken, maar gaf op dit type installatie eerder problemen.
    echo [Jumpstart] Loopt de installatie hieronder vast, installeer dan Python via
    echo [Jumpstart]   https://www.python.org/downloads/windows/
    echo [Jumpstart] ^(vink "Add python.exe to PATH" aan tijdens installatie^) en start
    echo [Jumpstart] dit bestand daarna opnieuw.
)
echo.

if exist "%INSTALL_MARKER%" del /f /q "%INSTALL_MARKER%" >nul 2>nul

echo [Jumpstart] Virtuele omgeving aanmaken...
"%PYEXE%" %PYARGS% -m venv "%VENVDIR%"
if errorlevel 1 goto :venv_failed

if not exist "%VENVDIR%\Scripts\python.exe" goto :venv_redirected

set "VENVPY=%VENVDIR%\Scripts\python.exe"

echo [Jumpstart] pip bijwerken...
"%VENVPY%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [Jumpstart] Mislukt, nieuwe poging...
    "%VENVPY%" -m pip install --upgrade pip
    if errorlevel 1 goto :install_failed
)

echo [Jumpstart] Certificaatfix installeren ^(voorkomt SSL-fouten op bedrijfsnetwerken^)...
"%VENVPY%" -m pip install pip-system-certs
if errorlevel 1 (
    echo [Jumpstart] Mislukt, nieuwe poging...
    "%VENVPY%" -m pip install pip-system-certs
    if errorlevel 1 goto :install_failed
)

echo [Jumpstart] Webinterface-packages installeren...
"%VENVPY%" -m pip install -r "%~dp0requirements-web.txt"
if errorlevel 1 (
    echo [Jumpstart] Mislukt, nieuwe poging...
    "%VENVPY%" -m pip install -r "%~dp0requirements-web.txt"
    if errorlevel 1 goto :install_failed
)

echo [Jumpstart] Analyse- en pose-detectiepackages installeren - dit duurt het langst...
"%VENVPY%" -m pip install -r "%~dp0requirements-analysis.txt"
if errorlevel 1 (
    echo [Jumpstart] Mislukt, nieuwe poging...
    "%VENVPY%" -m pip install -r "%~dp0requirements-analysis.txt"
    if errorlevel 1 goto :install_failed
)

echo Y> "%INSTALL_MARKER%"

echo.
echo [Jumpstart] Installatie voltooid.
echo.
goto :run

:no_python
echo [Jumpstart] Geen Python gevonden op dit systeem en automatisch
echo [Jumpstart] installeren is niet gelukt ^(of winget ontbreekt op deze pc^).
echo [Jumpstart] Installeer Python 3 handmatig via python.org en probeer het
echo [Jumpstart] daarna opnieuw:
echo [Jumpstart]   https://www.python.org/downloads/windows/
echo [Jumpstart] ^(vink "Add python.exe to PATH" aan tijdens installatie^)
pause
exit /b 1

:venv_failed
echo.
echo [Jumpstart] Aanmaken van de virtuele omgeving is mislukt - zie de foutmelding hierboven.
pause
exit /b 1

:venv_redirected
echo.
echo [Jumpstart] Aanmaken van de virtuele omgeving is "gelukt" volgens Python, maar
echo [Jumpstart]   %VENVDIR%\Scripts\python.exe
echo [Jumpstart] bestaat niet. Windows heeft de map stiekem omgeleid naar een andere
echo [Jumpstart] locatie - dit gebeurt met de Microsoft Store-versie van Python.
echo [Jumpstart] Installeer Python via python.org
echo [Jumpstart]   https://www.python.org/downloads/windows/
echo [Jumpstart] ^(vink "Add python.exe to PATH" aan tijdens installatie^), verwijder
echo [Jumpstart] daarna de map "%VENVDIR%" en start dit bestand opnieuw.
pause
exit /b 1

:install_failed
echo.
echo [Jumpstart] Installeren van packages is mislukt - zie de foutmelding hierboven.
echo [Jumpstart] Mogelijke oorzaken: geen/onstabiele internetverbinding, een strenge
echo [Jumpstart] firewall/proxy, of te weinig schijfruimte. Los dat op en start dit
echo [Jumpstart] bestand daarna gewoon opnieuw - een mislukte installatie wordt
echo [Jumpstart] automatisch hervat.
pause
exit /b 1

:run
call :ensure_firewall_rule

echo [Jumpstart] Server wordt gestart...
echo [Jumpstart] Dit venster laat de voortgang/logs zien - laat het openstaan
echo [Jumpstart] zolang je de webinterface gebruikt. Sluiten stopt de server.
echo.

start "" cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8000"

"%VENVDIR%\Scripts\python.exe" -m jumpstart_webapp.run_server

echo.
echo [Jumpstart] De server is gestopt.
pause
exit /b 0

:install_python_via_winget
where winget >nul 2>nul
if errorlevel 1 exit /b 1

echo [Jumpstart] Geen Python gevonden - wordt automatisch geinstalleerd via
echo [Jumpstart] winget ^(Windows Package Manager^). Dit duurt een paar
echo [Jumpstart] minuten en werkt zonder adminrechten...
winget install --id Python.Python.3.11 -e --scope user --silent --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo [Jumpstart] Automatisch installeren via winget is niet gelukt.
    exit /b 1
)

call :find_python
if not defined PYEXE exit /b 1

echo [Jumpstart] Python automatisch geinstalleerd: %PYDESC%
echo.
exit /b 0

:find_python
set "PYEXE="
set "PYARGS="
set "PYDESC="
set "STORE_PYTHON_ONLY="

where py >nul 2>nul
if not errorlevel 1 (
    set "PYEXE=py"
    set "PYARGS=-3"
    set "PYDESC=py -3 ^(Python Launcher for Windows^)"
    exit /b 0
)

for /f "delims=" %%P in ('where python 2^>nul') do (
    echo %%P | findstr /i "WindowsApps" >nul
    if errorlevel 1 (
        set "PYEXE=%%P"
        set "PYDESC=%%P"
        exit /b 0
    )
)

for %%C in (
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python39\python.exe"
    "%USERPROFILE%\anaconda3\python.exe"
    "%USERPROFILE%\miniconda3\python.exe"
    "%ProgramData%\Anaconda3\python.exe"
    "%ProgramData%\miniconda3\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
    "C:\Python310\python.exe"
) do (
    if exist "%%~C" (
        set "PYEXE=%%~C"
        set "PYDESC=%%~C"
        exit /b 0
    )
)

where python >nul 2>nul
if not errorlevel 1 (
    set "PYEXE=python"
    set "PYDESC=python ^(Microsoft Store-versie^)"
    set "STORE_PYTHON_ONLY=1"
    exit /b 0
)

exit /b 1

:ensure_firewall_rule
netsh advfirewall firewall show rule name="Jumpstart" >nul 2>nul
if not errorlevel 1 exit /b 0

echo.
echo [Jumpstart] Windows Firewall laat standaard geen verbinding toe vanaf
echo [Jumpstart] een telefoon of tablet. Dat wordt nu eenmalig geregeld.
echo [Jumpstart] Er verschijnt zo een venster van Windows dat om
echo [Jumpstart] beheerdersrechten vraagt - kies JA.
echo [Jumpstart] Kies je NEE, dan start Jumpstart gewoon door, maar werkt
echo [Jumpstart] alleen deze computer zelf en geen telefoon of tablet.
echo.

powershell -NoProfile -Command "try { Start-Process netsh -ArgumentList 'advfirewall','firewall','add','rule','name=Jumpstart','dir=in','action=allow','protocol=TCP','localport=8000','profile=any','remoteip=localsubnet' -Verb RunAs -Wait -WindowStyle Hidden } catch { exit 1 }" >nul 2>nul

netsh advfirewall firewall show rule name="Jumpstart" >nul 2>nul
if errorlevel 1 goto :firewall_not_added

echo [Jumpstart] Gelukt - telefoon en tablet kunnen nu verbinden.
echo.
exit /b 0

:firewall_not_added
echo [Jumpstart] De firewallregel is niet aangemaakt. Jumpstart werkt nu
echo [Jumpstart] alleen op deze computer zelf. Wil je het later alsnog
echo [Jumpstart] regelen: start PowerShell als administrator en plak deze
echo [Jumpstart] regel:
echo [Jumpstart]   netsh advfirewall firewall add rule name=Jumpstart dir=in action=allow protocol=TCP localport=8000 profile=any remoteip=localsubnet
echo.
exit /b 0
