@echo off
setlocal EnableExtensions
rem ===================================================================
rem  Jumpstart starten - dubbelklik dit bestand.
rem
rem  De eerste keer installeert uv (meegeleverd in Jumpstart\tools) een
rem  eigen Python 3.12 en precies de vastgelegde packages (pyproject.toml
rem  + uv.lock) in %USERPROFILE%\.jumpstart. Een Python die al op de
rem  computer staat (Microsoft Store, Anaconda, ...) wordt bewust NIET
rem  gebruikt. Daarna start dit bestand alleen nog de server.
rem ===================================================================

set "HERE=%~dp0"
set "APPDIR=%~dp0Jumpstart"
set "UV=%APPDIR%\tools\uv.exe"

rem --- Controle 1: niet rechtstreeks vanuit een zip gestart? ---
rem Dubbelklik je de .bat binnen een zip, dan pakt Windows alleen dit ene
rem bestand uit naar een tijdelijke map en ontbreekt de rest.
if /i not "%HERE:\AppData\Local\Temp\=%"=="%HERE%" goto :from_zip

rem --- Controle 2: is de map compleet? ---
if not exist "%APPDIR%\frontend_webapp\app.py" goto :incomplete
if not exist "%APPDIR%\backend_script\__init__.py" goto :incomplete
if not exist "%APPDIR%\pyproject.toml" goto :incomplete
if not exist "%APPDIR%\uv.lock" goto :incomplete
if not exist "%UV%" goto :incomplete

rem --- Alles van Jumpstart komt buiten OneDrive, in je eigen gebruikersmap ---
set "JS_HOME=%USERPROFILE%\.jumpstart"
set "UV_PYTHON_INSTALL_DIR=%JS_HOME%\python"
set "UV_PROJECT_ENVIRONMENT=%JS_HOME%\env"
set "UV_CACHE_DIR=%JS_HOME%\cache"
rem Alleen de eigen Python van Jumpstart, nooit een andere op deze computer.
set "UV_PYTHON_PREFERENCE=only-managed"
rem Certificaten van Windows gebruiken (nodig op bedrijfs-/ziekenhuisnetwerken).
set "UV_SYSTEM_CERTS=true"
set "UV_LINK_MODE=copy"
set "ENVPY=%JS_HOME%\env\Scripts\python.exe"

cd /d "%APPDIR%"

if not exist "%ENVPY%" goto :first_install_message
echo [Jumpstart] Installatie controleren...
goto :sync

:first_install_message
echo.
echo [Jumpstart] Eerste keer: Python en alle onderdelen worden geinstalleerd in
echo [Jumpstart]   %JS_HOME%
echo [Jumpstart] Dit duurt een paar minuten en heeft internet nodig. Daarna gaat
echo [Jumpstart] opstarten binnen enkele seconden, ook zonder internet.
echo.

:sync
"%UV%" sync --locked
if not errorlevel 1 goto :run
if not exist "%ENVPY%" goto :install_failed
echo.
echo [Jumpstart] LET OP: bijwerken is niet gelukt - geen internet? De bestaande
echo [Jumpstart] installatie wordt gebruikt.
echo.

:run
powershell -NoProfile -Command "$d=[Environment]::GetFolderPath('Desktop'); $p=Join-Path $d 'Jumpstart.lnk'; if (-not (Test-Path $p)) { $s=(New-Object -ComObject WScript.Shell).CreateShortcut($p); $s.TargetPath='%~dp0Jumpstart_starten.bat'; $s.WorkingDirectory='%~dp0'; $s.IconLocation='%APPDIR%\jumpstart_icon.ico'; $s.Save() }" >nul 2>nul

call :ensure_firewall_rule

echo [Jumpstart] Server wordt gestart - de browser opent vanzelf.
echo [Jumpstart] Laat dit venster open zolang je Jumpstart gebruikt.
echo [Jumpstart] Sluiten stopt de server.
echo.

"%ENVPY%" -m frontend_webapp.run_server

echo.
echo [Jumpstart] De server is gestopt.
pause
exit /b 0

:from_zip
echo.
echo [Jumpstart] Dit bestand is rechtstreeks vanuit een zip-bestand gestart.
echo [Jumpstart] Dat werkt niet: Windows pakt dan alleen dit ene bestand uit.
echo [Jumpstart] Doe dit: rechtermuisknop op de zip -^> "Alles uitpakken...",
echo [Jumpstart] open de uitgepakte map en dubbelklik daar Jumpstart_starten.bat.
pause
exit /b 1

:incomplete
echo.
echo [Jumpstart] De map is niet compleet. Naast dit bestand hoort de map
echo [Jumpstart] "Jumpstart" te staan, met daarin onder meer frontend_webapp,
echo [Jumpstart] backend_script, pyproject.toml, uv.lock en tools\uv.exe.
echo [Jumpstart] Dit bestand staat nu in:
echo [Jumpstart]   %HERE%
echo [Jumpstart] Download de hele map als zip ^(GitHub: Code -^> Download ZIP^),
echo [Jumpstart] pak hem uit en start Jumpstart_starten.bat vanuit de uitgepakte map.
pause
exit /b 1

:install_failed
echo.
echo [Jumpstart] Installeren is niet gelukt - zie de melding hierboven.
echo [Jumpstart] Mogelijke oorzaken: geen of onstabiele internetverbinding, een
echo [Jumpstart] netwerk dat downloads blokkeert, beveiliging die tools\uv.exe
echo [Jumpstart] tegenhoudt, of te weinig schijfruimte ^(nodig: ongeveer 3 GB^).
echo [Jumpstart] Los dat op en start dit bestand opnieuw - het gaat verder waar
echo [Jumpstart] het gebleven was.
pause
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
