@echo off
rem Builds dist\PropSim.exe on Windows 10/11.
rem Needs Python 3.10-3.13 from python.org (tick "Add python.exe to PATH" when installing).
setlocal
cd /d "%~dp0"

where py >nul 2>nul && (set PY=py -3) || (set PY=python)

echo Creating build environment...
%PY% -m venv .buildenv || goto :fail
call .buildenv\Scripts\activate.bat || goto :fail

echo Installing packages (numpy, scipy, trimesh, gmsh, customtkinter, pyinstaller)...
python -m pip install --upgrade pip >nul
python -m pip install numpy scipy trimesh manifold3d gmsh customtkinter pyinstaller || goto :fail

echo Building PropSim.exe...
python build.py || goto :fail

echo.
echo Finished. Your app is dist\PropSim.exe
pause
exit /b 0

:fail
echo.
echo Build failed - see the messages above.
pause
exit /b 1
