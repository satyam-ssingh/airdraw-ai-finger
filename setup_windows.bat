@echo off
setlocal

set FOUND=

py -3.12 -c "1" >nul 2>&1
if errorlevel 1 goto try311
set FOUND=3.12
goto found

:try311
py -3.11 -c "1" >nul 2>&1
if errorlevel 1 goto try310
set FOUND=3.11
goto found

:try310
py -3.10 -c "1" >nul 2>&1
if errorlevel 1 goto try39
set FOUND=3.10
goto found

:try39
py -3.9 -c "1" >nul 2>&1
if errorlevel 1 goto notfound
set FOUND=3.9
goto found

:notfound
echo.
echo ============================================================
echo  No compatible Python found. Need 3.9, 3.10, 3.11 or 3.12.
echo  MediaPipe does not yet support Python 3.13 or 3.14.
echo.
echo  Please install Python 3.12 from:
echo    https://www.python.org/downloads/release/python-3120/
echo  Check the box "Add python.exe to PATH" during install.
echo  Then re-run this script.
echo ============================================================
pause
exit /b 1

:found
echo Found compatible Python %FOUND% - creating virtual environment...
py -%FOUND% -m venv venv
if errorlevel 1 goto venvfail

call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ============================================================
echo  Setup complete! Using Python %FOUND% inside venv folder.
echo  To run AirDraw next time:
echo      venv\Scripts\activate
echo      python airdraw.py
echo ============================================================
pause
exit /b 0

:venvfail
echo.
echo Failed to create the virtual environment. See error above.
pause
exit /b 1
