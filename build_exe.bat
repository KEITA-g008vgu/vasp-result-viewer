@echo off
rem ============================================================
rem  VASP Result Viewer : build VASPResultViewer.exe
rem  (Windows + Python 3.14.  Run this file in the folder that
rem   contains vasp_viewer.py)
rem ============================================================
setlocal
cd /d "%~dp0"

if not exist .venv (
    py -m venv .venv || goto :err
)
call .venv\Scripts\activate.bat || goto :err

python -m pip install --upgrade pip
python -m pip install numpy spglib seekpath pillow matplotlib h5py PyOpenGL pyopengltk tkinterdnd2 pyinstaller || goto :err

rem  exe file icon (shown in Explorer): vasp_viewer.ico in this folder
set ICON_OPT=
if exist vasp_viewer.ico set ICON_OPT=--icon vasp_viewer.ico

pyinstaller --noconfirm --onefile --windowed --name VASPResultViewer %ICON_OPT% ^
    --collect-all tkinterdnd2 ^
    --collect-submodules OpenGL ^
    --collect-submodules seekpath ^
    --hidden-import PIL._tkinter_finder ^
    vasp_viewer.py || goto :err

echo.
echo Build finished:  dist\VASPResultViewer.exe
pause
exit /b 0

:err
echo.
echo Build failed.  See the messages above.
pause
exit /b 1
