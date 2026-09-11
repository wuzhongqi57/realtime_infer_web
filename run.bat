@echo off
setlocal

title IR Realtime Infer Web

cd /d "%~dp0"

rem ============================================================
rem  Usage:
rem    run.bat                          default (cam 0, cpu, port 5001)
rem    run.bat --device cuda            GPU inference (needs CUDA torch)
rem    run.bat --cam 1 --port 5002      other camera / port
rem    run.bat --source video:demo.avi  simulate camera with a video
rem ============================================================
set CAM=0
set DEVICE=cpu
set PORT=5001
set MODELS=
set SOURCE=

:parse_args
if "%~1"=="" goto args_done
if "%~1"=="--cam"        ( set "CAM=%~2"     & shift & shift & goto parse_args )
if "%~1"=="--device"     ( set "DEVICE=%~2"  & shift & shift & goto parse_args )
if "%~1"=="--port"       ( set "PORT=%~2"    & shift & shift & goto parse_args )
if "%~1"=="--models-dir" ( set "MODELS=%~2"  & shift & shift & goto parse_args )
if "%~1"=="--source"     ( set "SOURCE=%~2"  & shift & shift & goto parse_args )
shift
goto parse_args
:args_done

echo ================================================
echo   IR Realtime Inference Web App
echo ================================================
echo   Camera : %CAM%
echo   Device : %DEVICE%
echo   Port   : %PORT%
if defined MODELS echo   Models : %MODELS%
if defined SOURCE echo   Source : %SOURCE%
echo   URL    : http://127.0.0.1:%PORT%
echo   Press Ctrl+C to stop
echo ================================================
echo.

rem ---- activate conda env (prefer direct activate.bat, fall back to PATH conda) ----
set CONDA_ACT=C:\Users\Administrator\miniconda3\Scripts\activate.bat
if exist "%CONDA_ACT%" (
    call "%CONDA_ACT%" ir_realtime >nul 2>&1
    if errorlevel 1 goto direct_python
    set PYTHON=python
    goto launch
)
call conda activate ir_realtime >nul 2>&1
if errorlevel 1 goto direct_python
set PYTHON=python
goto launch

:direct_python
echo [info] conda activate failed, trying direct path...
set PYTHON=C:\Users\Administrator\miniconda3\envs\ir_realtime\python.exe
if not exist "%PYTHON%" (
    echo [error] ir_realtime env not found. Create it first:
    echo   conda create -n ir_realtime python=3.12 -y
    echo   conda run -n ir_realtime pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
    pause
    exit /b 1
)

:launch
rem ---- GPU check ----
if "%DEVICE%"=="cuda" (
    "%PYTHON%" -c "import torch; assert torch.cuda.is_available(), 'cuda unavailable'" 2>nul
    if errorlevel 1 (
        echo [error] CUDA not available. Install CUDA torch first:
        echo   conda run -n ir_realtime pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch
        pause
        exit /b 1
    )
)

rem ---- open browser after short delay ----
start "" /b cmd /c "ping -n 2 127.0.0.1 >nul & start http://127.0.0.1:%PORT%"

rem ---- build optional args ----
set MODELS_ARG=
if defined MODELS set MODELS_ARG=--models-dir "%MODELS%"
set SOURCE_ARG=
if defined SOURCE set SOURCE_ARG=--source "%SOURCE%"

echo [start] %PYTHON% app.py --cam %CAM% --device %DEVICE% --port %PORT% %MODELS_ARG% %SOURCE_ARG%
echo.
"%PYTHON%" app.py --cam %CAM% --device %DEVICE% --port %PORT% %MODELS_ARG% %SOURCE_ARG%

echo.
echo Server stopped.
pause
