@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "PYTHON_EXE=%PROJECT_DIR%.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python executable not found: %PYTHON_EXE%
    exit /b 1
)

pushd "%PROJECT_DIR%"
"%PYTHON_EXE%" -m pytest -p no:deepeval agent_evaluations\test_deterministic_golden_cases.py -q --disable-warnings %*
set "EXIT_CODE=%ERRORLEVEL%"
popd

exit /b %EXIT_CODE%
