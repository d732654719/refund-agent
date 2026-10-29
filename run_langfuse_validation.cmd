@echo off
setlocal EnableDelayedExpansion

set "PROJECT_DIR=%~dp0"
set "PYTHON_EXE=%PROJECT_DIR%.venv\Scripts\python.exe"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python executable not found: %PYTHON_EXE%
    exit /b 1
)

for %%V in (LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY LANGFUSE_BASE_URL LANGFUSE_TRACING_ENVIRONMENT LANGFUSE_RELEASE) do (
    if "!%%V!"=="" (
        echo [ERROR] %%V is not set in this terminal.
        exit /b 1
    )
)

pushd "%PROJECT_DIR%"
"%PYTHON_EXE%" -m agent_evaluations.validate_langfuse_traces
set "EXIT_CODE=%ERRORLEVEL%"
popd

exit /b %EXIT_CODE%
