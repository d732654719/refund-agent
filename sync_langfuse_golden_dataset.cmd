@echo off
setlocal
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python virtual environment not found: %PYTHON_EXE%
  exit /b 1
)
"%PYTHON_EXE%" -m agent_evaluations.sync_langfuse_golden_dataset %*
