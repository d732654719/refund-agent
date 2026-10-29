@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "DEEPEVAL_EXE=%PROJECT_DIR%.venv\Scripts\deepeval.exe"
set "DEEPEVAL_TELEMETRY_OPT_OUT=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if not exist "%DEEPEVAL_EXE%" (
    echo [ERROR] DeepEval executable not found: %DEEPEVAL_EXE%
    exit /b 1
)

pushd "%PROJECT_DIR%"
"%DEEPEVAL_EXE%" test run agent_evaluations --mark llm_eval --ignore-errors --tb=short %*
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo [REPORT] Structured JSON and offline HTML: %PROJECT_DIR%report\yyyy-mm-dd-xxx.json/html
echo [REPORT] Compatible full text: %PROJECT_DIR%agent_evaluations\latest_llm_evaluation_report.txt
popd

exit /b %EXIT_CODE%
