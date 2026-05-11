@echo off
REM save.bat - wrapper to run save_submission.py with arguments
setlocal

REM 取得此 .bat 所在資料夾
set "SCRIPT_DIR=%~dp0"

REM 如果設定了 PYTHON_HOME 就使用它
if defined PYTHON_HOME (
    set "PY_EXEC=%PYTHON_HOME%\python.exe"
) else (
    REM 嘗試在 PATH 找 python / python3
    for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY_EXEC set "PY_EXEC=%%P"
    for /f "delims=" %%P in ('where python3 2^>nul') do if not defined PY_EXEC set "PY_EXEC=%%P"
)

if not defined PY_EXEC (
    echo Python not found. Please install Python, add to PATH, or set PYTHON_HOME.
    exit /b 1
)

REM 呼叫 save_submission.py，%* 會把所有傳入參數傳給 Python 腳本
"%PY_EXEC%" "%SCRIPT_DIR%save_submission.py" %*
exit /b %ERRORLEVEL%