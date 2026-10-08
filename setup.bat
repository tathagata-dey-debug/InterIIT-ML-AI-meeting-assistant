@echo off
setlocal enabledelayedexpansion

echo ==================================================
echo   AI Meeting Assistant - Environment Setup (Windows)
echo   Inter IIT Tech Meet 15.0
echo ==================================================

:: 1. Check for Python 3.10+
set "PYTHON_CMD="

where python >nul 2>nul
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set "PY_VER=%%i"
    for /f "tokens=*" %%i in ('python -c "import sys; print(1 if sys.version_info >= (3, 10) else 0)"') do set "PY_OK=%%i"
    if "!PY_OK!"=="1" (
        set "PYTHON_CMD=python"
    )
)

if not defined PYTHON_CMD (
    where py >nul 2>nul
    if %errorlevel% equ 0 (
        for /f "tokens=*" %%i in ('py -3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set "PY_VER=%%i"
        for /f "tokens=*" %%i in ('py -3 -c "import sys; print(1 if sys.version_info >= (3, 10) else 0)"') do set "PY_OK=%%i"
        if "!PY_OK!"=="1" (
            set "PYTHON_CMD=py -3"
        )
    )
)

if not defined PYTHON_CMD (
    echo [X] Error: Python 3.10 or higher is required and was not found in PATH.
    echo     Please install Python 3.10+ from https://www.python.org/ and check 'Add to PATH'.
    exit /b 1
)

echo [v] Found Python !PY_VER! (!PYTHON_CMD!)

:: 2. Create virtual environment if it doesn't exist
if not exist "venv\Scripts\activate.bat" (
    echo [*] Creating virtual environment in .\venv ...
    !PYTHON_CMD! -m venv venv
    if %errorlevel% neq 0 (
        echo [X] Error creating virtual environment.
        exit /b 1
    )
    echo [v] Virtual environment created.
) else (
    echo [v] Virtual environment already exists.
)

:: 3. Activate venv & upgrade pip
echo [*] Activating virtual environment...
call venv\Scripts\activate.bat

echo [*] Upgrading pip...
python -m pip install --upgrade pip --quiet

:: 4. Install dependencies
echo [*] Installing dependencies from requirements.txt...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [X] Error installing dependencies.
    exit /b 1
)
echo [v] Dependencies installed successfully.

:: 5. Initialize credentials if not present
if not exist "keys\api_keys.json" (
    echo [*] Generating keys\api_keys.json template from example...
    if not exist "keys" mkdir keys
    copy "keys\api_keys.example.json" "keys\api_keys.json" >nul
    echo [!] Action required: Add your Gemini or OpenAI API key to keys\api_keys.json
) else (
    echo [v] keys\api_keys.json is already configured.
)

echo.
echo ==================================================
echo   Setup Complete! Next steps:
echo ==================================================
echo 1. Activate environment:  venv\Scripts\activate
echo 2. Add your API key to:   keys\api_keys.json
echo 3. Run interactive UI:    streamlit run app.py
echo 4. Run headless CLI:      python run_evaluation.py --audio samples\test_meeting.wav
echo 5. Run test suite:        pytest tests\ -v
echo ==================================================

endlocal
