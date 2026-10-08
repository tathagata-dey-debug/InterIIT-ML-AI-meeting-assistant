#!/usr/bin/env bash
set -e

echo "=================================================="
echo "  AI Meeting Assistant - Environment Setup (macOS/Linux)"
echo "  Inter IIT Tech Meet 15.0"
echo "=================================================="

# 1. Check for Python 3.10+
PYTHON_BIN=""
for candidate in python3 python python3.11 python3.12 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then
        MAJOR=$("$candidate" -c 'import sys; print(sys.version_info.major)')
        MINOR=$("$candidate" -c 'import sys; print(sys.version_info.minor)')
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON_BIN="$candidate"
            VER=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            echo "[✓] Found Python $VER ($candidate)"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "[✗] Error: Python 3.10 or higher is required."
    echo "    Please install Python 3.10+ and re-run this script."
    exit 1
fi

# 2. Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[*] Creating virtual environment in ./venv ..."
    "$PYTHON_BIN" -m venv venv
    echo "[✓] Virtual environment created."
else
    echo "[✓] Virtual environment already exists."
fi

# 3. Activate venv & upgrade pip
echo "[*] Activating virtual environment..."
# shellcheck source=/dev/null
source venv/bin/activate

echo "[*] Upgrading pip..."
pip install --upgrade pip --quiet

# 4. Install dependencies
echo "[*] Installing dependencies from requirements.txt..."
pip install -r requirements.txt --quiet
echo "[✓] Dependencies installed."

# 5. Initialize credentials if not present
if [ ! -f "keys/api_keys.json" ]; then
    echo "[*] Generating keys/api_keys.json template from example..."
    mkdir -p keys
    cp keys/api_keys.example.json keys/api_keys.json
    echo "[!] Action required: Add your Gemini / OpenAI API key to keys/api_keys.json"
else
    echo "[✓] keys/api_keys.json is already configured."
fi

echo ""
echo "=================================================="
echo "  Setup Complete! Next steps:"
echo "=================================================="
echo "1. Activate environment: source venv/bin/activate"
echo "2. Add your API key to: keys/api_keys.json"
echo "3. Run interactive UI:   streamlit run app.py"
echo "4. Run headless CLI:     python run_evaluation.py --audio samples/test_meeting.wav"
echo "5. Run test suite:       pytest tests/ -v"
echo "=================================================="
