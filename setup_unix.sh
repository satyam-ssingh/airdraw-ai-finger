#!/usr/bin/env bash
# =============================================================================
# AirDraw setup for macOS / Linux
# MediaPipe currently only supports Python 3.9-3.12. If your default
# `python3` is 3.13/3.14, this script finds a compatible interpreter and
# builds an isolated virtual environment, without touching your system Python.
# =============================================================================
set -e

FOUND=""
for v in 3.12 3.11 3.10 3.9; do
    if command -v "python$v" >/dev/null 2>&1; then
        FOUND="python$v"
        break
    fi
done

if [ -z "$FOUND" ]; then
    echo "============================================================"
    echo " No compatible Python found (need 3.9, 3.10, 3.11 or 3.12)."
    echo " MediaPipe does not yet support Python 3.13/3.14."
    echo ""
    echo " Install one, e.g.:"
    echo "   macOS (Homebrew):  brew install python@3.12"
    echo "   Ubuntu/Debian:     sudo apt install python3.12 python3.12-venv"
    echo " Then re-run this script."
    echo "============================================================"
    exit 1
fi

echo "Found compatible interpreter: $FOUND - creating virtual environment..."
"$FOUND" -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "============================================================"
echo " Setup complete! Using $FOUND inside ./venv"
echo " To run AirDraw:"
echo "     source venv/bin/activate"
echo "     python airdraw.py"
echo "============================================================"
