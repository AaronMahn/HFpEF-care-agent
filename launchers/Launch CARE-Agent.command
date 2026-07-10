#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  CARE-Agent Launcher
#  Double-click this file to start the app in your browser.
#  Works on any Mac — auto-detects Python, rebuilds the venv
#  if broken, and installs missing packages automatically.
# ─────────────────────────────────────────────────────────────

# Go to the project folder (same directory as this script)
cd "$(dirname "$0")/.."

# ── 1. Find a working Python 3 (3.9 or newer) ─────────────────
find_python() {
    for cmd in python3.14 python3.13 python3.12 python3.11 python3.10 python3.9 python3 python; do
        if command -v "$cmd" &>/dev/null; then
            ver=$("$cmd" -c "import sys; print(sys.version_info >= (3,9))" 2>/dev/null)
            if [ "$ver" = "True" ]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON=$(find_python)
if [ -z "$PYTHON" ]; then
    echo ""
    echo "  ❌  Python 3.9 or newer not found."
    echo ""
    echo "  Install it with one of these commands:"
    echo "    brew install python@3.12"
    echo "    — or download from https://python.org"
    echo ""
    read -p "  Press Enter to close..."
    exit 1
fi

echo ""
echo "  ✓  Using Python: $($PYTHON --version 2>&1)  ($PYTHON)"

# ── 2. Check if .venv Python symlink resolves ─────────────────
VENV_OK=false
if [ -d ".venv" ] && [ -f ".venv/bin/activate" ]; then
    if ".venv/bin/python" -c "import sys" &>/dev/null 2>&1; then
        VENV_OK=true
    fi
fi

# ── 3. Rebuild .venv if the Python interpreter is broken ──────
if [ "$VENV_OK" = false ]; then
    echo ""
    echo "  ⚙  Virtual environment missing or broken — rebuilding..."
    rm -rf .venv
    "$PYTHON" -m venv .venv
    if [ $? -ne 0 ]; then
        echo "  ❌  Failed to create virtual environment."
        read -p "  Press Enter to close..."
        exit 1
    fi
    echo "  ✓  Virtual environment created."
fi

# ── 4. Check that all required packages are installed ─────────
REQUIRED="streamlit pandas numpy openpyxl scikit-learn lifelines openai httpx"
MISSING=""
for pkg in $REQUIRED; do
    # normalise: scikit-learn is importable as sklearn
    import_name="$pkg"
    [ "$pkg" = "scikit-learn" ] && import_name="sklearn"
    if ! .venv/bin/python -c "import $import_name" &>/dev/null 2>&1; then
        MISSING="$MISSING $pkg"
    fi
done

if [ -n "$MISSING" ]; then
    echo ""
    echo "  📦  Installing missing packages:$MISSING"
    echo "      (this takes ~1 minute on first run)"
    .venv/bin/pip install --upgrade pip --quiet
    .venv/bin/pip install $MISSING --quiet
    if [ $? -ne 0 ]; then
        echo "  ❌  Package installation failed."
        echo "      Check your internet connection and try again."
        read -p "  Press Enter to close..."
        exit 1
    fi
    echo "  ✓  Packages installed."
fi

# ── 5. Activate the venv ───────────────────────────────────────
source .venv/bin/activate

# ── 6. Kill any existing Streamlit on port 8501 ───────────────
lsof -ti tcp:8501 | xargs kill -9 2>/dev/null

# ── 7. Open browser after a short delay ───────────────────────
sleep 2 && open http://localhost:8501 &

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🫀 CARE-Agent — HFpEF Risk Pipeline"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Starting app at http://localhost:8501"
echo "  Close this window to stop the server."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── 8. Start Streamlit ─────────────────────────────────────────
streamlit run app.py --server.headless true
