#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  CARE-Agent — Stratified Clinical Reports Launcher
#  Double-click to generate a bilingual HTML showing rule-based
#  reports for N patients per integrated risk tier (LOW / MOD /
#  HIGH / CRITICAL). No LLM, no API key needed — fully local.
# ─────────────────────────────────────────────────────────────
cd "$(dirname "$0")/.."

# ── 1. Find a working Python 3 (3.9+) ─────────────────────────
find_python() {
    for cmd in python3.14 python3.13 python3.12 python3.11 python3.10 python3.9 python3 python; do
        if command -v "$cmd" &>/dev/null; then
            ver=$("$cmd" -c "import sys; print(sys.version_info >= (3,9))" 2>/dev/null)
            [ "$ver" = "True" ] && { echo "$cmd"; return 0; }
        fi
    done
    return 1
}
PYTHON=$(find_python)
if [ -z "$PYTHON" ]; then
    echo "  ❌  Python 3.9+ not found. Install from https://python.org"
    read -p "  Press Enter to close..."; exit 1
fi

# ── 2. Reuse / rebuild the app's virtual environment ──────────
if [ ! -f ".venv/bin/python" ] || ! .venv/bin/python -c "import sys" &>/dev/null 2>&1; then
    echo "  ⚙  Building virtual environment..."
    rm -rf .venv && "$PYTHON" -m venv .venv
fi

# ── 3. Ensure required packages ───────────────────────────────
REQUIRED="pandas numpy openpyxl scikit-learn lifelines"
MISSING=""
for pkg in $REQUIRED; do
    import_name="$pkg"; [ "$pkg" = "scikit-learn" ] && import_name="sklearn"
    .venv/bin/python -c "import $import_name" &>/dev/null 2>&1 || MISSING="$MISSING $pkg"
done
if [ -n "$MISSING" ]; then
    echo "  📦  Installing missing packages:$MISSING"
    .venv/bin/pip install --upgrade pip --quiet
    .venv/bin/pip install $MISSING --quiet || { echo "  ❌  Install failed."; read -p "  Press Enter..."; exit 1; }
fi
source .venv/bin/activate

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🫀 CARE-Agent — Stratified Clinical Reports"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Picks N patients per integrated risk tier, runs Agents A→B→C"
echo "  (no LLM), writes JSON + CSV + bilingual HTML (EN / 中文)."
echo "  After the 18-month recalibration all four tiers are populated,"
echo "  so the report covers LOW / MODERATE / HIGH / CRITICAL."
echo ""

# ── 4. Options ────────────────────────────────────────────────
read -p "  Patients per tier [5]: " NP; NP=${NP:-5}
read -p "  Random seed [42]: " SD; SD=${SD:-42}

# ── 5. Run ────────────────────────────────────────────────────
echo ""
python stratified_clinical_reports.py --n-per-tier "$NP" --seed "$SD"
RC=$?

# ── 6. Open the report ────────────────────────────────────────
if [ $RC -eq 0 ]; then
    LATEST_DIR=$(ls -td comparison_out/stratified_run_* 2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ] && [ -f "$LATEST_DIR/stratified_reports.html" ]; then
        echo "  Run folder: $LATEST_DIR"
        open "$LATEST_DIR/stratified_reports.html"
    fi
fi

echo ""
read -p "  Done. Press Enter to close..."
