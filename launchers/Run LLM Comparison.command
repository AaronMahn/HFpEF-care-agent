#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  CARE-Agent — LLM Output Comparison Launcher
#  Double-click to capture the EXACT reasoning + safety-flag
#  outputs of each flagship model, side by side.
#  Uses your saved OpenRouter key and consumes API credits.
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
REQUIRED="pandas numpy openpyxl scikit-learn lifelines openai httpx"
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
echo "  🫀 CARE-Agent — LLM Output Comparison (stratified)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Flagship models: Claude · Gemini · GPT · DeepSeek · Kimi · Qwen"
echo "  Selection: N patients per integrated risk tier"
echo "             (LOW / MODERATE / HIGH / CRITICAL),"
echo "             quantile-spaced by Cox 18-month probability."
echo "  After the 18-month recalibration all four tiers are populated,"
echo "  so the report covers LOW / MODERATE / HIGH / CRITICAL."
echo ""

# ── 4. Options ────────────────────────────────────────────────
read -p "  Patients per tier [5]: " NP; NP=${NP:-5}
read -p "  Output language en/zh [en]: " LG; LG=${LG:-en}
read -p "  Auto-retry failed cells afterwards? [Y/n]: " AR; AR=${AR:-Y}
if [ "$AR" = "y" ] || [ "$AR" = "Y" ]; then RETRY="--auto-retry 2"; else RETRY=""; fi

# Estimate: 4 tiers × N patients × 6 models × 2 calls (reasoning + safety).
# All four tiers are populated after the 18-month recalibration.
EST=$((NP*4*6*2))

# ── 5. Free preview (no API calls) ────────────────────────────
echo ""
echo "  ── Preview (no API calls made yet) ──"
python compare_llm_outputs.py --dry-run --n-per-tier "$NP" --lang "$LG"

echo ""
echo "  ⚠  The live run calls OpenRouter and uses your credits (~$EST calls)."
echo "     (Make sure your OpenRouter key is saved in the app, or set OPENROUTER_API_KEY.)"
read -p "  Press Enter to run live, or close this window to cancel..."

# ── 6. Live run ───────────────────────────────────────────────
python compare_llm_outputs.py --n-per-tier "$NP" --lang "$LG" $RETRY
RC=$?

# ── 7. Open the reports ───────────────────────────────────────
if [ $RC -eq 0 ]; then
    LATEST_DIR=$(ls -td comparison_out/llm_run_* comparison_out/llm_mock_run_* 2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ]; then
        echo "  Run folder: $LATEST_DIR"
        [ -f "$LATEST_DIR/llm_comparison.html" ]                && open "$LATEST_DIR/llm_comparison.html"
        [ -f "$LATEST_DIR/rule_based_reports_bilingual.html" ]  && open "$LATEST_DIR/rule_based_reports_bilingual.html"
    fi
fi

echo ""
read -p "  Done. Press Enter to close..."
