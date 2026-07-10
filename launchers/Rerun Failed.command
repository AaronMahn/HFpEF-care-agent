#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  CARE-Agent — Re-run Failed Cells (and merge in place)
#  Double-click to retry ONLY the failed reasoning/safety cells
#  of your most recent LLM comparison, then merge the recovered
#  results back into the same files. Uses your saved OpenRouter
#  key and consumes a small amount of API credit (only failures).
# ─────────────────────────────────────────────────────────────
cd "$(dirname "$0")/.."

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

if [ ! -f ".venv/bin/python" ] || ! .venv/bin/python -c "import sys" &>/dev/null 2>&1; then
    echo "  ⚙  Building virtual environment..."
    rm -rf .venv && "$PYTHON" -m venv .venv
fi

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
echo "  🫀 CARE-Agent — Re-run Failed Cells"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Find the newest REAL comparison run. Excludes llm_mock_run_*/ by default and
# sorts by the timestamp baked into the folder name (so a recently-touched older
# real run can't get demoted by mtime). Set INCLUDE_MOCK=1 to opt mocks back in.
GLOBS="comparison_out/llm_run_*/llm_comparison.json
comparison_out/llm_run_*/llm_comparison_*.json
comparison_out/llm_comparison_*.json"
if [ "$INCLUDE_MOCK" = "1" ]; then
    GLOBS="$GLOBS
comparison_out/llm_mock_run_*/llm_comparison.json"
fi
# Print matching paths, drop backups, sort lexicographically (timestamps are
# YYYYMMDD_HHMMSS so lex == chrono), take last.
LATEST=$(printf '%s\n' $GLOBS | xargs -I{} ls {} 2>/dev/null \
        | grep -v '\.bak\.json$' | sort | tail -1)
if [ -z "$LATEST" ]; then
    echo "  ❌  No real comparison results found in comparison_out/."
    echo "      Run 'Run LLM Comparison.command' first, or"
    echo "      set INCLUDE_MOCK=1 to also consider mock runs."
    read -p "  Press Enter to close..."; exit 1
fi
echo "  Most recent REAL results:"
echo "      $LATEST"
echo ""

read -p "  Parallel workers (kept modest to avoid rate limits) [8]: " CC; CC=${CC:-8}

echo ""
echo "  ── Preview: which cells failed (no API calls yet) ──"
python rerun_failed.py --dry-run
echo ""
echo "  ⚠  This re-runs ONLY the failed cells live and merges them into the file"
echo "     above (the original is backed up to *.bak.json). Uses your credits."
echo "     (Make sure your OpenRouter key is saved in the app, or set OPENROUTER_API_KEY.)"
read -p "  Press Enter to re-run the failures, or close this window to cancel..."

python rerun_failed.py --concurrency "$CC"
RC=$?

if [ $RC -eq 0 ]; then
    # Re-run rewrote the HTML alongside the merged JSON. Open it.
    LATEST_HTML="${LATEST%.json}.html"
    [ -f "$LATEST_HTML" ] && { echo "  Opening updated report: $LATEST_HTML"; open "$LATEST_HTML"; }
fi

echo ""
read -p "  Done. Press Enter to close..."
