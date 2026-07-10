#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  CARE-Agent — Build Doctor Rating Sheets (Human Eval)
#  Double-click to generate three blinded HTML rating sheets
#  from the most recent LLM-comparison run, plus the private
#  un-blinding maps used at aggregation time.
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
    echo "  ❌  Python 3.9+ not found."
    read -p "  Press Enter to close..."; exit 1
fi

if [ ! -f ".venv/bin/python" ]; then
    echo "  ⚙  Building virtual environment..."
    "$PYTHON" -m venv .venv
fi
source .venv/bin/activate
.venv/bin/pip install --quiet pandas numpy openpyxl lifelines 2>/dev/null

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🩺 CARE-Agent — Build Doctor Rating Sheets"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Generates ONE blinded HTML rating sheet per doctor"
echo "  from the latest LLM comparison run."
echo ""

# Pick the newest LLM run folder
LATEST_DIR=$(ls -td comparison_out/llm_run_* comparison_out/llm_mock_run_* 2>/dev/null | head -1)
if [ -z "$LATEST_DIR" ]; then
    echo "  ❌  No comparison_out/llm_run_* or llm_mock_run_* folder found."
    echo "      Run the LLM Comparison launcher first."
    read -p "  Press Enter to close..."; exit 1
fi
echo "  Source run : $LATEST_DIR"
echo ""

read -p "  Doctor 1 name [医生1]: " D1; D1=${D1:-医生1}
read -p "  Doctor 2 name [医生2]: " D2; D2=${D2:-医生2}
read -p "  Doctor 3 name [医生3]: " D3; D3=${D3:-医生3}
read -p "  Sheet language en/zh [zh]: " LG; LG=${LG:-zh}

echo ""
python build_rating_sheets.py --llm-run "$LATEST_DIR" --doctors "$D1" "$D2" "$D3" --lang "$LG"
RC=$?

# Open the output folder in Finder + open the first sheet for the operator to spot-check
if [ $RC -eq 0 ]; then
    RATINGS_DIR="$LATEST_DIR/ratings"
    [ -d "$RATINGS_DIR" ] && open "$RATINGS_DIR"
    FIRST=$(ls "$RATINGS_DIR"/rating_sheet_*.html 2>/dev/null | head -1)
    [ -n "$FIRST" ] && open "$FIRST"
fi

echo ""
echo "  Now: email each doctor their own rating_sheet_*.html file."
echo "  When they email back the exported JSON, drop the JSONs into"
echo "      $LATEST_DIR/ratings/returned/"
echo "  then double-click 'Aggregate Ratings.command' to generate the leaderboard."
echo ""
read -p "  Done. Press Enter to close..."
