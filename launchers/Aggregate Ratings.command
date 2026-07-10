#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  CARE-Agent — Aggregate Doctor Ratings into a Leaderboard
#  Double-click after all three doctors have returned their
#  exported JSON files. Make sure those JSONs are inside:
#      <llm-run>/ratings/returned/
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
if [ ! -f ".venv/bin/python" ]; then "$PYTHON" -m venv .venv; fi
source .venv/bin/activate
.venv/bin/pip install --quiet pandas numpy openpyxl lifelines 2>/dev/null

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  📊 CARE-Agent — Aggregate Doctor Ratings"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
LATEST_DIR=$(ls -td comparison_out/llm_run_* comparison_out/llm_mock_run_* 2>/dev/null | head -1)
if [ -z "$LATEST_DIR" ]; then
    echo "  ❌  No comparison_out/llm_run_* folder found."
    read -p "  Press Enter to close..."; exit 1
fi
RET="$LATEST_DIR/ratings/returned"
mkdir -p "$RET"
N=$(ls "$RET"/*.json 2>/dev/null | wc -l | tr -d ' ')
echo "  Source run    : $LATEST_DIR"
echo "  Returned JSONs: $N file(s) in $RET"
if [ "$N" -eq 0 ]; then
    echo ""
    echo "  ⚠  No doctor JSON files found yet. Drop them in:"
    echo "        $RET"
    open "$RET"
    read -p "  Press Enter to close..."; exit 1
fi
echo ""

python aggregate_human_ratings.py --llm-run "$LATEST_DIR"
RC=$?

if [ $RC -eq 0 ]; then
    LATEST_AGG=$(ls -td "$LATEST_DIR"/ratings/aggregate_* 2>/dev/null | head -1)
    [ -n "$LATEST_AGG" ] && [ -f "$LATEST_AGG/leaderboard.html" ] && open "$LATEST_AGG/leaderboard.html"
fi

echo ""
read -p "  Done. Press Enter to close..."
