#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  CARE-Agent — LLM Benchmark Launcher
#  Double-click to score the flagship/shortlist models on the
#  agent task (success, latency, tokens, JSON compliance, and an
#  optional clinical-quality judge) and build an HTML leaderboard.
#  Uses your saved OpenRouter key and consumes API credits.
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
echo "  🫀 CARE-Agent — LLM Benchmark"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Mode: score saved comparison results, or run live ─────────────────────────
LATEST_CMP=$(ls -t comparison_out/llm_comparison_*.json 2>/dev/null | head -1)
MODE="live"
if [ -n "$LATEST_CMP" ]; then
    echo "  Found saved comparison results:"
    echo "      $LATEST_CMP"
    read -p "  Score those WITHOUT re-running the models (much cheaper)? [Y/n]: " SC
    case "$SC" in [Nn]*) MODE="live";; *) MODE="from";; esac
    echo ""
fi

read -p "  Include clinical-quality judge? [Y/n]: " JG
JUDGE_FLAG=""; JUDGE_ON=1
case "$JG" in [Nn]*) JUDGE_FLAG="--no-judge"; JUDGE_ON=0;; esac

if [ "$MODE" = "from" ]; then
    echo ""
    echo "  ── Preview (operational metrics are reused for free) ──"
    python benchmark_llms.py --from-comparison "$LATEST_CMP" $JUDGE_FLAG --dry-run
    echo ""
    if [ "$JUDGE_ON" = "1" ]; then
        echo "  ⚠  Only the judge makes API calls (estimate above); operational metrics are free."
        echo "     (Make sure your OpenRouter key is saved in the app, or set OPENROUTER_API_KEY.)"
    else
        echo "  ✓  --no-judge selected: this makes ZERO API calls."
    fi
    read -p "  Press Enter to score, or close this window to cancel..."
    python benchmark_llms.py --from-comparison "$LATEST_CMP" $JUDGE_FLAG
    RC=$?
else
    read -p "  Number of patients [20]: " NP; NP=${NP:-20}
    PER=2; [ "$JUDGE_ON" = "1" ] && PER=3
    echo ""
    echo "  ── Preview (no API calls made yet) ──"
    python benchmark_llms.py --dry-run --n "$NP" $JUDGE_FLAG
    echo ""
    echo "  ⚠  The live run calls OpenRouter and uses your credits (~$((NP*6*PER)) calls)."
    echo "     (Make sure your OpenRouter key is saved in the app, or set OPENROUTER_API_KEY.)"
    read -p "  Press Enter to run live, or close this window to cancel..."
    python benchmark_llms.py --n "$NP" $JUDGE_FLAG
    RC=$?
fi

if [ $RC -eq 0 ]; then
    LATEST=$(ls -t benchmark_out/benchmark_report_*.html 2>/dev/null | head -1)
    [ -n "$LATEST" ] && { echo "  Opening report: $LATEST"; open "$LATEST"; }
fi

echo ""
read -p "  Done. Press Enter to close..."
