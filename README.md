# HFpEF-CARE-Agent

**C**ardiac **A**trial-strain **R**isk **E**valuation Agent — a multi-agent
clinical decision-support pipeline for risk stratification in **HFpEF** (heart
failure with preserved ejection fraction), integrating right-atrial strain
imaging, biomarkers, renal function, and a Cox survival model. It also includes
a harness for benchmarking large language models against the pipeline's reports
with blinded multi-rater human evaluation.

## Pipeline

| Stage | Module | Role |
|-------|--------|------|
| Agent A | `agent_a_data_harvester.py` | Ingests a patient record, validates data quality, computes risk flags and an integrated risk tier (LOW / MODERATE / HIGH / CRITICAL). |
| Agent B | `agent_b_imaging_analyst.py` | Right-atrial strain analysis (reservoir / conduit / booster phases, RA myopathy). |
| Agent C | `agent_c_risk_synthesizer.py` | Synthesises the report, Cox 18-month event probability, and recommended clinical actions. |
| App | `app.py` | Streamlit user interface. |
| Benchmark | `benchmark_llms.py`, `compare_llm_outputs.py`, `build_rating_sheets.py`, `aggregate_human_ratings.py` | Cross-LLM comparison and blinded human rating. |

## Quick start

```bash
pip install -r requirements.txt

# Streamlit UI
streamlit run app.py

# CLI demo — runs Agent A over the bundled demo patients
python agent_a_data_harvester.py
```

## Data

The repository ships a small **demo dataset only** (`data/basic_data.xlsx`,
four example patients — one per risk tier). It contains **no real patient
data**. To use your own data, supply an `.xlsx` with one row per patient and a
header row (see the column names in the demo file or the app's upload panel).

## API keys

LLM-benchmark features require an OpenRouter key. No key ships with the repo —
provide your own at runtime via any of:

- the app's settings panel (saved locally to a git-ignored
  `.care_agent_settings.json`; copy `.care_agent_settings.example.json`),
- the `OPENROUTER_API_KEY` environment variable (see `.env.example`), or
- `--api-key` on the benchmark CLIs.

The core Agent A → B → C pipeline runs without any API key.

## Authors

This project is developed and maintained by:

- **Aaron Mah** ([@AaronMahn](https://github.com/AaronMahn)) — lead author. Multi-agent
  pipeline design and implementation, Cox survival modelling, LLM benchmarking
  harness, and the public de-identified release.
- **[@ljj33jj](https://github.com/ljj33jj)** — co-author. Clinical dataset
  preparation and curation, and review of the risk-stratification outputs.

Both authors contributed to the study this repository accompanies. Contributions
are also recorded in the repository's commit history.

## Disclaimer

Research software only. Not a medical device and not for clinical use.
