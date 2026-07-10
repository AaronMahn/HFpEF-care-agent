"""
CARE-Agent — LLM Benchmark Harness
==================================
Runs several LLMs through the *real* Agent A → B → C pipeline and measures how
well each performs at this clinical task. For every (model, patient) it captures:

  Operational metrics
    • reasoning-call success / safety-call success
    • latency (seconds) for the reasoning call and the safety call
    • token usage (reasoning + safety)
    • JSON-structured compliance (did the model return parseable structured JSON?)
    • output completeness (number of watchpoints + safety flags returned)

  Clinical quality (optional, LLM-as-judge)
    • a strong "judge" model scores each candidate's structured reasoning on a
      1–5 rubric: specificity, mechanism, safety, guideline alignment,
      non-redundancy, plus an overall score.

Outputs (written to --output-dir, default ./benchmark_out):
    • benchmark_runs_<ts>.csv      — one row per (model, patient)
    • benchmark_summary_<ts>.csv   — one row per model (aggregates)
    • benchmark_report_<ts>.html   — self-contained HTML report with charts

Models are accessed through a single OpenRouter key (read from
.care_agent_settings.json, the OPENROUTER_API_KEY env var, or --api-key).

Quick start
-----------
    # preview the plan + estimated number of API calls (no calls made):
    python benchmark_llms.py --dry-run

    # offline smoke test of the whole harness (no network, fake LLM):
    python benchmark_llms.py --mock

    # real run: 20 random patients, OpenRouter shortlist, with judge:
    python benchmark_llms.py --n 20

    # operational only (no judge), custom model list:
    python benchmark_llms.py --no-judge --models anthropic/claude-haiku-4.5 openai/gpt-5-mini

This is a research/simulation tool — not for clinical use.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# ── Project imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_a_data_harvester import DataHarvesterAgent          # noqa: E402
from agent_b_imaging_analyst import RaImagingAnalystAgent       # noqa: E402
from agent_c_risk_synthesizer import RiskSynthesizerAgent       # noqa: E402
from llm_client import LLMClient, LLMResponse                   # noqa: E402
import settings_manager as SM                                   # noqa: E402


# ── Curated, catalog-validated OpenRouter shortlist ─────────────────────────────
# A spread across vendors and price/quality tiers. Override with --models.
DEFAULT_MODELS = [
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5-mini",
    "google/gemini-2.5-flash",
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3-max",
    "moonshotai/kimi-k2.6",
]

# Strong default judge (kept separate from the candidates).
DEFAULT_JUDGE_MODEL = "anthropic/claude-opus-4.6"

JUDGE_DIMENSIONS = [
    "specificity",     # tailored to THIS patient, not generic HFpEF boilerplate
    "mechanism",       # pathophysiological reasoning is correct & coherent
    "safety",          # contraindications / cardiorenal safety handled well
    "guideline",       # consistent with current HF/cardiorenal guidance
    "non_redundancy",  # adds insight rather than restating the dashboard numbers
]


# ── Result row ──────────────────────────────────────────────────────────────────
@dataclass
class RunRow:
    model: str
    patient_id: str
    risk_tier: str = ""
    integrated_tier: str = ""
    # reasoning call
    reason_success: bool = False
    reason_structured: bool = False
    reason_latency_s: float = 0.0
    reason_tokens: int = 0
    n_watchpoints: int = 0
    reason_error: str = ""
    # safety call
    safety_success: bool = False
    safety_latency_s: float = 0.0
    safety_tokens: int = 0
    n_flags: int = 0
    safety_error: str = ""
    # judge
    judged: bool = False
    judge_overall: Optional[float] = None
    judge_scores: dict = field(default_factory=dict)
    judge_tokens: int = 0
    judge_error: str = ""

    @property
    def total_latency_s(self) -> float:
        return round(self.reason_latency_s + self.safety_latency_s, 3)

    @property
    def total_tokens(self) -> int:
        return self.reason_tokens + self.safety_tokens


# ── Mock client (for --mock; mirrors LLMClient.complete) ────────────────────────
class MockLLMClient:
    """Returns canned JSON containing keys for reasoning, safety AND judge, so the
    same object can stand in for any call. Simulates latency + token counts so the
    harness can be exercised end-to-end with no network."""

    def __init__(self, provider: str, model: str, **_):
        self.provider = provider
        self.model = model
        self._rng = random.Random(hash(model) & 0xFFFFFFFF)

    def complete(self, system_prompt: str, user_prompt: str,
                 force_json: bool = False) -> LLMResponse:
        time.sleep(self._rng.uniform(0.02, 0.12))  # simulate network latency
        # vary quality a little per model so charts are non-trivial
        base = self._rng.uniform(2.8, 4.6)
        scores = {d: round(min(5.0, max(1.0, base + self._rng.uniform(-0.6, 0.6))), 1)
                  for d in JUDGE_DIMENSIONS}
        payload = {
            # reasoning keys
            "mechanism": "RA dysfunction and CKD interact to raise filling pressures...",
            "clinical_priority": "Cardiorenal optimisation takes precedence.",
            "watchpoints": ["eGFR trend", "NT-proBNP", "volume status"],
            "patient_note": "Your heart's right side and kidneys need close monitoring.",
            "model_caveat": "Imputed Cox features reduce certainty for this patient.",
            # safety keys
            "flags": [{"action_code": "DIURETIC_OPTIMISATION",
                       "concern": "AKI risk with aggressive diuresis in CKD.",
                       "severity": "caution"}],
            "model_discordance": None,
            "overall_safe": True,
            "summary": "Recommendations broadly appropriate; watch renal function.",
            # judge keys
            "scores": scores,
            "overall": round(statistics.mean(scores.values()), 2),
        }
        content = json.dumps(payload, ensure_ascii=False)
        tokens = self._rng.randint(420, 1150)
        return LLMResponse(content=content, model=self.model, provider=self.provider,
                           success=True, tokens_used=tokens)


# ── Judge ───────────────────────────────────────────────────────────────────────
def judge_quality(judge_client, agent_c, snap, ra, report, lang: str) -> dict:
    """Score a candidate's structured reasoning. Returns dict with keys:
    scores{dim:1-5}, overall(float), tokens(int), error(str)."""
    data_block = agent_c._build_data_block(report, snap, ra)
    candidate = json.dumps({
        "mechanism": report.llm_mechanism,
        "clinical_priority": report.llm_clinical_priority,
        "watchpoints": report.llm_watchpoints,
        "patient_note": report.llm_patient_note,
        "model_caveat": report.llm_model_caveat,
    }, ensure_ascii=False, indent=2)

    system_prompt = (
        "You are a senior cardiologist grading another AI's clinical reasoning about an "
        "HFpEF + cardiorenal patient. Be a strict, calibrated grader. Score each dimension "
        "from 1 (poor) to 5 (excellent):\n"
        " - specificity: tailored to THIS patient vs. generic HFpEF statements\n"
        " - mechanism: pathophysiological reasoning is correct and coherent\n"
        " - safety: contraindications / cardiorenal safety handled appropriately\n"
        " - guideline: consistent with current HF/cardiorenal guidance (SGLT2i, etc.)\n"
        " - non_redundancy: adds insight rather than restating the provided numbers\n"
        "Return JSON only (no markdown fences):\n"
        '{"scores":{"specificity":N,"mechanism":N,"safety":N,"guideline":N,'
        '"non_redundancy":N},"overall":N.N,"comment":"one sentence"}'
    )
    user_prompt = (
        f"PATIENT CONTEXT:\n{data_block}\n\n"
        f"AI REASONING TO GRADE:\n{candidate}\n\n"
        "Return the JSON scorecard only."
    )

    t0 = time.perf_counter()
    resp = judge_client.complete(system_prompt, user_prompt)
    dt = time.perf_counter() - t0

    out = {"scores": {}, "overall": None, "tokens": getattr(resp, "tokens_used", 0),
           "error": "", "latency_s": round(dt, 3)}
    if not resp.success:
        out["error"] = resp.error
        return out

    parsed = agent_c._parse_llm_json(resp.content)
    if not parsed or not isinstance(parsed, dict):
        out["error"] = "judge JSON parse failed"
        return out

    raw_scores = parsed.get("scores", {}) or {}
    scores = {}
    for d in JUDGE_DIMENSIONS:
        v = raw_scores.get(d)
        try:
            scores[d] = float(v)
        except (TypeError, ValueError):
            pass
    out["scores"] = scores
    ov = parsed.get("overall")
    try:
        out["overall"] = float(ov)
    except (TypeError, ValueError):
        out["overall"] = round(statistics.mean(scores.values()), 2) if scores else None
    return out


# ── Single (model, patient) run ─────────────────────────────────────────────────
def run_one(agent_c, snap, ra, client, judge_client, lang: str, do_judge: bool) -> RunRow:
    row = RunRow(model=client.model, patient_id=snap.patient_id,
                 risk_tier=snap.risk_tier)

    # Fresh deterministic base report (Cox + rules + summaries; no LLM yet)
    report = agent_c.synthesise(snap, ra, llm_client=None, lang=lang)
    row.integrated_tier = report.integrated_risk_tier

    # 1) Reasoning call (timed in isolation)
    t0 = time.perf_counter()
    agent_c._llm_reasoning(report, snap, ra, client, lang=lang)
    row.reason_latency_s = round(time.perf_counter() - t0, 3)
    row.reason_success = report.llm_available
    row.reason_structured = report.llm_structured
    row.reason_tokens = report.llm_tokens_used
    row.n_watchpoints = len(report.llm_watchpoints or [])
    row.reason_error = report.llm_error or ""

    # 2) Safety / contradiction call (timed in isolation)
    t0 = time.perf_counter()
    agent_c._llm_contradiction_check(report, snap, ra, client, lang=lang)
    row.safety_latency_s = round(time.perf_counter() - t0, 3)
    row.safety_success = report.llm_contradiction_available
    row.safety_tokens = report.llm_contradiction_tokens
    row.n_flags = len(report.llm_contradiction_flags or [])
    row.safety_error = report.llm_contradiction_error or ""

    # 3) Judge (only if reasoning produced structured output)
    if do_judge and row.reason_success and row.reason_structured:
        j = judge_quality(judge_client, agent_c, snap, ra, report, lang)
        row.judged = bool(j["scores"]) and j["error"] == ""
        row.judge_scores = j["scores"]
        row.judge_overall = j["overall"]
        row.judge_tokens = j["tokens"]
        row.judge_error = j["error"]

    return row


# ── Aggregation ─────────────────────────────────────────────────────────────────
def _pct(part, whole):
    return round(100.0 * part / whole, 1) if whole else 0.0


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def _pctile(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = max(0, min(len(xs) - 1, int(round((q / 100.0) * (len(xs) - 1)))))
    return round(xs[k], 3)


def aggregate(rows: list[RunRow]) -> list[dict]:
    by_model: dict[str, list[RunRow]] = {}
    for r in rows:
        by_model.setdefault(r.model, []).append(r)

    summaries = []
    for model, rs in by_model.items():
        n = len(rs)
        reason_ok = [r for r in rs if r.reason_success]
        judged = [r for r in rs if r.judged and r.judge_overall is not None]
        dim_means = {}
        for d in JUDGE_DIMENSIONS:
            dim_means[d] = _mean([r.judge_scores.get(d) for r in judged if r.judge_scores])
        first_err = next((r.reason_error for r in rs if r.reason_error), "") or \
                    next((r.safety_error for r in rs if r.safety_error), "")
        summaries.append({
            "model": model,
            "n": n,
            "reason_success_rate": _pct(len(reason_ok), n),
            "safety_success_rate": _pct(len([r for r in rs if r.safety_success]), n),
            "json_structured_rate": _pct(len([r for r in rs if r.reason_structured]), n),
            "mean_latency_s": _mean([r.total_latency_s for r in rs]),
            "p95_latency_s": _pctile([r.total_latency_s for r in rs], 95),
            "mean_total_tokens": _mean([r.total_tokens for r in rs]),
            "mean_watchpoints": _mean([r.n_watchpoints for r in rs]),
            "mean_safety_flags": _mean([r.n_flags for r in rs]),
            "n_judged": len(judged),
            "judge_overall": _mean([r.judge_overall for r in judged]),
            **{f"judge_{d}": dim_means[d] for d in JUDGE_DIMENSIONS},
            "first_error": first_err[:160],
        })

    # rank: judge_overall desc (None last), then success rate, then latency asc
    summaries.sort(key=lambda s: (
        -(s["judge_overall"] or -1),
        -s["reason_success_rate"],
        s["mean_latency_s"] or 9e9,
    ))
    return summaries


# ── CSV writers ─────────────────────────────────────────────────────────────────
def write_runs_csv(rows: list[RunRow], path: str):
    import csv
    cols = ["model", "patient_id", "risk_tier", "integrated_tier",
            "reason_success", "reason_structured", "reason_latency_s", "reason_tokens",
            "n_watchpoints", "reason_error",
            "safety_success", "safety_latency_s", "safety_tokens", "n_flags", "safety_error",
            "judged", "judge_overall"] + [f"judge_{d}" for d in JUDGE_DIMENSIONS] + \
           ["judge_tokens", "judge_error", "total_latency_s", "total_tokens"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([
                r.model, r.patient_id, r.risk_tier, r.integrated_tier,
                r.reason_success, r.reason_structured, r.reason_latency_s, r.reason_tokens,
                r.n_watchpoints, r.reason_error,
                r.safety_success, r.safety_latency_s, r.safety_tokens, r.n_flags, r.safety_error,
                r.judged, r.judge_overall,
                *[r.judge_scores.get(d) for d in JUDGE_DIMENSIONS],
                r.judge_tokens, r.judge_error, r.total_latency_s, r.total_tokens,
            ])


def write_summary_csv(summary: list[dict], path: str):
    import csv
    if not summary:
        return
    cols = list(summary[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s in summary:
            w.writerow(s)


# ── HTML report ─────────────────────────────────────────────────────────────────
def write_html(summary: list[dict], rows: list[RunRow], meta: dict, path: str):
    labels = [s["model"] for s in summary]

    def col(key):
        return [s.get(key) for s in summary]

    dims = JUDGE_DIMENSIONS
    judge_dim_datasets = [
        {"label": d, "data": [s.get(f"judge_{d}") for s in summary]} for d in dims
    ]

    payload = {
        "labels": labels,
        "success": col("reason_success_rate"),
        "json_rate": col("json_structured_rate"),
        "latency": col("mean_latency_s"),
        "p95": col("p95_latency_s"),
        "tokens": col("mean_total_tokens"),
        "judge_overall": col("judge_overall"),
        "judge_dims": judge_dim_datasets,
        "summary": summary,
        "meta": meta,
    }
    data_json = json.dumps(payload, ensure_ascii=False)

    # Build the summary table rows
    head_cols = ["model", "n", "reason_success_rate", "json_structured_rate",
                 "mean_latency_s", "p95_latency_s", "mean_total_tokens",
                 "mean_watchpoints", "mean_safety_flags", "n_judged", "judge_overall"]
    nice = {
        "model": "Model", "n": "N", "reason_success_rate": "Success %",
        "json_structured_rate": "JSON %", "mean_latency_s": "Latency (s)",
        "p95_latency_s": "p95 (s)", "mean_total_tokens": "Tokens",
        "mean_watchpoints": "Watchpts", "mean_safety_flags": "Flags",
        "n_judged": "Judged", "judge_overall": "Judge /5",
    }
    thead = "".join(f"<th>{nice[c]}</th>" for c in head_cols)
    trows = ""
    for i, s in enumerate(summary):
        cells = ""
        for c in head_cols:
            v = s.get(c)
            v = "—" if v is None else v
            cells += f"<td>{v}</td>"
        rank_cls = ' class="top"' if i == 0 else ""
        trows += f"<tr{rank_cls}>{cells}</tr>"

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CARE-Agent · LLM Benchmark Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         margin: 0; background: #f1f5f9; color: #0f172a; }}
  .wrap {{ max-width: 1100px; margin: 0 auto; padding: 28px 22px 60px; }}
  h1 {{ font-size: 24px; margin: 0 0 4px; }}
  .sub {{ color: #64748b; font-size: 13px; margin-bottom: 18px; }}
  .meta {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:22px; }}
  .chip {{ background:#fff; border:1px solid #e2e8f0; border-radius:20px;
          padding:5px 14px; font-size:12px; color:#475569; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .card {{ background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:16px 18px;
          box-shadow:0 4px 16px rgba(15,23,42,.05); }}
  .card h3 {{ font-size:13px; text-transform:uppercase; letter-spacing:.05em; color:#64748b;
             margin:0 0 10px; }}
  .full {{ grid-column:1 / -1; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th, td {{ text-align:right; padding:8px 10px; border-bottom:1px solid #eef2f7; }}
  th:first-child, td:first-child {{ text-align:left; }}
  th {{ color:#64748b; font-size:11px; text-transform:uppercase; letter-spacing:.04em; }}
  tr.top td {{ background:#f0fdf4; font-weight:600; }}
  .foot {{ color:#94a3b8; font-size:11px; margin-top:24px; text-align:center; }}
  @media (max-width:760px) {{ .grid {{ grid-template-columns:1fr; }} }}
</style></head>
<body><div class="wrap">
  <h1>🫀 CARE-Agent · LLM Benchmark</h1>
  <div class="sub">Each model run through the real Agent A → B → C pipeline · simulation study, not for clinical use</div>
  <div class="meta" id="meta"></div>

  <div class="card full" style="margin-bottom:16px">
    <h3>Ranked Summary</h3>
    <table><thead><tr>{thead}</tr></thead><tbody>{trows}</tbody></table>
  </div>

  <div class="grid">
    <div class="card"><h3>Reasoning success rate (%)</h3><canvas id="cSuccess"></canvas></div>
    <div class="card"><h3>JSON-structured compliance (%)</h3><canvas id="cJson"></canvas></div>
    <div class="card"><h3>Mean latency per patient (s)</h3><canvas id="cLatency"></canvas></div>
    <div class="card"><h3>Mean total tokens</h3><canvas id="cTokens"></canvas></div>
    <div class="card full"><h3>Judge clinical-quality score (/5)</h3><canvas id="cJudge" height="120"></canvas></div>
    <div class="card full"><h3>Judge score by dimension (/5)</h3><canvas id="cDims" height="140"></canvas></div>
  </div>
  <div class="foot">Generated by benchmark_llms.py</div>
</div>
<script>
const D = {data_json};
const PAL = ["#2563eb","#16a34a","#ea580c","#7c3aed","#dc2626","#0ea5e9","#d97706","#0891b2"];
const m = D.meta;
document.getElementById('meta').innerHTML =
  ['Generated: '+m.timestamp, 'Patients: '+m.n_patients, 'Models: '+m.n_models,
   'Judge: '+(m.judge||'off'), 'Lang: '+m.lang, (m.mock?'MOCK run':'live run')]
  .map(t=>'<span class="chip">'+t+'</span>').join('');

function bar(id, data, color, opts={{}}) {{
  new Chart(document.getElementById(id), {{
    type:'bar',
    data:{{ labels:D.labels, datasets:[{{ data, backgroundColor:color, borderRadius:6 }}] }},
    options:Object.assign({{ plugins:{{legend:{{display:false}}}},
      scales:{{ x:{{ ticks:{{ font:{{size:10}} }} }}, y:{{ beginAtZero:true }} }} }}, opts)
  }});
}}
bar('cSuccess', D.success, '#16a34a', {{scales:{{y:{{beginAtZero:true,max:100}}}}}});
bar('cJson',    D.json_rate,'#2563eb', {{scales:{{y:{{beginAtZero:true,max:100}}}}}});
bar('cLatency', D.latency, '#ea580c');
bar('cTokens',  D.tokens,  '#7c3aed');
bar('cJudge',   D.judge_overall, '#0891b2', {{scales:{{y:{{beginAtZero:true,max:5}}}}}});

new Chart(document.getElementById('cDims'), {{
  type:'bar',
  data:{{ labels:D.labels,
    datasets:D.judge_dims.map((ds,i)=>({{label:ds.label, data:ds.data,
      backgroundColor:PAL[i%PAL.length], borderRadius:4}})) }},
  options:{{ responsive:true, scales:{{ y:{{beginAtZero:true,max:5}} }},
            plugins:{{legend:{{position:'bottom', labels:{{font:{{size:10}}}}}}}} }}
}});
</script>
</body></html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# ── Patient sampling ────────────────────────────────────────────────────────────
def sample_patients(agent_a, n: int, seed: int, use_all: bool) -> list[str]:
    names = list(agent_a._raw["_pid"].dropna().tolist())
    if use_all or n >= len(names):
        return names
    rng = random.Random(seed)
    return rng.sample(names, n)


# ── Orchestration ───────────────────────────────────────────────────────────────
def make_client(provider, model, api_key, mock, timeout, max_tokens, temperature):
    if mock:
        return MockLLMClient(provider=provider, model=model)
    return LLMClient(provider=provider, api_key=api_key, model=model,
                     timeout=timeout, max_tokens=max_tokens, temperature=temperature)


# ── Build rows from a saved comparison JSON (no model re-runs) ───────────────────
def _runrow_from_capture(model, patient_id, snap_tier, integrated_tier, reasoning, safety) -> RunRow:
    row = RunRow(model=model, patient_id=patient_id, risk_tier=snap_tier or "")
    row.integrated_tier   = integrated_tier or ""
    row.reason_success    = bool(reasoning.get("success"))
    row.reason_structured = bool(reasoning.get("structured"))
    row.reason_latency_s  = float(reasoning.get("latency_s") or 0.0)
    row.reason_tokens     = int(reasoning.get("tokens") or 0)
    row.n_watchpoints     = len(reasoning.get("watchpoints") or [])
    row.reason_error      = reasoning.get("error") or ""
    row.safety_success    = bool(safety.get("available"))
    row.safety_latency_s  = float(safety.get("latency_s") or 0.0)
    row.safety_tokens     = int(safety.get("tokens") or 0)
    row.n_flags           = len(safety.get("flags") or [])
    row.safety_error      = safety.get("error") or ""
    return row


def rows_from_comparison(path, do_judge, judge_client, lang):
    """Reconstruct benchmark rows from a compare_llm_outputs.py JSON.

    Operational metrics (success, latency, tokens, JSON-structure, completeness)
    are taken straight from the captured data — zero API calls. If judging is on,
    we re-derive the deterministic patient context (Agent A/B + base Cox/rules, no
    LLM) so the judge sees the same data block, then score the *captured* reasoning
    — so the only API calls are the judge calls.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    patients = data.get("patients", [])

    agent_a = agent_b = agent_c = None
    if do_judge:
        print("[bench] Loading agents to rebuild patient context for judging...")
        agent_a = DataHarvesterAgent()
        agent_b = RaImagingAnalystAgent()
        agent_c = RiskSynthesizerAgent()

    rows: list[RunRow] = []
    for pi, rec in enumerate(patients, 1):
        pid = rec.get("patient_id")
        snap_tier  = (rec.get("snapshot") or {}).get("agent_a_tier", "")
        integrated = (rec.get("original") or {}).get("integrated_tier", "")

        snap = ra = None
        if do_judge:
            try:
                snap = agent_a.harvest(pid)
                ra = agent_b.analyse(snap)
            except Exception as e:
                print(f"[bench] WARN: cannot rebuild context for {pid} (judging skipped): {e}")

        print(f"[bench] ({pi}/{len(patients)}) {pid}")
        for m in rec.get("models", []):
            model = m.get("model", "?")
            reasoning = m.get("reasoning", {}) or {}
            safety    = m.get("safety", {}) or {}
            row = _runrow_from_capture(model, pid, snap_tier, integrated, reasoning, safety)

            if do_judge and snap is not None and row.reason_success and row.reason_structured:
                report = agent_c.synthesise(snap, ra, llm_client=None, lang=lang)
                report.llm_mechanism         = reasoning.get("mechanism", "")
                report.llm_clinical_priority = reasoning.get("clinical_priority", "")
                report.llm_watchpoints       = reasoning.get("watchpoints", []) or []
                report.llm_patient_note      = reasoning.get("patient_note", "")
                report.llm_model_caveat      = reasoning.get("model_caveat", "")
                j = judge_quality(judge_client, agent_c, snap, ra, report, lang)
                row.judged        = bool(j["scores"]) and j["error"] == ""
                row.judge_scores  = j["scores"]
                row.judge_overall = j["overall"]
                row.judge_tokens  = j["tokens"]
                row.judge_error   = j["error"]
            rows.append(row)
            jstr = f" judge={row.judge_overall}" if row.judged else ""
            print(f"      {model}: reused {row.total_latency_s}s {row.total_tokens}tok{jstr}")
    return rows


def _finalize(rows, models, n_patients, args, lang, mock_flag) -> int:
    summary = aggregate(rows)
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_csv = os.path.join(args.output_dir, f"benchmark_runs_{ts}.csv")
    sum_csv = os.path.join(args.output_dir, f"benchmark_summary_{ts}.csv")
    html = os.path.join(args.output_dir, f"benchmark_report_{ts}.html")

    write_runs_csv(rows, runs_csv)
    write_summary_csv(summary, sum_csv)
    write_html(summary, rows, {
        "timestamp": ts, "n_patients": n_patients, "n_models": len(models),
        "judge": args.judge_model if not args.no_judge else "", "lang": lang,
        "mock": mock_flag,
    }, html)

    print("\n" + "=" * 60)
    print("  LEADERBOARD (ranked by judge score, then success, then latency)")
    print("=" * 60)
    print(f"  {'model':<34} {'succ%':>6} {'json%':>6} {'lat(s)':>7} {'tok':>6} {'judge':>6}")
    for s in summary:
        print(f"  {s['model']:<34} {s['reason_success_rate']:>6} "
              f"{s['json_structured_rate']:>6} {str(s['mean_latency_s']):>7} "
              f"{str(s['mean_total_tokens']):>6} {str(s['judge_overall']):>6}")
    print("\n[bench] Wrote:")
    for p in (runs_csv, sum_csv, html):
        print(f"  - {p}")
    return 0


def run_benchmark(args) -> int:
    do_judge = not args.no_judge

    # ── Mode B: score an existing comparison JSON (no model re-runs) ────────────
    if args.from_comparison:
        if not os.path.isfile(args.from_comparison):
            print(f"[bench] ERROR: file not found: {args.from_comparison}")
            return 2
        with open(args.from_comparison, encoding="utf-8") as f:
            peek = json.load(f)
        pats = peek.get("patients", [])
        cmeta = peek.get("meta", {})
        lang = cmeta.get("lang", args.lang)
        n_models = cmeta.get("n_models") or (len(pats[0]["models"]) if pats else 0)
        n_judge_calls = (sum(
            1 for rec in pats for m in rec.get("models", [])
            if (m.get("reasoning") or {}).get("success") and (m.get("reasoning") or {}).get("structured")
        ) if do_judge else 0)

        print("\n" + "=" * 60)
        print("  Mode          : score saved comparison (no model re-runs)")
        print(f"  Source        : {args.from_comparison}")
        print(f"  Patients      : {len(pats)}  |  Models: {n_models}")
        print("  Operational   : reused from file (0 API calls)")
        print(f"  Judge         : {'on -> ' + args.judge_model if do_judge else 'off'}")
        print(f"  Est. API calls: ~{n_judge_calls}  ({'MOCK' if args.mock else 'judge calls only'})")
        print("=" * 60 + "\n")
        if cmeta.get("mock"):
            print("[bench] NOTE: source is a --mock comparison; judging its placeholder text is not meaningful.\n")
        if args.dry_run:
            print("[bench] --dry-run: nothing called.")
            return 0

        judge_client = None
        if do_judge:
            api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "") or SM.get_api_key(args.provider)
            if not args.mock and not api_key:
                print(f"[bench] ERROR: judging needs an API key for '{args.provider}'. "
                      f"Use --no-judge to score operational metrics only (0 calls).")
                return 2
            judge_client = make_client(args.provider, args.judge_model, api_key, args.mock,
                                       args.timeout, args.max_tokens, args.temperature)

        rows = rows_from_comparison(args.from_comparison, do_judge, judge_client, lang)
        models = sorted({r.model for r in rows})
        return _finalize(rows, models, len(pats), args, lang, args.mock or bool(cmeta.get("mock")))

    # ── Mode A: live run (calls every model) ───────────────────────────────────
    print("[bench] Loading agents + cohort...")
    agent_a = DataHarvesterAgent()
    agent_b = RaImagingAnalystAgent()
    agent_c = RiskSynthesizerAgent()

    patients = sample_patients(agent_a, args.n, args.seed, args.all)
    models = args.models or DEFAULT_MODELS

    n_calls = len(patients) * len(models) * (2 + (1 if do_judge else 0))
    print("\n" + "=" * 60)
    print(f"  Provider     : {args.provider}")
    print(f"  Models       : {len(models)} -> {', '.join(models)}")
    print(f"  Patients     : {len(patients)} (seed={args.seed})")
    print(f"  Judge        : {'on -> ' + args.judge_model if do_judge else 'off'}")
    print(f"  Language     : {args.lang}")
    print(f"  Est. API calls: ~{n_calls}  ({'MOCK - no real calls' if args.mock else 'live - consumes credits'})")
    print("=" * 60 + "\n")

    if args.dry_run:
        print("[bench] --dry-run set: nothing was called. Patients sampled:")
        print("        " + ", ".join(patients))
        return 0

    # Resolve API key (not needed for --mock)
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "") or SM.get_api_key(args.provider)
    if not args.mock and not api_key:
        print(f"[bench] ERROR: no API key for '{args.provider}'. Provide --api-key, set "
              f"OPENROUTER_API_KEY, or save one in the app sidebar.")
        return 2

    judge_client = None
    if do_judge:
        judge_client = make_client(args.provider, args.judge_model, api_key, args.mock,
                                   args.timeout, args.max_tokens, args.temperature)

    # Pre-harvest each patient once (deterministic; reused across all models)
    print("[bench] Harvesting patient snapshots...")
    snaps = {}
    for pid in patients:
        try:
            snap = agent_a.harvest(pid)
            snaps[pid] = (snap, agent_b.analyse(snap))
        except Exception as e:
            print(f"[bench] WARN: skipping {pid}: {e}")

    rows: list[RunRow] = []
    for mi, model in enumerate(models, 1):
        client = make_client(args.provider, model, api_key, args.mock,
                             args.timeout, args.max_tokens, args.temperature)
        print(f"\n[bench] ({mi}/{len(models)}) Model: {model}")
        for pi, pid in enumerate(patients, 1):
            if pid not in snaps:
                continue
            snap, ra = snaps[pid]
            try:
                row = run_one(agent_c, snap, ra, client, judge_client, args.lang, do_judge)
            except Exception as e:
                row = RunRow(model=model, patient_id=pid, reason_error=f"harness error: {e}")
            rows.append(row)
            status = "ok" if row.reason_success else f"FAIL ({row.reason_error[:40]})"
            jstr = f" judge={row.judge_overall}" if row.judged else ""
            print(f"   [{pi}/{len(patients)}] {pid}: {status} "
                  f"{row.total_latency_s}s {row.total_tokens}tok{jstr}")
            if args.sleep:
                time.sleep(args.sleep)

    return _finalize(rows, models, len(patients), args, args.lang, args.mock)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark LLMs through the CARE-Agent pipeline.")
    p.add_argument("--from-comparison", default="", dest="from_comparison",
                   help="path to a compare_llm_outputs.py JSON; score it WITHOUT re-running models "
                        "(operational metrics reused free; only the judge makes calls)")
    p.add_argument("--n", type=int, default=20, help="number of random patients (default 20)")
    p.add_argument("--all", action="store_true", help="use all patients (overrides --n)")
    p.add_argument("--seed", type=int, default=42, help="random seed for patient sampling")
    p.add_argument("--models", nargs="*", default=None,
                   help="OpenRouter model IDs to test (default: curated shortlist)")
    p.add_argument("--provider", default="openrouter", help="LLM provider (default openrouter)")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, dest="judge_model",
                   help=f"judge model (default {DEFAULT_JUDGE_MODEL})")
    p.add_argument("--no-judge", action="store_true", help="skip clinical-quality judging")
    p.add_argument("--lang", default="en", choices=["en", "zh"], help="report language")
    p.add_argument("--api-key", default="", dest="api_key",
                   help="API key (else OPENROUTER_API_KEY env or saved settings)")
    p.add_argument("--output-dir", default="benchmark_out", dest="output_dir")
    p.add_argument("--timeout", type=float, default=150.0)
    p.add_argument("--max-tokens", type=int, default=4000, dest="max_tokens")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--sleep", type=float, default=0.0, help="seconds to sleep between calls")
    p.add_argument("--dry-run", action="store_true", help="print the plan + call estimate, make no calls")
    p.add_argument("--mock", action="store_true", help="offline smoke test with a fake LLM (no network)")
    return p


def main():
    args = build_parser().parse_args()
    try:
        return run_benchmark(args)
    except KeyboardInterrupt:
        print("\n[bench] interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
