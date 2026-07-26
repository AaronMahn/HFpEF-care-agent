"""
CARE-Agent — LLM Output Comparison
==================================
Captures the EXACT outputs of the two LLM functions in Agent C, side by side,
across one flagship model per vendor, so you can compare what each model actually
says for the same patients.

For each of N patients it records:
  • the original rule-based report (deterministic — no LLM)
  • per model, the structured REASONING output:
        mechanism · clinical_priority · watchpoints · patient_note · model_caveat
        (+ the raw text and JSON-structured flag)
  • per model, the SAFETY / contradiction FLAGGING output:
        flags [{action_code, concern, severity}] · summary

Flagship model list (one per vendor, validated against the OpenRouter catalog):
    Claude    anthropic/claude-opus-4.7
    Gemini    google/gemini-3.1-pro-preview
    GPT       openai/gpt-5.5
    DeepSeek  deepseek/deepseek-v4-pro
    Kimi      moonshotai/kimi-k2.6
    Qwen      qwen/qwen3-max

Outputs (to --output-dir, default ./comparison_out):
    llm_comparison_<ts>.json   — full fidelity (everything, machine-readable)
    llm_comparison_<ts>.csv    — one row per (patient, model) for spreadsheet diffing
    llm_comparison_<ts>.html   — readable per-patient side-by-side

Usage
-----
    python compare_llm_outputs.py --dry-run          # preview plan, no calls
    python compare_llm_outputs.py --mock             # offline smoke test
    python compare_llm_outputs.py --n 20             # real run (uses OpenRouter key)
    python compare_llm_outputs.py --n 20 --lang zh   # capture Chinese outputs

Reuses the real Agent A→B→C code, so it reflects the agent's true behaviour.
Simulation/research tool — not for clinical use.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_a_data_harvester import DataHarvesterAgent           # noqa: E402
from agent_b_imaging_analyst import RaImagingAnalystAgent        # noqa: E402
from agent_c_risk_synthesizer import RiskSynthesizerAgent        # noqa: E402
import settings_manager as SM                                    # noqa: E402
# Reuse client factory + offline mock from the scored benchmark
from benchmark_llms import make_client, sample_patients          # noqa: E402
# Reuse the stratified sampler from the rule-based reports script
from stratified_clinical_reports import (                        # noqa: E402
    build_patient_record, stratified_sample, TIERS as ALL_TIERS, TIER_ZH,
)


# ── Flagship model per vendor (override with --models) ──────────────────────────
FLAGSHIP_MODELS = [
    "anthropic/claude-opus-4.7",      # Claude
    "google/gemini-3.1-pro-preview",  # Gemini
    "openai/gpt-5.5",                 # GPT
    "deepseek/deepseek-v4-pro",       # DeepSeek
    "moonshotai/kimi-k2.6",           # Kimi
    "qwen/qwen3-max",                 # Qwen
]

# Vendor → display name + accent colour (for the HTML)
VENDOR_STYLE = {
    "anthropic":   ("Claude",   "#d97706"),
    "google":      ("Gemini",   "#2563eb"),
    "openai":      ("GPT",      "#16a34a"),
    "deepseek":    ("DeepSeek", "#7c3aed"),
    "moonshotai":  ("Kimi",     "#db2777"),
    "qwen":        ("Qwen",     "#0891b2"),
}


def vendor_of(model: str):
    key = model.split("/")[0]
    return VENDOR_STYLE.get(key, (key, "#475569"))


def _num(v):
    """Coerce pandas/numpy scalars to native Python; NaN/None -> None."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) else f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        f = float(o)
        return None if math.isnan(f) else f
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ── Capture one model's exact outputs for one patient ───────────────────────────
def capture_model(agent_c, snap, ra, client, lang: str) -> dict:
    # Fresh deterministic base report, then the two LLM calls (real agent code)
    report = agent_c.synthesise(snap, ra, llm_client=None, lang=lang)

    t0 = time.perf_counter()
    agent_c._llm_reasoning(report, snap, ra, client, lang=lang)
    reason_latency = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    agent_c._llm_contradiction_check(report, snap, ra, client, lang=lang)
    safety_latency = round(time.perf_counter() - t0, 3)

    return {
        "model": client.model,
        "reasoning": {
            "success": report.llm_available,
            "structured": report.llm_structured,
            "latency_s": reason_latency,
            "tokens": report.llm_tokens_used,
            "error": report.llm_error or "",
            "mechanism": report.llm_mechanism or "",
            "clinical_priority": report.llm_clinical_priority or "",
            "watchpoints": list(report.llm_watchpoints or []),
            "patient_note": report.llm_patient_note or "",
            "model_caveat": report.llm_model_caveat or "",
            "raw": report.llm_reasoning or "",
        },
        "safety": {
            "available": report.llm_contradiction_available,
            "latency_s": safety_latency,
            "tokens": report.llm_contradiction_tokens,
            "error": report.llm_contradiction_error or "",
            "flags": list(report.llm_contradiction_flags or []),
            "summary": report.llm_contradiction_summary or "",
        },
    }


def original_report(agent_c, snap, ra, lang: str) -> dict:
    """Deterministic, rule-based report context (no LLM)."""
    rep = agent_c.synthesise(snap, ra, llm_client=None, lang=lang)
    text = rep.summary_zh if lang == "zh" else rep.clinical_summary
    return {
        "integrated_tier": rep.integrated_risk_tier,
        "cox_2yr_event_prob": rep.cox_2yr_event_prob,
        "cox_risk_label": rep.cox_risk_label,
        "primary_action": rep.primary_action,
        "actions": [code for code, _, _ in rep.all_actions],
        "risk_drivers": list(rep.risk_drivers_zh if lang == "zh" else rep.risk_drivers),
        "summary_text": text,
    }


def snapshot_brief(snap) -> dict:
    is_male = str(snap.sex).strip().lower() in ("male", "m", "男")
    return {
        "age": int(snap.age) if snap.age is not None else None,
        "sex": "Male" if is_male else "Female",
        "nyha": int(snap.nyha_class) if snap.nyha_class else None,
        "agent_a_tier": snap.risk_tier,
        "egfr": _num(snap.egfr), "ntprobnp": _num(snap.ntprobnp),
        "ra_reservoir": _num(snap.ra_reservoir),
        "af": _num(snap.af), "ckd": _num(snap.ckd),
    }


# ── Writers ─────────────────────────────────────────────────────────────────────
def write_json(records: list[dict], meta: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "patients": records}, f, ensure_ascii=False, indent=2,
                  default=_json_default)


def _flags_to_str(flags) -> str:
    parts = []
    for fl in flags:
        if isinstance(fl, dict):
            parts.append(f"[{fl.get('severity','?')}] {fl.get('action_code','')}: {fl.get('concern','')}")
        else:
            parts.append(str(fl))
    return " || ".join(parts)


def write_csv(records: list[dict], path: str):
    cols = ["patient_id", "age", "sex", "nyha", "integrated_tier", "model",
            "reason_success", "reason_structured", "reason_latency_s", "reason_tokens",
            "mechanism", "clinical_priority", "watchpoints", "patient_note", "model_caveat",
            "reason_error",
            "safety_available", "safety_latency_s", "safety_tokens",
            "safety_flags", "safety_summary", "safety_error"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for rec in records:
            sb = rec["snapshot"]
            orig = rec["original"]
            for m in rec["models"]:
                r, s = m["reasoning"], m["safety"]
                w.writerow([
                    rec["patient_id"], sb["age"], sb["sex"], sb["nyha"],
                    orig["integrated_tier"], m["model"],
                    r["success"], r["structured"], r["latency_s"], r["tokens"],
                    r["mechanism"], r["clinical_priority"], " | ".join(r["watchpoints"]),
                    r["patient_note"], r["model_caveat"], r["error"],
                    s["available"], s["latency_s"], s["tokens"],
                    _flags_to_str(s["flags"]), s["summary"], s["error"],
                ])


SEV_COLOR = {"warning": "#dc2626", "caution": "#d97706", "info": "#2563eb"}


def _flag_html(flags) -> str:
    if not flags:
        return '<div class="muted">— no flags —</div>'
    out = []
    for fl in flags:
        if not isinstance(fl, dict):
            out.append(f'<div class="flag">{html.escape(str(fl))}</div>')
            continue
        sev = str(fl.get("severity", "caution")).lower()
        c = SEV_COLOR.get(sev, "#d97706")
        code = html.escape(str(fl.get("action_code", "")))
        concern = html.escape(str(fl.get("concern", "")))
        out.append(
            f'<div class="flag" style="border-left:3px solid {c}">'
            f'<b style="color:{c}">{sev.upper()}</b> '
            f'<span class="code">{code}</span><br>{concern}</div>'
        )
    return "".join(out)


def _wp_html(wps) -> str:
    if not wps:
        return '<span class="muted">—</span>'
    return "<ul class='wp'>" + "".join(f"<li>{html.escape(str(w))}</li>" for w in wps) + "</ul>"


TIER_BANNER = {
    "LOW":      ("#1b6a4a", "#dff0e6"),
    "MODERATE": ("#8a6a00", "#fff4cf"),
    "HIGH":     ("#b15a14", "#fde9d4"),
    "CRITICAL": ("#b71d3b", "#fde2e7"),
}


def write_html(records: list[dict], meta: dict, path: str, tier_of: dict | None = None):
    tier_of = tier_of or {}

    # Reorder records by tier when we have a tier mapping (LOW → CRITICAL)
    if tier_of:
        order = {t: i for i, t in enumerate(ALL_TIERS)}
        records = sorted(
            records,
            key=lambda r: (order.get(tier_of.get(r["patient_id"], "Unknown"), 99),
                           r["patient_id"]),
        )

    # Group records by tier so we can render section headers between groups
    grouped: dict[str, list[tuple[int, dict]]] = {}
    for i, r in enumerate(records):
        t = tier_of.get(r["patient_id"]) or r["original"].get("integrated_tier", "Unknown")
        grouped.setdefault(t, []).append((i, r))

    # Top navigation — patient pids grouped by tier
    nav_parts = []
    for t in ALL_TIERS + ["Unknown"]:
        if t not in grouped:
            continue
        fg, bg = TIER_BANNER.get(t, ("#475569", "#e2e8f0"))
        links = " ".join(
            f'<a href="#p{i}">{html.escape(r["patient_id"])}</a>'
            for i, r in grouped[t]
        )
        nav_parts.append(
            f'<span class="navg" style="background:{bg};color:{fg}">'
            f'<b>{html.escape(t)} ({len(grouped[t])})</b> {links}</span>'
        )
    nav = " ".join(nav_parts)

    sel = meta.get("selection_mode", "random")
    chip_texts = [
        "Generated: " + meta["timestamp"],
        f'Patients: {meta["n_patients"]}',
        f'Models: {meta["n_models"]}',
        f'Selection: {sel}',
        f'Lang: {meta["lang"]}',
        ("MOCK run" if meta["mock"] else "live run"),
    ]
    chips = "".join(f'<span class="chip">{html.escape(t)}</span>' for t in chip_texts)

    blocks = []
    last_tier = None
    for i, rec in enumerate(records):
        t = tier_of.get(rec["patient_id"]) or rec["original"].get("integrated_tier", "Unknown")
        if t != last_tier:
            fg, bg = TIER_BANNER.get(t, ("#475569", "#e2e8f0"))
            tier_zh = TIER_ZH.get(t, t)
            blocks.append(
                f'<h2 class="tierhd" style="color:{fg};background:{bg}">'
                f'{html.escape(t)} · {html.escape(tier_zh)} · {len(grouped.get(t, []))} patient(s)</h2>'
            )
            last_tier = t

        # NOTE: keep the original per-patient block builder intact below
        _ = rec   # alias so the body below still works on `rec`
        sb = rec["snapshot"]
        orig = rec["original"]
        prob = orig["cox_2yr_event_prob"]
        prob_s = f"{prob:.1%}" if isinstance(prob, (int, float)) else "N/A"
        demo = f'{sb["age"]}y · {sb["sex"]} · NYHA {sb["nyha"]}'
        head = (
            f'<div class="phead"><div><span class="pid">{html.escape(rec["patient_id"])}</span>'
            f'<span class="demo">{html.escape(demo)}</span></div>'
            f'<div class="tierwrap"><span class="tier">{html.escape(orig["integrated_tier"])}</span>'
            f'<span class="cox">Cox 18-mo: <b>{prob_s}</b></span></div></div>'
        )
        orig_box = (
            '<details class="orig" open>'
            '<summary>Original rule-based report (no LLM) — click to collapse</summary>'
            f'<pre>{html.escape(orig["summary_text"])}</pre></details>'
        )

        cards = []
        for m in rec["models"]:
            vname, vcolor = vendor_of(m["model"])
            r, s = m["reasoning"], m["safety"]
            status = ("ok" if r["success"] else "FAIL")
            err = ""
            if not r["success"] and r["error"]:
                err = f'<div class="err">{html.escape(r["error"])}</div>'
            cards.append(f"""
<div class="card">
  <div class="chead" style="background:{vcolor}">
    <span class="vn">{html.escape(vname)}</span>
    <span class="mn">{html.escape(m["model"])}</span>
    <span class="meta2">{status} · {r["latency_s"]+s["latency_s"]:.2f}s · {r["tokens"]+s["tokens"]}tok</span>
  </div>
  <div class="cbody">
    {err}
    <div class="lbl">Mechanism</div><div class="txt">{html.escape(r["mechanism"]) or '<span class="muted">—</span>'}</div>
    <div class="lbl">Clinical priority</div><div class="txt">{html.escape(r["clinical_priority"]) or '<span class="muted">—</span>'}</div>
    <div class="lbl">Watchpoints</div>{_wp_html(r["watchpoints"])}
    <div class="lbl">Patient note</div><div class="txt note">{html.escape(r["patient_note"]) or '<span class="muted">—</span>'}</div>
    <div class="lbl">Model caveat</div><div class="txt muted2">{html.escape(r["model_caveat"]) or '<span class="muted">—</span>'}</div>
    <div class="lbl safety">Safety flags</div>{_flag_html(s["flags"])}
    <div class="lbl">Safety summary</div><div class="txt">{html.escape(s["summary"]) or '<span class="muted">—</span>'}</div>
  </div>
</div>""")

        blocks.append(
            f'<section id="p{i}" class="pblock">{head}{orig_box}'
            f'<div class="grid">{"".join(cards)}</div></section>'
        )

    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CARE-Agent · LLM Output Comparison</title>
<style>
  body {{ font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         margin:0; background:#f1f5f9; color:#0f172a; }}
  .wrap {{ max-width:1280px; margin:0 auto; padding:26px 20px 80px; }}
  h1 {{ font-size:23px; margin:0 0 4px; }}
  .sub {{ color:#64748b; font-size:13px; margin-bottom:14px; }}
  .chip {{ background:#fff; border:1px solid #e2e8f0; border-radius:20px; padding:5px 13px;
          font-size:12px; color:#475569; margin-right:6px; }}
  .nav {{ font-size:12px; color:#2563eb; margin:14px 0 8px; line-height:1.9; }}
  .nav a {{ color:#2563eb; text-decoration:none; }}
  .pblock {{ background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:16px 18px;
            margin:18px 0; box-shadow:0 4px 16px rgba(15,23,42,.05); scroll-margin-top:12px; }}
  .phead {{ display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; }}
  .pid {{ font-size:20px; font-weight:800; }}
  .demo {{ color:#64748b; font-size:13px; margin-left:12px; }}
  .tierwrap {{ text-align:right; }}
  .tier {{ font-weight:800; color:#dc2626; }}
  .cox {{ color:#64748b; font-size:12px; margin-left:12px; }}
  details.orig {{ margin:10px 0 14px; }}
  details.orig summary {{ cursor:pointer; font-size:12px; color:#64748b; font-weight:600; }}
  details.orig pre {{ background:#0f172a; color:#e2e8f0; padding:14px; border-radius:8px;
                     white-space:pre-wrap; font-size:11.5px; line-height:1.6; overflow:auto; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:12px; }}
  .card {{ border:1px solid #e2e8f0; border-radius:10px; overflow:hidden; background:#fff; }}
  .chead {{ color:#fff; padding:8px 12px; }}
  .chead .vn {{ font-weight:800; font-size:14px; }}
  .chead .mn {{ font-size:11px; opacity:.92; margin-left:6px; }}
  .chead .meta2 {{ display:block; font-size:10.5px; opacity:.9; margin-top:2px; }}
  .cbody {{ padding:10px 12px; font-size:12.5px; }}
  .lbl {{ font-size:10px; text-transform:uppercase; letter-spacing:.05em; color:#94a3b8;
         font-weight:700; margin:9px 0 2px; }}
  .lbl.safety {{ color:#dc2626; }}
  .txt {{ line-height:1.55; color:#1e293b; }}
  .txt.note {{ font-style:italic; color:#6d28d9; }}
  .muted2 {{ color:#64748b; font-size:11.5px; }}
  .muted {{ color:#cbd5e1; }}
  ul.wp {{ margin:2px 0 0; padding-left:18px; }}
  ul.wp li {{ margin:1px 0; }}
  .flag {{ background:#f8fafc; border-radius:0 6px 6px 0; padding:6px 9px; margin:3px 0; font-size:12px; }}
  .flag .code {{ color:#64748b; font-size:10.5px; }}
  .err {{ background:#fef2f2; color:#b91c1c; border:1px solid #fecaca; border-radius:6px;
         padding:6px 9px; font-size:11.5px; margin-bottom:6px; }}
  .foot {{ color:#94a3b8; font-size:11px; margin-top:24px; text-align:center; }}
  .tierhd {{ font-size:16px; font-weight:700; margin:24px 0 10px; padding:8px 14px;
             border-radius:8px; letter-spacing:.02em; }}
  .navg {{ display:inline-block; padding:6px 12px; border-radius:18px; margin:0 6px 6px 0;
           font-size:12px; line-height:1.7; }}
  .navg a {{ text-decoration:none; color:inherit; opacity:.85; margin:0 4px; }}
  .navg a:hover {{ opacity:1; text-decoration:underline; }}
</style></head>
<body><div class="wrap">
  <h1>🫀 CARE-Agent · LLM Output Comparison</h1>
  <div class="sub">Exact reasoning + safety-flagging outputs per flagship model · simulation study, not for clinical use</div>
  <div>{chips}</div>
  <div class="nav" style="margin-top:14px">{nav}</div>
  {''.join(blocks)}
  <div class="foot">Generated by compare_llm_outputs.py</div>
</div></body></html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html_doc)


# ── Orchestration ───────────────────────────────────────────────────────────────
def _select_patients_stratified(agent_a, agent_b, agent_c, args):
    """Run the deterministic pipeline on the whole cohort and pick N patients
    per integrated risk tier (quantile-spaced over Cox probability inside each
    tier). Returns (ordered_patient_ids, tier_of_pid, selection_stats, picked).
    `picked` is the OrderedDict (tier -> list of rule-based record dicts)
    suitable for the bilingual companion HTML writer."""
    import io, contextlib
    print(f"[compare] Stratified mode — building rule-based reports for the cohort "
          f"(no LLM, ~few seconds)...")
    buf = io.StringIO()
    records: list[dict] = []
    with contextlib.redirect_stdout(buf):
        snaps = agent_a.harvest_all()
        for s in snaps:
            try:
                ra = agent_b.analyse(s)
            except Exception:
                continue
            rec = build_patient_record(agent_c, s, ra)
            if rec is not None:
                records.append(rec)

    picked, stats = stratified_sample(records, args.n_per_tier, args.tiers, args.seed)
    patient_ids: list[str] = []
    tier_of: dict[str, str] = {}
    for t in args.tiers:
        for r in picked.get(t, []):
            patient_ids.append(r["patient_id"])
            tier_of[r["patient_id"]] = t
    return patient_ids, tier_of, stats, picked


def run(args) -> int:
    print("[compare] Loading agents + cohort...")
    agent_a = DataHarvesterAgent()
    agent_b = RaImagingAnalystAgent()
    agent_c = RiskSynthesizerAgent()

    # ── Patient selection ─────────────────────────────────────────────────────────
    picked = None    # OrderedDict tier -> [rule-based records] (only in stratified mode)
    if args.random_sample or args.all:
        patients = sample_patients(agent_a, args.n, args.seed, args.all)
        tier_of: dict[str, str] = {}
        selection_mode = "random" if not args.all else "all"
        sel_stats: dict = {}
    else:
        patients, tier_of, sel_stats, picked = _select_patients_stratified(
            agent_a, agent_b, agent_c, args
        )
        selection_mode = f"stratified (target {args.n_per_tier} per tier)"

    models = args.models or FLAGSHIP_MODELS
    n_calls = len(patients) * len(models) * 2

    print("\n" + "=" * 60)
    print(f"  Provider      : {args.provider}")
    print(f"  Models        : {len(models)}")
    for m in models:
        print(f"      - {m}")
    print(f"  Selection     : {selection_mode}")
    print(f"  Patients      : {len(patients)} (seed={args.seed})")
    if selection_mode.startswith("stratified") and sel_stats:
        for t in args.tiers:
            avail = sel_stats["available_per_tier"].get(t, 0)
            deliv = sel_stats["delivered_per_tier"].get(t, 0)
            flag = "  ⚠ under-filled" if deliv < args.n_per_tier else ""
            print(f"      {t:9s} cohort={avail:4d}  selected={deliv}{flag}")
    print(f"  Language      : {args.lang}")
    print(f"  Est. API calls: ~{n_calls}  ({'MOCK - no real calls' if args.mock else 'live - consumes credits'})")
    print("=" * 60 + "\n")

    if args.dry_run:
        print("[compare] --dry-run: nothing called. Patients: " + ", ".join(patients))
        return 0

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "") or SM.get_api_key(args.provider)
    if not args.mock and not api_key:
        print(f"[compare] ERROR: no API key for '{args.provider}'. Use --api-key, set "
              f"OPENROUTER_API_KEY, or save one in the app.")
        return 2

    clients = {m: make_client(args.provider, m, api_key, args.mock,
                              args.timeout, args.max_tokens, args.temperature)
               for m in models}

    # ── Build per-patient context up front (local, no network) ───────────────────
    contexts = []   # list of dicts: {pid, snap, ra, rec}
    for pid in patients:
        try:
            snap = agent_a.harvest(pid)
            ra = agent_b.analyse(snap)
        except Exception as e:
            print(f"[compare] WARN: skipping {pid}: {e}")
            continue
        contexts.append({
            "pid": pid, "snap": snap, "ra": ra,
            "rec": {
                "patient_id": pid,
                "snapshot": snapshot_brief(snap),
                "original": original_report(agent_c, snap, ra, args.lang),
                "models": [None] * len(models),   # filled by index, preserves order
            },
        })

    def _capture_one(ci, mi):
        """Run both LLM calls for one (patient, model); never raises."""
        ctx = contexts[ci]; m = models[mi]
        try:
            return capture_model(agent_c, ctx["snap"], ctx["ra"], clients[m], args.lang)
        except Exception as e:
            # Keep the exception type — it's the single most useful clue when
            # debugging "harness error: ..." rows in the results.
            err = f"harness error: {type(e).__name__}: {e}"
            return {"model": m,
                    "reasoning": {"success": False, "structured": False, "latency_s": 0,
                                  "tokens": 0, "error": err,
                                  "mechanism": "", "clinical_priority": "", "watchpoints": [],
                                  "patient_note": "", "model_caveat": "", "raw": ""},
                    "safety": {"available": False, "latency_s": 0, "tokens": 0,
                               "error": err, "flags": [], "summary": ""}}

    # ── Flat task list over EVERY (patient, model) pair ──────────────────────────
    tasks = [(ci, mi) for ci in range(len(contexts)) for mi in range(len(models))]
    total = len(tasks)
    conc = int(args.concurrency)
    workers = total if conc <= 0 else min(conc, total)   # <=0 means "max parallelism"
    workers = max(1, workers)
    print(f"[compare] {total} tasks across {len(contexts)} patients × {len(models)} models "
          f"| parallel workers: {workers}"
          + ("  (consider --concurrency to throttle if you hit rate limits)"
             if (workers > 8 and not args.mock) else ""))

    done = 0
    if workers == 1:
        for (ci, mi) in tasks:
            contexts[ci]["rec"]["models"][mi] = _capture_one(ci, mi)
            done += 1
            if args.sleep:
                time.sleep(args.sleep)
    else:
        from concurrent.futures import as_completed
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut = {ex.submit(_capture_one, ci, mi): (ci, mi) for (ci, mi) in tasks}
            for f in as_completed(fut):
                ci, mi = fut[f]
                out = f.result()
                contexts[ci]["rec"]["models"][mi] = out
                done += 1
                r = out["reasoning"]
                print(f"  [{done}/{total}] {contexts[ci]['pid']} · {out['model']}: "
                      f"{'ok' if r['success'] else 'FAIL'} {r['latency_s']}s "
                      f"{r['tokens']}tok flags={len(out['safety']['flags'])}")

    records = [c["rec"] for c in contexts]

    # Per-run subfolder so each run's files stay together.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "llm_mock" if args.mock else "llm"
    run_dir = os.path.join(args.output_dir, f"{tag}_run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    meta = {"timestamp": ts, "n_patients": len(records), "n_models": len(models),
            "lang": args.lang, "mock": args.mock, "models": models,
            "selection_mode": selection_mode,
            "n_per_tier": args.n_per_tier if not (args.random_sample or args.all) else None,
            "tiers": args.tiers, "selection_stats": sel_stats,
            "seed": args.seed, "run_dir": run_dir}

    jpath = os.path.join(run_dir, "llm_comparison.json")
    cpath = os.path.join(run_dir, "llm_comparison.csv")
    hpath = os.path.join(run_dir, "llm_comparison.html")
    write_json(records, meta, jpath)
    write_csv(records, cpath)
    write_html(records, meta, hpath, tier_of=tier_of)
    written = [jpath, cpath, hpath]

    # Optional in-place auto-retry of failed cells (off by default; uses API credits).
    if getattr(args, "auto_retry", 0):
        try:
            import rerun_failed as _rr   # lazy import avoids a circular dependency
            from argparse import Namespace
        except Exception as e:
            print(f"[compare] auto-retry unavailable: {e}")
        else:
            def _count_fail(jp):
                with open(jp, encoding="utf-8") as _f:
                    _d = json.load(_f)
                return sum(1 for _p in _d["patients"] for _m in _p["models"]
                           if _rr.reason_failed(_m["reasoning"]) or _rr.safety_failed(_m["safety"]))
            prev = _count_fail(jpath)
            for attempt in range(1, int(args.auto_retry) + 1):
                if prev == 0:
                    break
                print(f"\n[compare] Auto-retry {attempt}/{args.auto_retry}: {prev} failed cell(s) remaining...")
                _rr.run(Namespace(
                    in_path=jpath, output_dir=args.output_dir, provider=args.provider,
                    api_key=args.api_key, lang=args.lang, timeout=args.timeout,
                    max_tokens=args.max_tokens, temperature=args.temperature,
                    concurrency=(args.concurrency or 6), dry_run=False,
                    mock=args.mock, include_mock=False,
                ))
                now = _count_fail(jpath)
                if now >= prev:
                    print(f"[compare] Auto-retry: no further progress ({now} remaining); stopping.")
                    break
                prev = now
            print(f"[compare] Auto-retry complete — {prev} cell(s) still failing.")

    # Companion bilingual rule-based report HTML (only in stratified mode, since
    # we already have the picked records). Same patients, no LLM content.
    if picked is not None:
        try:
            from stratified_clinical_reports import (
                write_html as write_strat_html,
                write_json as write_strat_json,
                write_csv as write_strat_csv,
            )
            strat_meta = {
                "timestamp":  ts,
                "n_per_tier": args.n_per_tier,
                "tiers":      args.tiers,
                "seed":       args.seed,
                "selection":  "quantile-spaced by Cox 18-month event probability inside each tier",
            }
            strat_hpath = os.path.join(run_dir, "rule_based_reports_bilingual.html")
            strat_jpath = os.path.join(run_dir, "rule_based_reports.json")
            strat_cpath = os.path.join(run_dir, "rule_based_reports.csv")
            write_strat_html(picked, sel_stats, strat_meta, strat_hpath)
            write_strat_json(picked, sel_stats, strat_meta, strat_jpath)
            write_strat_csv(picked, strat_cpath)
            written.extend([strat_hpath, strat_jpath, strat_cpath])
        except Exception as e:
            print(f"[compare] WARN: could not write companion bilingual HTML: {e}")

    print("\n[compare] Wrote:")
    for p in written:
        print(f"  - {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare exact LLM outputs across flagship models.")
    # Selection — stratified is the new default (5 per tier across the four tiers).
    p.add_argument("--n-per-tier", type=int, default=5, dest="n_per_tier",
                   help="patients per integrated risk tier in stratified mode (default 5). "
                        "Picked quantile-spaced by Cox 18-month probability inside each tier.")
    p.add_argument("--tiers", nargs="+", default=list(ALL_TIERS),
                   help=f"tiers to include, in order (default: {' '.join(ALL_TIERS)})")
    p.add_argument("--random-sample", action="store_true", dest="random_sample",
                   help="legacy: ignore tiers and pick --n patients at random")
    p.add_argument("--n", type=int, default=20,
                   help="legacy random-mode patient count (only used with --random-sample)")
    p.add_argument("--all", action="store_true", help="use all patients (overrides stratification)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", nargs="*", default=None, help="override flagship list")
    p.add_argument("--provider", default="openrouter")
    p.add_argument("--lang", default="en", choices=["en", "zh"])
    p.add_argument("--api-key", default="", dest="api_key")
    p.add_argument("--output-dir", default="comparison_out", dest="output_dir")
    p.add_argument("--timeout", type=float, default=150.0)
    p.add_argument("--max-tokens", type=int, default=4000, dest="max_tokens")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--sleep", type=float, default=0.0,
                   help="seconds between calls (sequential mode only, i.e. --concurrency 1)")
    p.add_argument("--concurrency", type=int, default=0,
                   help="parallel (patient×model) tasks. 0 = max (all at once, default); "
                        "1 = sequential; N = cap at N workers if you hit rate limits")
    p.add_argument("--auto-retry", type=int, default=0, dest="auto_retry", metavar="N",
                   help="after the run, automatically re-run failed cells in place up to N "
                        "times (0 = off, default; uses your API key/credits)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--mock", action="store_true", help="offline smoke test (no network)")
    return p


def main():
    args = build_parser().parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n[compare] interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
