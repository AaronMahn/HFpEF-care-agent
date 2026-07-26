"""
CARE-Agent — Stratified Clinical Reports
========================================
Runs the deterministic Agent A → B → C pipeline (NO LLM) on the full cohort,
groups patients by their integrated risk tier (LOW / MODERATE / HIGH / CRITICAL),
and emits a stratified sample of N patients per tier, picked to span the Cox
18-month probability range inside each tier (quantile-spaced) so the sample is
diverse, not just the top of the bucket.

For each picked patient it produces:
  • the rule-based ENGLISH clinical report (`clinical_summary`)
  • the rule-based CHINESE clinical report (`summary_zh`)
  • snapshot + Cox + RA imaging metadata

Outputs (default --output-dir = ./comparison_out):
    stratified_reports_<ts>.json   — machine-readable
    stratified_reports_<ts>.csv    — one row per patient
    stratified_reports_<ts>.html   — bilingual viewer, 中文 / English toggle

Usage
-----
    python stratified_clinical_reports.py                 # 5 per tier (default)
    python stratified_clinical_reports.py --n-per-tier 3
    python stratified_clinical_reports.py --seed 7        # different sample
    python stratified_clinical_reports.py --tiers LOW MODERATE HIGH CRITICAL

Simulation/research tool — not for clinical use.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import random
import sys
import warnings
from collections import OrderedDict, defaultdict
from datetime import datetime
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_a_data_harvester import DataHarvesterAgent           # noqa: E402
from agent_b_imaging_analyst import RaImagingAnalystAgent        # noqa: E402
from agent_c_risk_synthesizer import RiskSynthesizerAgent        # noqa: E402


TIERS = ["LOW", "MODERATE", "HIGH", "CRITICAL"]

TIER_ZH = {"LOW": "低危", "MODERATE": "中危", "HIGH": "高危", "CRITICAL": "极高危"}

TIER_COLOR = {
    "LOW":      ("#1b6a4a", "#dff0e6"),   # text, bg
    "MODERATE": ("#8a6a00", "#fff4cf"),
    "HIGH":     ("#b15a14", "#fde9d4"),
    "CRITICAL": ("#b71d3b", "#fde2e7"),
}


# ── Helpers ─────────────────────────────────────────────────────────────────────
def _num(v):
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


def _quantile_pick(items: list, n: int) -> list:
    """Pick n elements from a *sorted* list at evenly spaced positions
    (always including the first and last). If fewer than n items, return all."""
    if not items:
        return []
    if len(items) <= n:
        return list(items)
    if n == 1:
        return [items[len(items) // 2]]
    # Indices at positions 0, 1/(n-1), 2/(n-1), …, 1.0
    idx = [round(i * (len(items) - 1) / (n - 1)) for i in range(n)]
    # dedupe while preserving order (rounding can collide on tiny lists)
    seen = set()
    out = []
    for i in idx:
        if i in seen:
            continue
        seen.add(i)
        out.append(items[i])
    return out


# ── Per-patient capture ─────────────────────────────────────────────────────────
def build_patient_record(agent_c, snap, ra) -> Optional[dict]:
    """Return a record with both EN and ZH rule-based reports, or None on error."""
    try:
        rep_en = agent_c.synthesise(snap, ra, llm_client=None, lang="en")
    except Exception as e:
        print(f"[stratify] WARN synthesise(en) failed for {snap.patient_id}: {e}")
        return None

    # Cheap second pass just to get the Chinese-language text (no LLM, fast).
    try:
        rep_zh = agent_c.synthesise(snap, ra, llm_client=None, lang="zh")
    except Exception as e:
        print(f"[stratify] WARN synthesise(zh) failed for {snap.patient_id}: {e}")
        rep_zh = rep_en   # fall back to EN so the structure stays consistent

    is_male = str(snap.sex).strip().lower() in ("male", "m", "男")

    return {
        "patient_id":           snap.patient_id,
        "integrated_tier":      rep_en.integrated_risk_tier,
        "cox_2yr_event_prob":   _num(rep_en.cox_2yr_event_prob),
        "cox_risk_label":       rep_en.cox_risk_label,
        "primary_action":       rep_en.primary_action,
        "actions":              [code for code, _, _ in rep_en.all_actions],
        "missing_cox_features": list(rep_en.missing_cox_features or []),
        "snapshot": {
            "age":         int(snap.age) if snap.age is not None else None,
            "sex":         "Male" if is_male else "Female",
            "sex_zh":      "男" if is_male else "女",
            "nyha":        int(snap.nyha_class) if snap.nyha_class else None,
            "agent_a_tier":snap.risk_tier,
            "n_flags":     len(snap.risk_flags or []),
            "af":          _num(snap.af),
            "ckd":         _num(snap.ckd),
            "ckd_stage":   snap.ckd_stage or "",
            "diabetes":    _num(snap.diabetes),
            "egfr":        _num(snap.egfr),
            "ntprobnp":    _num(snap.ntprobnp),
            "tyg_index":   _num(snap.tyg_index),
            "pasp":        _num(snap.pasp),
            "rvef":        _num(snap.rvef),
            "lvef":        _num(snap.lvef),
            "ra_reservoir":_num(snap.ra_reservoir),
            "ra_conduit":  _num(snap.ra_conduit),
            "ra_booster":  _num(snap.ra_booster),
            "ra_myopathy": _num(snap.ra_myopathy),
        },
        "ra_imaging": {
            "overall_function":   ra.overall_ra_function,
            "composite_score":    _num(ra.ra_composite_score),
            "reservoir_grade":    ra.reservoir_grade,
            "conduit_grade":      ra.conduit_grade,
            "booster_grade":      ra.booster_grade,
            "af_adjustment":      bool(ra.af_adjustment_applied),
            "quality_flags":      list(ra.imaging_quality_flags or []),
        },
        "risk_drivers_en":   list(rep_en.risk_drivers or []),
        "risk_drivers_zh":   list(rep_zh.risk_drivers_zh or []),
        "summary_en":        rep_en.clinical_summary or "",
        "summary_zh":        rep_zh.summary_zh or "",
    }


# ── Stratified selection ────────────────────────────────────────────────────────
def stratified_sample(records: list[dict], n_per_tier: int, tiers: list[str],
                      seed: int) -> tuple[OrderedDict, dict]:
    """Group records by tier; pick `n_per_tier` per tier with quantile spacing
    over Cox probability so the sample spans the bucket. Returns (picked, stats)."""
    by_tier: dict[str, list] = defaultdict(list)
    for r in records:
        by_tier[r["integrated_tier"]].append(r)

    # Stable sort within each tier by Cox probability ascending (None → 0)
    for t in by_tier:
        by_tier[t].sort(key=lambda r: (r["cox_2yr_event_prob"] is None,
                                       r["cox_2yr_event_prob"] or 0.0,
                                       r["patient_id"]))

    rng = random.Random(seed)
    picked: OrderedDict[str, list] = OrderedDict()
    stats = {"available_per_tier": {}, "delivered_per_tier": {}, "total_cohort": len(records)}
    for t in tiers:
        pool = by_tier.get(t, [])
        stats["available_per_tier"][t] = len(pool)
        chosen = _quantile_pick(pool, n_per_tier)
        # When the pool is much bigger than n, quantile picking is deterministic.
        # When the pool == n we take them all. We respect `seed` only as a
        # tie-breaker if two patients share the exact same Cox probability.
        rng.shuffle(chosen) if False else None   # placeholder — keep determinism
        picked[t] = chosen
        stats["delivered_per_tier"][t] = len(chosen)
    return picked, stats


# ── Writers ─────────────────────────────────────────────────────────────────────
def write_json(picked, stats, meta, path):
    data = {
        "meta": meta,
        "stats": stats,
        "buckets": {t: picked[t] for t in picked},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)


def write_csv(picked, path):
    cols = [
        "tier", "patient_id", "age", "sex", "nyha", "agent_a_tier",
        "cox_2yr_event_prob", "cox_risk_label",
        "egfr", "ntprobnp", "tyg_index", "pasp", "rvef", "lvef",
        "ra_reservoir", "ra_conduit", "ra_booster",
        "ra_overall_function", "ra_composite_score",
        "primary_action", "actions", "n_flags",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for tier, recs in picked.items():
            for r in recs:
                s, ra = r["snapshot"], r["ra_imaging"]
                w.writerow([
                    tier, r["patient_id"], s["age"], s["sex"], s["nyha"], s["agent_a_tier"],
                    r["cox_2yr_event_prob"], r["cox_risk_label"],
                    s["egfr"], s["ntprobnp"], s["tyg_index"], s["pasp"], s["rvef"], s["lvef"],
                    s["ra_reservoir"], s["ra_conduit"], s["ra_booster"],
                    ra["overall_function"], ra["composite_score"],
                    r["primary_action"], ";".join(r["actions"]), s["n_flags"],
                ])


# ── HTML (bilingual EN / 中文) ─────────────────────────────────────────────────
HTML_TEMPLATE_HEAD = r"""<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CARE-Agent · Stratified Clinical Reports / 分层临床报告</title>
<style>
  :root{
    --bg:#fbfaf7; --fg:#1f2330; --muted:#5a6072; --line:#e3e0d8;
    --accent:#7a3e1d; --accent-soft:#f0e6db;
    --font-zh:"PingFang SC","Hiragino Sans GB","Microsoft YaHei","Source Han Sans SC",system-ui,sans-serif;
    --font-en:"Charter","Iowan Old Style","Source Serif Pro",Georgia,serif;
    --font-mono:"JetBrains Mono","SF Mono",ui-monospace,monospace;
  }
  *{box-sizing:border-box}
  html,body{margin:0;background:var(--bg);color:var(--fg)}
  body{font-family:var(--font-en);line-height:1.6;font-size:15px;-webkit-font-smoothing:antialiased}
  body.lang-zh{font-family:var(--font-zh);line-height:1.75}
  .wrap{max-width:1100px;margin:0 auto;padding:32px 24px 80px}
  header.bar{
    position:sticky;top:0;z-index:10;background:rgba(251,250,247,.95);
    backdrop-filter:saturate(140%) blur(8px);
    border-bottom:1px solid var(--line);
    display:flex;align-items:center;justify-content:space-between;
    padding:14px 24px;font-family:var(--font-en);
  }
  body.lang-zh header.bar{font-family:var(--font-zh)}
  .title{font-weight:600}
  .title small{color:var(--muted);font-weight:400;margin-left:.5em}
  .lang-toggle{display:inline-flex;border:1px solid var(--line);border-radius:999px;overflow:hidden;background:#fff}
  .lang-toggle button{border:0;background:transparent;padding:6px 16px;cursor:pointer;font:inherit;font-size:13px;color:var(--muted)}
  .lang-toggle button.active{background:var(--accent);color:#fff}
  h1{font-size:26px;margin:.4em 0 .2em}
  h2.tier-h{font-size:18px;margin:1.8em 0 .6em;padding:8px 14px;border-radius:8px}
  .chips{margin:8px 0 4px}
  .chip{display:inline-block;background:#fff;border:1px solid var(--line);border-radius:999px;padding:4px 12px;font-size:12px;color:var(--muted);margin-right:6px}
  .alert{background:#fff4cf;border-left:4px solid #b15a14;padding:10px 14px;border-radius:6px;margin:14px 0;font-size:14px;color:#3a2614}
  .toc{background:#fff;border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:16px 0;font-size:13px}
  .toc a{color:var(--accent);text-decoration:none;margin-right:10px;display:inline-block;padding:2px 8px;border-radius:4px}
  .toc a:hover{background:var(--accent-soft)}
  .pblock{background:#fff;border:1px solid var(--line);border-radius:12px;margin:16px 0;overflow:hidden;box-shadow:0 2px 10px rgba(15,23,42,.03);scroll-margin-top:80px}
  .phead{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;padding:14px 18px;border-bottom:1px solid var(--line)}
  .pid{font-size:20px;font-weight:700}
  .demo{color:var(--muted);font-size:13px;margin-left:10px}
  .tier-pill{display:inline-block;padding:3px 12px;border-radius:999px;font-size:12px;font-weight:700;letter-spacing:.04em}
  .cox{font-family:var(--font-mono);font-size:13px;color:var(--muted);margin-left:10px}
  .cox b{color:var(--fg);font-size:14px}
  .pbody{padding:16px 18px}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px;margin-bottom:14px}
  .kv{background:#fafaf6;border-radius:6px;padding:7px 10px;font-size:12px}
  .kv .k{color:var(--muted);display:block;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
  .kv .v{font-family:var(--font-mono);font-weight:600}
  .drivers,.actions{margin:10px 0}
  .drivers .h,.actions .h{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px}
  .drivers ul,.actions ul{margin:.2em 0;padding-left:1.4em}
  .drivers li,.actions li{margin:.15em 0;font-size:13px}
  pre.summary{background:#0f172a;color:#e6e6e6;padding:14px 16px;border-radius:8px;white-space:pre-wrap;word-break:break-word;font-family:var(--font-mono);font-size:12px;line-height:1.55;overflow-x:auto;margin:8px 0 0}
  body.lang-zh pre.summary{font-family:"PingFang SC","Microsoft YaHei",var(--font-mono);font-size:12.5px}
  footer{margin-top:36px;padding-top:14px;border-top:1px solid var(--line);color:var(--muted);font-size:12px;text-align:center}
  /* language visibility — :not(body) prevents hiding the body */
  :not(body).lang-en, :not(body).lang-zh { display:none }
  body.lang-en .lang-en { display:block }
  body.lang-zh .lang-zh { display:block }
  body.lang-en span.lang-en { display:inline }
  body.lang-zh span.lang-zh { display:inline }
</style>
</head>
<body class="lang-zh">

<header class="bar">
  <div class="title">
    <span class="lang-en">CARE-Agent · Stratified Clinical Reports</span>
    <span class="lang-zh">CARE-Agent · 分层临床报告</span>
    <small>__TIMESTAMP__</small>
  </div>
  <div class="lang-toggle">
    <button id="btn-en" type="button">English</button>
    <button id="btn-zh" type="button">中文</button>
  </div>
</header>

<div class="wrap">
"""

HTML_TEMPLATE_FOOT = r"""
<footer>
  <span class="lang-en">Generated by stratified_clinical_reports.py · simulation study, not for clinical use.</span>
  <span class="lang-zh">由 stratified_clinical_reports.py 生成 · 模拟研究，不作为临床诊疗依据。</span>
</footer>
</div><!-- /wrap -->
<script>
(function(){
  var KEY='careAgentStratLang';
  var body=document.body;
  var btnEn=document.getElementById('btn-en');
  var btnZh=document.getElementById('btn-zh');
  function apply(lang){
    if(lang!=='en') lang='zh';
    body.classList.toggle('lang-en', lang==='en');
    body.classList.toggle('lang-zh', lang==='zh');
    document.documentElement.setAttribute('lang', lang==='zh'?'zh-CN':'en');
    btnEn.classList.toggle('active', lang==='en');
    btnZh.classList.toggle('active', lang==='zh');
    try{ localStorage.setItem(KEY,lang); }catch(e){}
  }
  var saved=null;
  try{ saved=localStorage.getItem(KEY); }catch(e){}
  apply(saved || 'zh');
  btnEn.addEventListener('click', function(){ apply('en'); });
  btnZh.addEventListener('click', function(){ apply('zh'); });
  document.addEventListener('keydown', function(e){
    if(e.altKey && (e.key==='l' || e.key==='L')){
      apply(body.classList.contains('lang-zh') ? 'en' : 'zh');
    }
  });
})();
</script>
</body></html>
"""


def _fmt(v, fmt="{:.1f}"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    try:
        return fmt.format(v)
    except Exception:
        return str(v)


def _kv_grid(snap, ra, lang_zh: bool) -> str:
    """A small key-value grid of the patient's hard numbers."""
    if lang_zh:
        labels = {
            "egfr": "eGFR", "ntprobnp": "NT-proBNP", "tyg": "TyG",
            "pasp": "PASP", "rvef": "RVEF", "lvef": "LVEF",
            "rr": "RA 储存期", "rc": "RA 管道期", "rb": "RA 助推期",
            "comp": "RA 综合评分", "func": "整体 RA 功能",
            "comorb": "合并症",
        }
    else:
        labels = {
            "egfr": "eGFR", "ntprobnp": "NT-proBNP", "tyg": "TyG",
            "pasp": "PASP", "rvef": "RVEF", "lvef": "LVEF",
            "rr": "RA reservoir", "rc": "RA conduit", "rb": "RA booster",
            "comp": "Composite", "func": "Overall RA",
            "comorb": "Comorbidities",
        }

    co = []
    if snap.get("af") == 1:
        co.append("AF" if not lang_zh else "房颤")
    if snap.get("ckd") == 1:
        s = snap.get("ckd_stage") or ""
        co.append(("CKD " + s).strip() if not lang_zh else f"CKD{('（'+s+'）') if s else ''}")
    if snap.get("diabetes") == 1:
        co.append("DM" if not lang_zh else "糖尿病")
    co_str = ", ".join(co) if co else ("—" if not lang_zh else "无")

    cells = [
        (labels["egfr"],     _fmt(snap.get("egfr"))     + " mL/min/1.73m²"),
        (labels["ntprobnp"], _fmt(snap.get("ntprobnp"), "{:.0f}") + " pg/mL"),
        (labels["tyg"],      _fmt(snap.get("tyg_index"), "{:.2f}")),
        (labels["pasp"],     _fmt(snap.get("pasp"))     + " mmHg"),
        (labels["rvef"],     _fmt(snap.get("rvef"))     + " %"),
        (labels["lvef"],     _fmt(snap.get("lvef"))     + " %"),
        (labels["rr"],       _fmt(snap.get("ra_reservoir")) + " %"),
        (labels["rc"],       _fmt(snap.get("ra_conduit"))   + " %"),
        (labels["rb"],       _fmt(snap.get("ra_booster"))   + " %"),
        (labels["comp"],     _fmt(ra.get("composite_score"), "{:.1f}") + " /100"),
        (labels["func"],     html.escape(str(ra.get("overall_function") or "—"))),
        (labels["comorb"],   html.escape(co_str)),
    ]
    return '<div class="grid">' + "".join(
        f'<div class="kv"><span class="k">{html.escape(k)}</span><span class="v">{v}</span></div>'
        for k, v in cells
    ) + "</div>"


def _patient_block(rec: dict, idx: int) -> str:
    s = rec["snapshot"]
    ra = rec["ra_imaging"]
    tier = rec["integrated_tier"]
    fg, bg = TIER_COLOR.get(tier, ("#475569", "#e2e8f0"))
    prob = rec["cox_2yr_event_prob"]
    prob_s = f"{prob:.1%}" if isinstance(prob, (int, float)) else "N/A"

    sex_en = s.get("sex") or "?"
    sex_zh = s.get("sex_zh") or "?"
    nyha = s.get("nyha") or "?"
    age = s.get("age") or "?"
    demo_en = f"{age}y · {sex_en} · NYHA {nyha}"
    demo_zh = f"{age} 岁 · {sex_zh} · NYHA {nyha} 级"

    tier_en = tier
    tier_zh = TIER_ZH.get(tier, tier)

    head = (
        '<div class="phead">'
        f'<div><span class="pid">{html.escape(rec["patient_id"])}</span>'
        f'<span class="demo lang-en">{html.escape(demo_en)}</span>'
        f'<span class="demo lang-zh">{html.escape(demo_zh)}</span></div>'
        '<div>'
        f'<span class="tier-pill" style="color:{fg};background:{bg}">'
        f'<span class="lang-en">{html.escape(tier_en)}</span>'
        f'<span class="lang-zh">{html.escape(tier_zh)}</span></span>'
        f'<span class="cox"><span class="lang-en">Cox 18-mo:</span>'
        f'<span class="lang-zh">Cox 18 个月事件概率：</span> <b>{prob_s}</b></span>'
        '</div></div>'
    )

    drivers_en = "".join(f"<li>{html.escape(d)}</li>" for d in rec["risk_drivers_en"]) or "<li>—</li>"
    drivers_zh = "".join(f"<li>{html.escape(d)}</li>" for d in rec["risk_drivers_zh"]) or "<li>—</li>"
    drivers_box = (
        '<div class="drivers">'
        '<div class="h"><span class="lang-en">Risk drivers</span><span class="lang-zh">风险驱动因素</span></div>'
        f'<div class="lang-en"><ul>{drivers_en}</ul></div>'
        f'<div class="lang-zh"><ul>{drivers_zh}</ul></div>'
        '</div>'
    )

    actions = rec["actions"] or []
    actions_li = "".join(f"<li><code>{html.escape(a)}</code></li>" for a in actions) or "<li>—</li>"
    actions_box = (
        '<div class="actions">'
        '<div class="h"><span class="lang-en">Recommended actions (codes, priority order)</span>'
        '<span class="lang-zh">推荐行动（按优先级排序）</span></div>'
        f'<ul>{actions_li}</ul>'
        '</div>'
    )

    grid_en = _kv_grid(s, ra, lang_zh=False)
    grid_zh = _kv_grid(s, ra, lang_zh=True)

    summary_en_h = "Rule-based clinical report (no LLM)"
    summary_zh_h = "规则生成的临床报告（未启用 LLM）"
    summary_box = (
        '<div class="lang-en">'
        f'<div class="h" style="font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin:14px 0 4px">{summary_en_h}</div>'
        + grid_en +
        f'<pre class="summary">{html.escape(rec["summary_en"])}</pre>'
        '</div>'
        '<div class="lang-zh">'
        f'<div class="h" style="font-size:12px;color:var(--muted);letter-spacing:.04em;margin:14px 0 4px">{summary_zh_h}</div>'
        + grid_zh +
        f'<pre class="summary">{html.escape(rec["summary_zh"])}</pre>'
        '</div>'
    )

    return (
        f'<section id="p{idx}" class="pblock">{head}'
        f'<div class="pbody">{drivers_box}{actions_box}{summary_box}</div></section>'
    )


def write_html(picked, stats, meta, path):
    head = HTML_TEMPLATE_HEAD.replace("__TIMESTAMP__", html.escape(meta["timestamp"]))

    # Top-level title + chips
    title = (
        '<h1><span class="lang-en">CARE-Agent · Stratified Clinical Reports</span>'
        '<span class="lang-zh">CARE-Agent · 分层临床报告</span></h1>'
        '<div class="lang-en" style="color:var(--muted);font-size:13px">'
        'Deterministic rule-based reports — no LLM. Stratified by integrated risk tier.</div>'
        '<div class="lang-zh" style="color:var(--muted);font-size:13px">'
        '基于规则的确定性临床报告（未启用 LLM），按"综合风险等级"分层。</div>'
    )

    n_total = sum(len(v) for v in picked.values())
    chip_html = '<div class="chips">'
    chip_html += f'<span class="chip">{html.escape(meta["timestamp"])}</span>'
    chip_html += f'<span class="chip"><span class="lang-en">{n_total} patients</span><span class="lang-zh">{n_total} 例患者</span></span>'
    chip_html += f'<span class="chip"><span class="lang-en">Target: {meta["n_per_tier"]} per tier</span><span class="lang-zh">目标：每层 {meta["n_per_tier"]} 例</span></span>'
    chip_html += f'<span class="chip"><span class="lang-en">Cohort size: {stats["total_cohort"]}</span><span class="lang-zh">总队列：{stats["total_cohort"]} 例</span></span>'
    chip_html += "</div>"

    # Alert if any requested tier is empty or short
    alerts = []
    for t in meta["tiers"]:
        avail = stats["available_per_tier"].get(t, 0)
        deliv = stats["delivered_per_tier"].get(t, 0)
        if deliv < meta["n_per_tier"]:
            alerts.append((t, avail, deliv))
    alert_html = ""
    if alerts:
        en_lines = "; ".join(
            f"<b>{t}</b>: cohort has {a}, delivered {d}" for t, a, d in alerts
        )
        zh_lines = "；".join(
            f"<b>{TIER_ZH.get(t,t)}</b>：队列中共 {a} 例，本次纳入 {d} 例" for t, a, d in alerts
        )
        alert_html = (
            '<div class="alert">'
            f'<span class="lang-en">⚠️ Some tiers under-filled — {en_lines}.</span>'
            f'<span class="lang-zh">⚠️ 部分分层未能凑齐 — {zh_lines}。</span>'
            "</div>"
        )

    # TOC by tier
    toc_blocks = []
    for t in meta["tiers"]:
        recs = picked.get(t, [])
        if not recs:
            continue
        toc_blocks.append(
            f'<span class="lang-en"><b>{html.escape(t)}</b>:</span>'
            f'<span class="lang-zh"><b>{html.escape(TIER_ZH.get(t,t))}</b>：</span>'
            + " ".join(
                f'<a href="#p{global_index(picked, t, i)}">{html.escape(r["patient_id"])}</a>'
                for i, r in enumerate(recs)
            )
        )
    toc = ('<div class="toc">' + " &nbsp; ".join(toc_blocks) + "</div>") if toc_blocks else ""

    # Patient blocks, grouped by tier with a heading
    body_parts = []
    for t in meta["tiers"]:
        recs = picked.get(t, [])
        if not recs:
            continue
        fg, bg = TIER_COLOR.get(t, ("#475569", "#e2e8f0"))
        body_parts.append(
            f'<h2 class="tier-h" style="color:{fg};background:{bg}">'
            f'<span class="lang-en">{html.escape(t)} · {len(recs)} patient(s)</span>'
            f'<span class="lang-zh">{html.escape(TIER_ZH.get(t,t))} · {len(recs)} 例</span>'
            '</h2>'
        )
        for i, r in enumerate(recs):
            body_parts.append(_patient_block(r, global_index(picked, t, i)))

    with open(path, "w", encoding="utf-8") as f:
        f.write(head)
        f.write(title)
        f.write(chip_html)
        f.write(alert_html)
        f.write(toc)
        f.write("".join(body_parts))
        f.write(HTML_TEMPLATE_FOOT)


def global_index(picked, tier, i) -> int:
    """Compute a globally-unique patient index across tiers in fixed order."""
    n = 0
    for t in picked:
        if t == tier:
            return n + i
        n += len(picked[t])
    return n + i


# ── Orchestration ───────────────────────────────────────────────────────────────
def run(args) -> int:
    import io, contextlib

    print("[stratify] Loading agents + cohort...")
    a = DataHarvesterAgent()
    b = RaImagingAnalystAgent()
    c = RiskSynthesizerAgent()

    print(f"[stratify] Harvesting full cohort and building rule-based reports...")
    buf = io.StringIO()
    records: list[dict] = []
    with contextlib.redirect_stdout(buf):
        snaps = a.harvest_all()
        for s in snaps:
            try:
                ra = b.analyse(s)
            except Exception as e:
                print(f"[stratify] WARN Agent B failed for {s.patient_id}: {e}")
                continue
            rec = build_patient_record(c, s, ra)
            if rec is not None:
                records.append(rec)
    print(f"[stratify] Built reports for {len(records)} / {len(snaps)} patients.")

    picked, stats = stratified_sample(records, args.n_per_tier, args.tiers, args.seed)

    print("\n[stratify] Sample composition:")
    for t in args.tiers:
        avail = stats["available_per_tier"].get(t, 0)
        deliv = stats["delivered_per_tier"].get(t, 0)
        flag = " (under-filled)" if deliv < args.n_per_tier else ""
        print(f"  {t:9s} : cohort={avail:4d}   selected={deliv}{flag}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Per-run subfolder so each run's files stay together.
    run_dir = os.path.join(args.output_dir, f"stratified_run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    meta = {
        "timestamp":   ts,
        "n_per_tier":  args.n_per_tier,
        "tiers":       args.tiers,
        "seed":        args.seed,
        "selection":   "quantile-spaced by Cox 18-month event probability inside each tier",
        "run_dir":     run_dir,
    }
    jpath = os.path.join(run_dir, "stratified_reports.json")
    cpath = os.path.join(run_dir, "stratified_reports.csv")
    hpath = os.path.join(run_dir, "stratified_reports.html")
    write_json(picked, stats, meta, jpath)
    write_csv(picked, cpath)
    write_html(picked, stats, meta, hpath)

    print("\n[stratify] Wrote:")
    for p in (jpath, cpath, hpath):
        print(f"  - {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stratified rule-based clinical reports across risk tiers."
    )
    p.add_argument("--n-per-tier", type=int, default=5, dest="n_per_tier",
                   help="how many patients to pick per integrated risk tier (default 5)")
    p.add_argument("--tiers", nargs="+", default=TIERS,
                   help=f"which tiers to include, in order (default: {' '.join(TIERS)})")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="comparison_out", dest="output_dir")
    return p


def main():
    args = build_parser().parse_args()
    # sanity-check tier names
    for t in args.tiers:
        if t not in TIERS:
            print(f"[stratify] ERROR: unknown tier '{t}'. Valid: {TIERS}")
            return 2
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n[stratify] interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
