"""
CARE-Agent — Full Pipeline Web Interface
==========================================
Three-agent clinical pipeline for HFpEF cardiorenal risk.
Run locally:  streamlit run app.py

Features
--------
• Bilingual UI — English / 中文 (toggle persists across sessions)
• Dataset merging — uploaded data is combined with the built-in cohort
• API key persistence — keys are saved to disk and restored on reload
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import asdict

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="CARE-Agent · Patient Risk Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from agent_a_data_harvester import DataHarvesterAgent
from agent_b_imaging_analyst import RaImagingAnalystAgent
from agent_c_risk_synthesizer import RiskSynthesizerAgent
from llm_client import LLMClient, PROVIDERS, PROVIDER_ORDER
from translations import get_translator, DECISION_RULES_ZH, RISK_TIER_ZH, RA_FUNCTION_ZH
import settings_manager as SM

# ── Initialise language from saved preference ──────────────────────────────────
if "lang" not in st.session_state:
    st.session_state["lang"] = SM.get_preference("lang", "en")

T = get_translator(st.session_state["lang"])

# ── Global styles ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .tier-badge {
        display: inline-block; padding: 4px 16px; border-radius: 20px;
        font-weight: 600; font-size: 15px; letter-spacing: 0.04em;
    }
    .flag-item {
        padding: 6px 12px; border-radius: 6px; margin: 3px 0; font-size: 14px;
    }
    .section-header {
        font-size: 13px; font-weight: 600; color: #6c757d;
        text-transform: uppercase; letter-spacing: 0.06em;
        margin-top: 1rem; margin-bottom: 0.5rem;
        border-bottom: 1px solid #e9ecef; padding-bottom: 4px;
    }
    .agent-note {
        background: #f8f9fa; border-left: 3px solid #4361ee;
        padding: 10px 14px; border-radius: 0 6px 6px 0;
        font-size: 13px; color: #495057; line-height: 1.6;
    }
    .agent-note-b {
        background: #f0f7ff; border-left: 3px solid #0ea5e9;
        padding: 10px 14px; border-radius: 0 6px 6px 0;
        font-size: 13px; color: #1e3a5f; line-height: 1.6;
    }
    .agent-note-c {
        background: #f0fdf4; border-left: 3px solid #16a34a;
        padding: 10px 14px; border-radius: 0 6px 6px 0;
        font-size: 13px; color: #14532d; line-height: 1.6;
    }
    .action-card {
        background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 10px 14px; margin: 6px 0; font-size: 13px; color: #374151;
    }
    .action-primary { border-left: 4px solid #16a34a; background: #f0fdf4; }
    .clinical-summary {
        font-family: monospace; font-size: 12px; background: #1e293b;
        color: #e2e8f0; padding: 16px; border-radius: 8px;
        white-space: pre-wrap; line-height: 1.7;
    }
    .stAlert { border-radius: 8px; }
    [data-testid="stMetricValue"] { font-size: 18px !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
TIER_COLORS = {
    "CRITICAL": ("#dc2626", "#fee2e2"),
    "HIGH":     ("#ea580c", "#fff7ed"),
    "MODERATE": ("#d97706", "#fffbeb"),
    "LOW":      ("#16a34a", "#f0fdf4"),
    "UNKNOWN":  ("#6b7280", "#f3f4f6"),
}
TIER_EMOJI = {"CRITICAL": "🔴", "HIGH": "🟠", "MODERATE": "🟡", "LOW": "🟢", "UNKNOWN": "⚪"}
FLAG_ICONS = {
    "CRITICAL": "🔴", "SEVERE": "🔴", "MYOPATHY": "🔴",
    "IMPAIRED": "🟠", "ELEVATED": "🟠", "HIGH": "🟠",
    "RV": "🟠", "PULMONARY": "🟠", "ATRIAL": "🟡", "CKD": "🟡",
}
RA_GRADE_COLORS = {
    "Normal":              ("#16a34a", "#f0fdf4"),
    "Mildly impaired":     ("#d97706", "#fffbeb"),
    "Moderately impaired": ("#ea580c", "#fff7ed"),
    "Severely impaired":   ("#dc2626", "#fee2e2"),
    "Severely impaired (RA myopathy)": ("#7c3aed", "#f5f3ff"),
    "Unknown":             ("#6b7280", "#f3f4f6"),
}
ACTION_PRIORITY_ICONS = {1: "🚨", 2: "⚠️", 3: "💡", 4: "📋"}

REQUIRED_COLS = [
    "name", "age", "sex", "egfr", "ntprobnp", "tyg_index",
    "ra_reservoir", "ra_conduit", "ra_booster", "lvef", "pasp",
    "followup_time", "event_all",
]

_HERE_APP    = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = (
    os.path.join(_HERE_APP, "data", "basic_data.xlsx")
    if os.path.exists(os.path.join(_HERE_APP, "data", "basic_data.xlsx"))
    else os.path.join(_HERE_APP, "basic_data.xlsx")
)


# ── Helpers ────────────────────────────────────────────────────────────────────
def flag_icon(flag_text: str) -> str:
    for key, icon in FLAG_ICONS.items():
        if key in flag_text:
            return icon
    return "🔵"


def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj) if not np.isnan(obj) else None
    return obj


def validate_columns(df: pd.DataFrame) -> list:
    return [c for c in REQUIRED_COLS if c not in df.columns]


def df_fingerprint(df: pd.DataFrame) -> str:
    """Stable MD5 hash of a DataFrame — used as cache key."""
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()


def cohort_stats(df: pd.DataFrame) -> dict:
    """Compute live cohort statistics from whichever dataset is active."""
    n = len(df)
    event_rate = f"{df['event_all'].mean():.1%}" if "event_all" in df.columns else "—"
    ckd_prev   = f"{df['ckd'].mean():.1%}"       if "ckd"       in df.columns else "—"
    followup   = f"{df['followup_time'].mean():.1f} mo" if "followup_time" in df.columns else "—"
    return {"n": n, "event_rate": event_rate, "ckd_prev": ckd_prev, "followup": followup}


def ra_grade_chip(grade: str, value: float = None, label: str = ""):
    color, bg = RA_GRADE_COLORS.get(grade, RA_GRADE_COLORS["Unknown"])
    val_str = f" {value:.1f}%" if value is not None else ""
    st.markdown(
        f'<span style="background:{bg};color:{color};border:1px solid {color};'
        f'border-radius:12px;padding:2px 10px;font-size:12px;font-weight:600;">'
        f'{label}{val_str} — {grade}</span>',
        unsafe_allow_html=True,
    )


def score_gauge(score: float, label: str = ""):
    if score is None:
        return
    color = ("#16a34a" if score >= 65 else
             "#d97706" if score >= 50 else
             "#ea580c" if score >= 35 else "#dc2626")
    st.markdown(f"""
<div style="margin: 6px 0 12px 0">
    <div style="font-size:12px;color:#6c757d;margin-bottom:3px">{label}</div>
    <div style="display:flex;align-items:center;gap:10px">
        <div style="flex:1;background:#e9ecef;border-radius:6px;height:14px;overflow:hidden">
            <div style="width:{min(score,100):.0f}%;background:{color};height:14px;border-radius:6px"></div>
        </div>
        <div style="font-size:16px;font-weight:700;color:{color};min-width:60px">{score:.1f}/100</div>
    </div>
</div>""", unsafe_allow_html=True)


# ── Clinical report card (modern dashboard style) ──────────────────────────────
def render_report_card(snap, ra_report, risk_rep, is_zh: bool):
    """Render the consolidated clinical report as modern dashboard cards.

    Pulls values straight from the agent dataclasses (so demographics are always
    correct) and renders a tier-coloured header, stat cards, RA-strain pills, a
    labs/echo grid, comorbidity chips, risk drivers and recommended actions.
    """
    def f1(v, unit=""):
        return "—" if v is None else f"{v:.1f}{unit}"

    def f0(v, unit=""):
        return "—" if v is None else f"{v:.0f}{unit}"

    tier       = risk_rep.integrated_risk_tier
    tier_color, tier_bg = TIER_COLORS.get(tier, TIER_COLORS["UNKNOWN"])
    tier_emoji = TIER_EMOJI.get(tier, "⚪")
    tier_label = RISK_TIER_ZH.get(tier, tier) if is_zh else tier
    is_male    = str(snap.sex).strip().lower() in ("male", "m", "男")
    prob_str   = f"{risk_rep.cox_2yr_event_prob:.1%}" if risk_rep.cox_2yr_event_prob is not None else "N/A"

    ra_fn      = ra_report.overall_ra_function
    ra_fn_disp = RA_FUNCTION_ZH.get(ra_fn, ra_fn) if is_zh else ra_fn
    ra_color, ra_bg = RA_GRADE_COLORS.get(ra_fn, RA_GRADE_COLORS["Unknown"])
    ra_score   = ra_report.ra_composite_score
    ra_score_str = f"{ra_score:.0f}/100" if ra_score is not None else "—"

    if is_zh:
        title    = "CARE-Agent 风险评估报告"
        sex_disp = "男" if is_male else "女"
        age_disp = f"{int(snap.age)}岁" if snap.age is not None else "年龄未知"
        nyha_disp = f"NYHA {int(snap.nyha_class)}级" if snap.nyha_class else "NYHA —"
        L = dict(integrated="综合风险等级", cox="Cox 18个月事件概率", ra_fn="右心房功能",
                 ra_score="RA 综合评分", strain="右心房应变分析", labs="关键生化与影像指标",
                 comorbid="合并症", drivers="主要风险驱动因素", actions="推荐临床行动方案",
                 reservoir="储存期", conduit="管道期", booster="助推期",
                 none="无主要合并症", primary="首要",
                 disclaimer="本报告由 CARE-Agent 生成 · 模拟研究，仅供科研参考，不作为临床诊疗依据")
    else:
        title    = "CARE-Agent Risk Report"
        sex_disp = "Male" if is_male else "Female"
        age_disp = f"{int(snap.age)} yrs" if snap.age is not None else "Age N/A"
        nyha_disp = f"NYHA {int(snap.nyha_class)}" if snap.nyha_class else "NYHA —"
        L = dict(integrated="Integrated Risk", cox="Cox 18-mo Event Prob.", ra_fn="RA Function",
                 ra_score="RA Composite", strain="Right-Atrial Strain", labs="Key Labs & Imaging",
                 comorbid="Comorbidities", drivers="Key Risk Drivers", actions="Recommended Actions",
                 reservoir="Reservoir", conduit="Conduit", booster="Booster",
                 none="No major comorbidities", primary="PRIMARY",
                 disclaimer="Generated by CARE-Agent · Simulation study — not for clinical use")

    h = []  # HTML fragments (joined with no separator to avoid markdown code blocks)

    # ── Outer card ──
    h.append('<div style="border:1px solid #e2e8f0;border-radius:16px;overflow:hidden;'
             'box-shadow:0 6px 24px rgba(15,23,42,0.08);background:#ffffff;margin:4px 0 8px 0">')

    # ── Header band (tier gradient) ──
    h.append(f'<div style="background:linear-gradient(135deg,{tier_color} 0%,{tier_color}cc 100%);'
             'padding:20px 26px;color:#ffffff;display:flex;align-items:center;justify-content:space-between;gap:18px;flex-wrap:wrap">')
    h.append('<div style="min-width:220px">'
             f'<div style="font-size:12px;letter-spacing:0.12em;text-transform:uppercase;opacity:0.85">{title}</div>'
             f'<div style="font-size:26px;font-weight:800;line-height:1.15;margin-top:2px">{snap.patient_id}</div>'
             f'<div style="font-size:13px;opacity:0.92;margin-top:6px">{age_disp} · {sex_disp} · {nyha_disp}</div>'
             '</div>')
    h.append('<div style="text-align:right;background:rgba(255,255,255,0.16);border-radius:12px;padding:10px 18px">'
             f'<div style="font-size:11px;letter-spacing:0.1em;text-transform:uppercase;opacity:0.9">{L["integrated"]}</div>'
             f'<div style="font-size:22px;font-weight:800;margin-top:2px">{tier_emoji} {tier_label}</div>'
             f'<div style="font-size:12px;opacity:0.92;margin-top:2px">{L["cox"]}: <b>{prob_str}</b></div>'
             '</div>')
    h.append('</div>')

    # ── Body ──
    h.append('<div style="padding:18px 22px 22px 22px">')

    # Stat cards row
    def stat(label, value, color):
        return ('<div style="flex:1;min-width:130px;background:#f8fafc;border:1px solid #eef2f7;'
                'border-radius:12px;padding:12px 14px">'
                f'<div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.05em">{label}</div>'
                f'<div style="font-size:20px;font-weight:800;color:{color};margin-top:4px">{value}</div></div>')
    h.append('<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:6px">')
    h.append(stat(L["integrated"], f"{tier_emoji} {tier_label}", tier_color))
    h.append(stat(L["cox"], prob_str, tier_color))
    h.append(stat(L["ra_fn"], ra_fn_disp, ra_color))
    h.append(stat(L["ra_score"], ra_score_str, ra_color))
    h.append('</div>')

    # RA strain pills
    def pill(label, value, grade):
        g_color, g_bg = RA_GRADE_COLORS.get(grade, RA_GRADE_COLORS["Unknown"])
        return (f'<div style="flex:1;min-width:150px;background:{g_bg};border:1px solid {g_color};'
                'border-radius:12px;padding:10px 14px">'
                f'<div style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:0.05em">{label}</div>'
                f'<div style="font-size:18px;font-weight:800;color:{g_color};margin-top:3px">{f1(value, "%")}</div></div>')
    h.append(f'<div style="font-size:12px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;margin:14px 0 6px 0">{L["strain"]}</div>')
    h.append('<div style="display:flex;gap:10px;flex-wrap:wrap">')
    h.append(pill(L["reservoir"], ra_report.ra_reservoir, ra_report.reservoir_grade))
    h.append(pill(L["conduit"],   ra_report.ra_conduit,   ra_report.conduit_grade))
    h.append(pill(L["booster"],   ra_report.ra_booster,   ra_report.booster_grade))
    h.append('</div>')

    # Labs & imaging grid
    labs = [
        ("eGFR", f1(snap.egfr), "mL/min/1.73m²"),
        ("CKD", snap.ckd_stage or "—", ""),
        ("Creatinine", f0(snap.creatinine), "μmol/L"),
        ("NT-proBNP", f0(snap.ntprobnp), "pg/mL"),
        ("TyG", f"{snap.tyg_index:.2f}" if snap.tyg_index is not None else "—", ""),
        ("LVEF", f1(snap.lvef), "%"),
        ("LAVI", f1(snap.lavi), "mL/m²"),
        ("E/e'", f1(snap.mitral_ee), ""),
        ("PASP", f0(snap.pasp), "mmHg"),
        ("RVEF", f1(snap.rvef), "%"),
        ("TAPSE", f0(snap.tapse), "mm"),
    ]
    h.append(f'<div style="font-size:12px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;margin:16px 0 6px 0">{L["labs"]}</div>')
    h.append('<div style="display:flex;gap:8px;flex-wrap:wrap">')
    for lab, val, unit in labs:
        unit_html = f'<span style="font-size:10px;color:#94a3b8;margin-left:3px">{unit}</span>' if unit else ""
        h.append('<div style="flex:1;min-width:96px;background:#ffffff;border:1px solid #eef2f7;'
                 'border-radius:10px;padding:8px 10px">'
                 f'<div style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.04em">{lab}</div>'
                 f'<div style="font-size:15px;font-weight:700;color:#0f172a;margin-top:2px">{val}{unit_html}</div></div>')
    h.append('</div>')

    # Comorbidity chips
    chips = []
    if snap.af == 1:           chips.append(("AF / 房颤" if is_zh else "AF", "#dc2626"))
    if snap.ckd == 1:          chips.append((f"CKD {snap.ckd_stage or ''}".strip(), "#ea580c"))
    if snap.diabetes == 1:     chips.append(("糖尿病" if is_zh else "Diabetes", "#d97706"))
    if snap.hypertension == 1: chips.append(("高血压" if is_zh else "Hypertension", "#7c3aed"))
    if snap.cad == 1:          chips.append(("冠心病" if is_zh else "CAD", "#0ea5e9"))
    h.append(f'<div style="font-size:12px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;margin:16px 0 6px 0">{L["comorbid"]}</div>')
    if chips:
        h.append('<div style="display:flex;gap:6px;flex-wrap:wrap">')
        for label, c in chips:
            h.append(f'<span style="background:{c}1a;color:{c};border:1px solid {c}55;border-radius:20px;'
                     f'padding:3px 12px;font-size:12px;font-weight:600">{label}</span>')
        h.append('</div>')
    else:
        h.append(f'<div style="font-size:13px;color:#94a3b8">{L["none"]}</div>')

    # Risk drivers
    drivers = risk_rep.risk_drivers_zh if is_zh else risk_rep.risk_drivers
    if drivers:
        h.append(f'<div style="font-size:12px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;margin:16px 0 6px 0">{L["drivers"]}</div>')
        h.append('<div style="display:flex;flex-direction:column;gap:5px">')
        for d in drivers:
            h.append('<div style="background:#f8fafc;border-left:3px solid '
                     f'{tier_color};border-radius:0 8px 8px 0;padding:7px 12px;font-size:13px;color:#334155">{d}</div>')
        h.append('</div>')

    # Recommended actions
    actions = risk_rep.all_actions[:5]
    if actions:
        h.append(f'<div style="font-size:12px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;margin:16px 0 6px 0">{L["actions"]}</div>')
        h.append('<div style="display:flex;flex-direction:column;gap:7px">')
        for i, (code, priority, text) in enumerate(actions):
            p_icon = ACTION_PRIORITY_ICONS.get(priority, "📌")
            atext  = DECISION_RULES_ZH.get(code, text) if is_zh else text
            is_primary = (i == 0)
            bcolor = tier_color if is_primary else "#e2e8f0"
            bgc    = tier_bg if is_primary else "#ffffff"
            badge  = (f'<span style="background:{tier_color};color:#fff;border-radius:6px;font-size:9px;'
                      f'font-weight:700;padding:1px 7px;letter-spacing:0.08em;margin-left:8px">{L["primary"]}</span>'
                      if is_primary else "")
            h.append(f'<div style="background:{bgc};border:1px solid {bcolor};border-left:4px solid {bcolor};'
                     'border-radius:8px;padding:9px 13px;font-size:13px;color:#334155">'
                     f'<div style="font-weight:700;color:#1e293b">{p_icon} {code.replace("_"," ")}{badge}</div>'
                     f'<div style="color:#64748b;margin-top:3px;line-height:1.55">{atext}</div></div>')
        h.append('</div>')

    # Footer
    h.append(f'<div style="margin-top:18px;padding-top:12px;border-top:1px solid #eef2f7;'
             f'font-size:11px;color:#94a3b8;text-align:center">{L["disclaimer"]}</div>')

    h.append('</div></div>')  # close body + outer card
    st.markdown("".join(h), unsafe_allow_html=True)


# ── Cached resource loaders ────────────────────────────────────────────────────
@st.cache_resource
def _load_agent_a_builtin(data_path: str):
    return DataHarvesterAgent(data_path=data_path)


@st.cache_resource
def _load_agent_a_merged(data_path: str, fingerprint: str, _extra_df: pd.DataFrame):
    """fingerprint drives the cache key; _extra_df (underscore) bypasses Streamlit hashing."""
    return DataHarvesterAgent(data_path=data_path, extra_df=_extra_df)


@st.cache_resource
def load_agent_b():
    return RaImagingAnalystAgent()


@st.cache_resource
def load_agent_c():
    return RiskSynthesizerAgent()


@st.cache_data
def get_patient_list(_agent):
    return sorted(_agent._raw["_pid"].tolist())


@st.cache_data(show_spinner=False)
def run_pipeline(
    patient_id: str,
    lang: str,
    llm_enabled: bool,
    provider: str,
    model: str,
    key_hash: str,
    dataset_fp: str,
    _agent_a,
    _agent_b,
    _agent_c,
    _llm_client,
):
    """Run the full A→B→C pipeline and cache the result.

    The cache key is (patient, lang, llm config, dataset). Agent objects and the
    LLM client are underscore-prefixed so Streamlit does not try to hash them.
    Caching is what stops the two LLM calls from re-firing on every rerun (tab
    clicks, widget changes), which is the main cause of the slow LLM rendering.
    """
    snap      = _agent_a.harvest(patient_id)
    ra_report = _agent_b.analyse(snap)
    risk_rep  = _agent_c.synthesise(snap, ra_report, llm_client=_llm_client, lang=lang)
    return snap, ra_report, risk_rep


# ── LLM selection state ────────────────────────────────────────────────────────
# Defaults so the pipeline cache key is always defined, even when LLM is off.
selected_provider_key = ""
selected_model        = ""
api_key               = ""

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=60)
    st.title(T("sidebar_title"))
    st.caption(T("sidebar_caption"))
    st.divider()

    # ── I. Language toggle ─────────────────────────────────────────────────────
    lang_choice = st.radio(
        T("language"),
        options=["English", "中文"],
        index=0 if st.session_state["lang"] == "en" else 1,
        horizontal=True,
    )
    new_lang = "en" if lang_choice == "English" else "zh"
    if new_lang != st.session_state["lang"]:
        st.session_state["lang"] = new_lang
        SM.set_preference("lang", new_lang)
        st.rerun()

    st.divider()

    # ── II. Dataset (merge mode) ───────────────────────────────────────────────
    st.markdown(f"### {T('dataset_section')}")
    dataset_mode = st.radio(
        T("data_source"),
        [T("use_builtin"), T("upload_merge")],
        label_visibility="collapsed",
    )

    extra_df   = None   # populated when user uploads a file
    merge_info = {}     # counts shown in success message

    if dataset_mode == T("use_builtin"):
        st.caption(T("builtin_caption"))

    else:  # upload & merge
        with st.expander(T("requirements"), expanded=True):
            st.markdown("""
**File format**
- Excel file (`.xlsx`), any sheet name
- One row per patient, first row = column headers
- Patient identifier column must be named `name`

**Required columns**

| Column | Type | Description |
|---|---|---|
| `name` | text | Patient identifier |
| `age` | integer | Age in years |
| `sex` | 0 / 1 | 1 = Female, 0 = Male |
| `egfr` | decimal | eGFR (mL/min/1.73m²) |
| `ntprobnp` | decimal | NT-proBNP (pg/mL) |
| `logntprobnp` | decimal | log(NT-proBNP) |
| `tyg_index` | decimal | TyG index |
| `ra_reservoir` | decimal | RA reservoir strain (%) |
| `ra_conduit` | decimal | RA conduit strain (%) |
| `ra_booster` | decimal | RA booster strain (%) |
| `lvef` | decimal | LVEF (%) |
| `pasp` | integer | PASP (mmHg) |
| `followup_time` | decimal | Follow-up duration (months) |
| `event_all` | 0 / 1 | Composite outcome |

**Recommended columns** *(used for risk flags if present)*

`ckd`, `af`, `diabetes`, `hypertension`, `cad`, `smoking`,
`rvef`, `rvfwls`, `tapse`, `mitral_ee`, `lavi`, `lvmi`,
`ra_myopathy`, `scr`, `fbg`, `tg`, `bmi`, `bsa`,
`sbp`, `dbp`, `nyha_class`, `tyg_tertile`,
`SGLT-2i`, `diuretics`, `bate_blockers`, `acri_arb`, `insulin`,
`hf_rehos`, `cv_death`

**Value rules**
- Binary columns: use `0` or `1` only
- Missing values: leave blank (not "N/A" or "-")
- Strain values: positive % (sign not required)
""")

        uploaded = st.file_uploader(
            T("upload_label"),
            type=["xlsx"],
            label_visibility="collapsed",
        )

        if uploaded is not None:
            try:
                upload_df = pd.read_excel(uploaded, sheet_name=0)
                missing_cols = validate_columns(upload_df)
                if missing_cols:
                    st.error(f"{T('missing_cols')}: {', '.join(missing_cols)}")
                else:
                    upload_clean = upload_df.dropna(subset=["name"])
                    # Pre-read built-in to compute counts (not cached here)
                    builtin_df = pd.read_excel(
                        DEFAULT_PATH, sheet_name="395全量HFpEF数据"
                    ).dropna(subset=["name"])
                    # Match the merge logic: UPDATE only on a unique name match; novel or
                    # ambiguous (repeated) names are added as NEW patients.
                    _bcounts  = builtin_df["name"].value_counts()
                    _uc       = upload_clean["name"].map(lambda v: int(_bcounts.get(v, 0)))
                    n_updated = int((_uc == 1).sum())
                    n_amb     = int((_uc > 1).sum())
                    n_new     = int((_uc != 1).sum())
                    n_total   = len(builtin_df) + n_new
                    merge_info = {
                        "n_new": n_new, "n_updated": n_updated, "n_amb": n_amb,
                        "n_builtin": len(builtin_df), "n_total": n_total,
                    }
                    extra_df = upload_clean
                    st.success(T("merge_success", **merge_info))
                    if n_updated > 0:
                        st.caption(T("merge_updated", n_updated=n_updated))
                    if n_amb > 0:
                        st.caption(T("merge_ambiguous", n_amb=n_amb))
            except Exception as e:
                st.error(f"{T('read_error')}: {e}")
        else:
            st.info(T("no_file"))

        with open(DEFAULT_PATH, "rb") as f:
            st.download_button(
                T("download_template"),
                data=f.read(),
                file_name="CARE_Agent_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    st.divider()

    # ── Load agents (after dataset section is resolved) ────────────────────────
    try:
        if extra_df is not None:
            fp = df_fingerprint(extra_df)
            agent_a = _load_agent_a_merged(DEFAULT_PATH, fp, extra_df)
        else:
            agent_a = _load_agent_a_builtin(DEFAULT_PATH)

        agent_b      = load_agent_b()
        agent_c      = load_agent_c()
        patient_list = get_patient_list(agent_a)

        if not patient_list:
            st.error(T("no_patients"))
            st.stop()
    except Exception as e:
        st.error(f"{T('init_error')}: {e}")
        st.stop()

    # ── III. LLM settings — with API key persistence ───────────────────────────
    st.markdown(f"### {T('ai_reasoning')}")
    llm_enabled = st.toggle(T("enable_llm"), value=False)

    llm_client = None
    if llm_enabled:
        selected_provider_key = st.selectbox(
            T("provider"),
            options=PROVIDER_ORDER,
            format_func=lambda k: f"{PROVIDERS[k]['group'].split()[0]} {PROVIDERS[k]['label']}",
            index=PROVIDER_ORDER.index("openrouter"),
        )
        pinfo = PROVIDERS[selected_provider_key]

        base_url_override = None
        if pinfo.get("custom_base_url"):
            base_url_override = st.text_input(T("server_url"), value=pinfo["base_url"])

        if pinfo["models"]:
            _custom_label = "✏️ " + ("自定义模型…" if st.session_state["lang"] == "zh" else "Custom model…")
            _model_choice = st.selectbox(
                T("model"), options=pinfo["models"] + [_custom_label], index=0,
            )
            if _model_choice == _custom_label:
                selected_model = st.text_input(
                    T("model_name"),
                    value=pinfo["default_model"],
                    placeholder="provider/model-id",
                    help=("Enter the exact model ID as the provider expects it. "
                          "Use this if a listed model returns a 404 / 'model not found'."),
                )
            else:
                selected_model = _model_choice
        else:
            selected_model = st.text_input(
                T("model_name"),
                value=pinfo["default_model"],
                placeholder="llama3.2" if selected_provider_key == "ollama" else "type model name",
                help=(
                    "Ollama: run `ollama list` in terminal to see installed models.\n"
                    "LM Studio: use the model identifier shown in the app."
                ),
            )

        api_key = ""
        if pinfo["needs_key"]:
            # Pre-fill from persisted settings
            saved_key = SM.get_api_key(selected_provider_key)
            key_help  = f"Get your key at: {pinfo['api_key_url']}" if pinfo["api_key_url"] else ""

            api_key = st.text_input(
                pinfo["api_key_label"],
                type="password",
                value=saved_key,
                placeholder=pinfo["key_placeholder"],
                help=key_help,
            )

            col_save, col_clear = st.columns([1, 1])
            with col_save:
                if st.button(T("save_key"), use_container_width=True, disabled=not api_key):
                    SM.set_api_key(selected_provider_key, api_key)
                    st.success(T("key_saved_msg"))
            with col_clear:
                if st.button(T("clear_key"), use_container_width=True, disabled=not saved_key):
                    SM.clear_api_key(selected_provider_key)
                    st.rerun()

            if not api_key:
                st.info(f"Enter your {pinfo['label']} API key above to enable AI reasoning.")

        ready = (not pinfo["needs_key"]) or bool(api_key)
        if ready and selected_model:
            llm_client = LLMClient(
                provider=selected_provider_key,
                api_key=api_key,
                model=selected_model,
                base_url_override=base_url_override,
                temperature=0.3,
                max_tokens=4000,   # generous budget so reasoning models don't truncate (matches benchmark/compare)
            )
            if not pinfo["needs_key"]:
                st.success(f"✓ {pinfo['label']} — {selected_model}")
            elif api_key:
                st.success(f"✓ {pinfo['label']} / {selected_model}")

        with st.expander(T("provider_guide"), expanded=False):
            st.markdown("""
**🏠 Local (no API key, completely free & offline)**
- **Ollama** — `ollama pull qwen3:8b` or `deepseek-r1:7b` · [ollama.com](https://ollama.com)
- **LM Studio** — GUI app, start local server on port 1234 · [lmstudio.ai](https://lmstudio.ai)

**🌐 International APIs**
- **OpenRouter** — one key for 100+ models incl. Claude, GPT-4.1, Gemini 2.5 · [openrouter.ai/keys](https://openrouter.ai/keys) *(free tier available)*
- **Groq** — very fast inference, free tier · [console.groq.com](https://console.groq.com/keys)
- **OpenAI** — GPT-4.1, o3, o4-mini · [platform.openai.com](https://platform.openai.com/api-keys)
- **Mistral** — Magistral reasoning + Mistral Large · [console.mistral.ai](https://console.mistral.ai)
- **Together AI** — Llama 4, Kimi K2, DeepSeek R1 · [api.together.ai](https://api.together.ai)

**🇨🇳 Chinese APIs**
- **Qwen** — 通义千问 Qwen3 (235B MoE), QwQ · [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com/apiKey)
- **Kimi** — 月之暗面 moonshot-v1 · [platform.moonshot.cn](https://platform.moonshot.cn/console/api-keys)
- **DeepSeek** — deepseek-chat (V3) + deepseek-reasoner (R1) · [platform.deepseek.com](https://platform.deepseek.com/api_keys)
- **Zhipu** — 智谱 GLM-4.5, GLM-Z1 reasoning · [open.bigmodel.cn](https://open.bigmodel.cn/usercenter/apikeys)
- **Baichuan** — 百川 Baichuan4-Turbo · [platform.baichuan-ai.com](https://platform.baichuan-ai.com)

💡 **Recommended for getting started:** OpenRouter (free credits) or Groq (free tier) · For local: Ollama + `qwen3:8b`
""")

    # ── Saved key manager ──────────────────────────────────────────────────────
    saved_providers = SM.list_saved_providers()
    if saved_providers:
        st.divider()
        with st.expander(f"⚙️ {'API 密钥管理' if st.session_state['lang'] == 'zh' else 'API Key Manager'}", expanded=False):
            st.caption(f"{'已保存' if st.session_state['lang'] == 'zh' else 'Saved'}: {', '.join(PROVIDERS[p]['label'] for p in saved_providers)}")
            for p in saved_providers:
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"`{PROVIDERS[p]['label']}`")
                if c2.button("🗑", key=f"del_{p}", help=f"Remove saved key for {PROVIDERS[p]['label']}"):
                    SM.clear_api_key(p)
                    st.rerun()

    st.divider()

    # ── Patient selector ───────────────────────────────────────────────────────
    st.markdown(f"### {T('select_patient')}")
    selected = st.selectbox(
        T("search_by_name"),
        options=[T("select_placeholder")] + patient_list,
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(f"### {T('pipeline_section')}")
    st.markdown(T("pipeline_desc"))
    st.divider()
    st.caption(f"CARE-Agent · {T('simulation_note')} · April 2026")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.title(T("page_title"))
st.caption(T("app_caption"))

if selected == T("select_placeholder"):
    st.info(T("select_patient_hint"))
    st.divider()

    # ── Cohort overview — computed live from the active (possibly merged) dataset
    st.markdown(f"### {T('cohort_overview')}")
    stats    = cohort_stats(agent_a._raw)
    ds_label = T("merged_label") if extra_df is not None else T("builtin_label")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(T("total_patients"), str(stats["n"]))
    col2.metric(T("dataset_label"),  ds_label)
    col3.metric(T("event_rate"),     stats["event_rate"])
    col4.metric(T("ckd_prev"),       stats["ckd_prev"])
    col5.metric(T("mean_followup"),  stats["followup"])

    st.divider()
    st.markdown(f"### {T('how_it_works')}")
    if st.session_state["lang"] == "zh":
        steps = [
            "1. 在左侧边栏**选择患者**",
            "2. **Agent A** 自动采集所有临床数据，验证质量并标记风险",
            "3. **Agent B** 验证并分类三期 RA 应变，计算 RA 综合功能评分",
            "4. **Agent C** 运行预训练 Cox 模型，整合所有输出，生成临床建议",
            "5. 通过下方四个标签页**浏览结果**",
        ]
    else:
        steps = [
            "1. **Select a patient** in the sidebar",
            "2. **Agent A** automatically harvests all clinical data, validates quality, and flags risks",
            "3. **Agent B** validates and classifies the three RA strain phases, computing a composite RA function score",
            "4. **Agent C** runs the pre-trained Cox model, integrates all outputs, and generates clinical recommendations",
            "5. **Browse results** across the four tabs below",
        ]
    for s in steps:
        st.markdown(s)
    st.image("https://img.icons8.com/color/200/mind-map.png", width=120)

else:
    # ── Run the three-agent pipeline (cached) ──────────────────────────────────
    spinner_msg = (
        f"{T('running')} {selected}... {T('llm_enabled_note')}"
        if llm_client
        else f"{T('running')} {selected}..."
    )

    # Cache-key components — result is reused across reruns unless one changes,
    # so the LLM calls don't re-fire when you switch tabs or toggle widgets.
    _ds_fp    = df_fingerprint(extra_df) if extra_df is not None else "builtin"
    _provider = selected_provider_key if llm_client else ""
    _model    = selected_model if llm_client else ""
    _key_hash = (
        hashlib.md5(api_key.encode()).hexdigest()[:8]
        if (llm_client and api_key) else ""
    )

    # Small "re-run" control so a cached (possibly failed) LLM result can be refreshed.
    _rerun_label = "🔄 重新分析" if st.session_state["lang"] == "zh" else "🔄 Re-run analysis"
    _rc1, _rc2 = st.columns([1, 5])
    with _rc1:
        if st.button(_rerun_label, use_container_width=True,
                     help="清除缓存并重新运行（含 LLM 推理）" if st.session_state["lang"] == "zh"
                          else "Clear the cache and run the pipeline again (incl. LLM reasoning)"):
            run_pipeline.clear()
            st.rerun()

    with st.spinner(spinner_msg):
        try:
            snap, ra_report, risk_rep = run_pipeline(
                selected,
                st.session_state["lang"],
                bool(llm_client),
                _provider,
                _model,
                _key_hash,
                _ds_fp,
                agent_a,
                agent_b,
                agent_c,
                llm_client,
            )
        except Exception as e:
            st.error(f"{T('pipeline_error')}: {e}")
            st.stop()

    # ── Top risk banner ────────────────────────────────────────────────────────
    tier_color, tier_bg = TIER_COLORS.get(risk_rep.integrated_risk_tier, TIER_COLORS["UNKNOWN"])
    tier_emoji = TIER_EMOJI.get(risk_rep.integrated_risk_tier, "⚪")
    prob_str   = f"{risk_rep.cox_2yr_event_prob:.1%}" if risk_rep.cox_2yr_event_prob is not None else "N/A"

    st.markdown(f"""
<div style="background:{tier_bg};border:2px solid {tier_color};border-radius:12px;
            padding:16px 28px;margin-bottom:1rem;display:flex;align-items:center;gap:20px">
    <div style="font-size:40px">{tier_emoji}</div>
    <div style="flex:1">
        <div style="font-size:24px;font-weight:700;color:{tier_color}">
            {risk_rep.integrated_risk_tier} {T('risk')} &nbsp;
            <span style="font-size:16px;font-weight:400;color:#6c757d">{T('integrated')}</span>
        </div>
        <div style="font-size:13px;color:#6c757d;margin-top:2px">
            {snap.patient_id} &nbsp;·&nbsp; Cox 18-mo: <strong>{prob_str}</strong>
            &nbsp;·&nbsp; RA: <strong>{ra_report.overall_ra_function}</strong>
            &nbsp;·&nbsp; {len(snap.risk_flags)} flag(s)
        </div>
    </div>
    <div style="text-align:right">
        <div style="font-size:13px;color:#6c757d">{T('primary_action_label')}</div>
        <div style="font-size:15px;font-weight:600;color:{tier_color}">
            {DECISION_RULES_ZH.get(risk_rep.primary_action, risk_rep.primary_action).split("—")[0].split("。")[0].strip() if st.session_state["lang"] == "zh" else risk_rep.primary_action.replace("_"," ")}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab_overview, tab_a, tab_b, tab_c = st.tabs([
        T("tab_overview"), T("tab_a"), T("tab_b"), T("tab_c"),
    ])

    # Shared language flag (used across all tabs)
    _is_zh = st.session_state["lang"] == "zh"

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    with tab_overview:
        st.markdown(f"#### {T('pipeline_summary')}")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(T("agent_a_risk_tier"),  RISK_TIER_ZH.get(snap.risk_tier, snap.risk_tier) if _is_zh else snap.risk_tier)
        _ra_fn_display = RA_FUNCTION_ZH.get(ra_report.overall_ra_function, ra_report.overall_ra_function) if _is_zh else ra_report.overall_ra_function.split(" (")[0]
        m2.metric(T("ra_function"),        _ra_fn_display)
        m3.metric(T("ra_composite_score"), f"{ra_report.ra_composite_score}/100" if ra_report.ra_composite_score else "—")
        m4.metric(T("cox_2yr_prob"),       prob_str)
        m5.metric(T("integrated_risk"),    RISK_TIER_ZH.get(risk_rep.integrated_risk_tier, risk_rep.integrated_risk_tier) if _is_zh else risk_rep.integrated_risk_tier)

        st.divider()
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown(f"**{T('risk_drivers')}**")
            _drivers_to_show = risk_rep.risk_drivers_zh if st.session_state["lang"] == "zh" else risk_rep.risk_drivers
            for driver in _drivers_to_show:
                st.markdown(f"• {driver}")

            if snap.risk_flags:
                st.markdown(f"**{T('agent_a_flags')}**")
                for flag in snap.risk_flags:
                    icon  = flag_icon(flag)
                    parts = flag.split(":", 1)
                    tag   = parts[0].strip().replace("_", " ").title()
                    detail = parts[1].strip() if len(parts) > 1 else ""
                    st.markdown(
                        f'<div class="flag-item" style="background:#f8f9fa;border-left:3px solid {tier_color}">'
                        f'{icon} <strong>{tag}</strong> — {detail}</div>',
                        unsafe_allow_html=True,
                    )

        with col_right:
            st.markdown(f"**{T('recommended_actions')}**")
            for i, (code, priority, text) in enumerate(risk_rep.all_actions):
                p_icon = ACTION_PRIORITY_ICONS.get(priority, "📌")
                card_class = "action-card action-primary" if i == 0 else "action-card"
                _action_text = DECISION_RULES_ZH.get(code, text) if _is_zh else text
                _code_label  = code.replace("_", " ")
                st.markdown(
                    f'<div class="{card_class}">'
                    f'{p_icon} <strong>{_code_label}</strong><br>'
                    f'<span style="color:#6c757d">{_action_text}</span></div>',
                    unsafe_allow_html=True,
                )

        st.divider()
        render_report_card(snap, ra_report, risk_rep, _is_zh)
        _txt_label = "📄 纯文本报告（便于复制）" if _is_zh else "📄 Plain-text report (for copy/paste)"
        with st.expander(_txt_label, expanded=False):
            st.code(risk_rep.summary_zh if _is_zh else risk_rep.clinical_summary, language=None)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — AGENT A · DATA
    # ══════════════════════════════════════════════════════════════════════════
    with tab_a:
        st.markdown(
            f'<div class="agent-note">🤖 <strong>Agent A:</strong> {snap.agent_a_notes}</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        left, right = st.columns([1.1, 1])

        with left:
            st.markdown(f'<div class="section-header">{T("clinical_risk_flags")}</div>',
                        unsafe_allow_html=True)
            if snap.risk_flags:
                for flag in snap.risk_flags:
                    icon  = flag_icon(flag)
                    parts = flag.split(":", 1)
                    tag   = parts[0].strip().replace("_", " ").title()
                    detail = parts[1].strip() if len(parts) > 1 else ""
                    st.markdown(
                        f'<div class="flag-item" style="background:#f8f9fa;border-left:3px solid {tier_color}">'
                        f'{icon} <strong>{tag}</strong> — {detail}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.success(T("no_flags"))

            st.divider()
            st.markdown(f'<div class="section-header">{T("data_quality")}</div>',
                        unsafe_allow_html=True)
            dq = snap.data_quality
            q1, q2 = st.columns(2)
            q1.metric(T("status"),       dq.get("status", "—"))
            q2.metric(T("completeness"), dq.get("completeness", "—"))
            if dq.get("missing_critical"):
                st.warning(f"{T('missing_critical')}: {', '.join(dq['missing_critical'])}")
            if dq.get("implausible"):
                st.error(f"{T('implausible_vals')}: {', '.join(dq['implausible'])}")

        with right:
            st.markdown(f'<div class="section-header">{T("demographics")}</div>',
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric(T("age"),        f"{int(snap.age)}y" if snap.age else "—")
            c2.metric(T("sex"),        snap.sex or "—")
            c3.metric(T("nyha_class"), snap.nyha_class or "—")
            c4, c5 = st.columns(2)
            c4.metric(T("bmi"), f"{snap.bmi:.1f}" if snap.bmi else "—")
            c5.metric(T("af"),  T("yes") if snap.af == 1 else T("no"))

            st.markdown(f'<div class="section-header">{T("renal_function")}</div>',
                        unsafe_allow_html=True)
            r1, r2, r3 = st.columns(3)
            r1.metric(T("egfr"),       f"{snap.egfr:.1f}" if snap.egfr else "—", help="mL/min/1.73m²")
            r2.metric(T("ckd_stage"),  snap.ckd_stage or "—")
            r3.metric(T("creatinine"), f"{snap.creatinine:.0f}" if snap.creatinine else "—", help="μmol/L")

            st.markdown(f'<div class="section-header">{T("biomarkers")}</div>',
                        unsafe_allow_html=True)
            b1, b2, b3 = st.columns(3)
            b1.metric("NT-proBNP",  f"{snap.ntprobnp:.0f}" if snap.ntprobnp else "—", help="pg/mL")
            b2.metric("TyG Index",  f"{snap.tyg_index:.2f}" if snap.tyg_index else "—")
            b3.metric("TyG Tertile", snap.tyg_tertile or "—")

            st.markdown(f'<div class="section-header">{T("ra_strain")}</div>',
                        unsafe_allow_html=True)
            ra1, ra2, ra3 = st.columns(3)
            ra_r = snap.ra_reservoir
            ra_r_delta = T("low_delta") if ra_r and ra_r < 30 else (T("normal_delta") if ra_r else None)
            ra1.metric(T("ra_reservoir"), f"{ra_r:.1f}%" if ra_r else "—",
                       delta=ra_r_delta, delta_color="inverse")
            ra2.metric(T("ra_conduit"),   f"{snap.ra_conduit:.1f}%" if snap.ra_conduit else "—")
            ra3.metric(T("ra_booster"),   f"{snap.ra_booster:.1f}%" if snap.ra_booster else "—")
            myopathy_str = T("positive") if snap.ra_myopathy == 1 else T("negative")
            st.markdown(f"**{T('ra_myopathy')}:** {myopathy_str}")

            st.markdown(f'<div class="section-header">{T("right_heart")}</div>',
                        unsafe_allow_html=True)
            e1, e2, e3 = st.columns(3)
            e1.metric("PASP",  f"{snap.pasp} mmHg" if snap.pasp else "—")
            e2.metric("RVEF",  f"{snap.rvef:.1f}%" if snap.rvef else "—")
            e3.metric("TAPSE", f"{snap.tapse} mm" if snap.tapse else "—")

        with st.expander(T("comorbidities_meds"), expanded=False):
            cm1, cm2 = st.columns(2)
            with cm1:
                st.markdown(f"**{T('comorbidities')}**")
                for name, val in [
                    ("Hypertension", snap.hypertension), ("Diabetes", snap.diabetes),
                    ("CKD", snap.ckd), ("CAD", snap.cad),
                    ("Atrial Fibrillation", snap.af), ("Smoking", snap.smoking),
                ]:
                    st.markdown(f"{'✅' if val == 1 else '⬜'} {name}")
            with cm2:
                st.markdown(f"**{T('medications')}**")
                for name, val in [
                    ("SGLT-2 Inhibitor", snap.sglt2i), ("Diuretics", snap.diuretics),
                    ("Beta-Blocker", snap.beta_blocker), ("ACEi / ARB", snap.acei_arb),
                    ("Insulin", snap.insulin),
                ]:
                    st.markdown(f"{'✅' if val == 1 else '⬜'} {name}")

        with st.expander(T("left_heart"), expanded=False):
            lh1, lh2, lh3, lh4 = st.columns(4)
            lh1.metric("LVEF", f"{snap.lvef:.1f}%" if snap.lvef else "—")
            lh2.metric("LAVI", f"{snap.lavi:.1f}" if snap.lavi else "—", help="mL/m²")
            lh3.metric("E/e'", f"{snap.mitral_ee:.1f}" if snap.mitral_ee else "—")
            lh4.metric("LVMI", f"{snap.lvmi:.1f}" if snap.lvmi else "—", help="g/m²")

        st.divider()
        col_dl1, col_dl2 = st.columns([1, 3])
        with col_dl1:
            json_data = json.dumps(make_serializable(asdict(snap)), indent=2, ensure_ascii=False)
            st.download_button(
                T("download_agent_a"),
                data=json_data,
                file_name=f"agent_a_{snap.patient_id}_{snap.harvest_timestamp[:10]}.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_dl2:
            st.caption(T("agent_a_json_caption"))

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — AGENT B · RA IMAGING
    # ══════════════════════════════════════════════════════════════════════════
    with tab_b:
        st.markdown(
            f'<div class="agent-note-b">🖼️ <strong>Agent B:</strong> {ra_report.agent_b_notes}</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        b_status_colors = {"PASS": "🟢", "WARN": "🟡", "FAIL": "🔴", "PENDING": "⚪"}
        st.markdown(
            f"**{T('imaging_qc')}:** "
            f"{b_status_colors.get(ra_report.agent_b_status, '⚪')} `{ra_report.agent_b_status}`"
        )
        for qf in ra_report.imaging_quality_flags:
            st.warning(qf)

        st.divider()
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(f'<div class="section-header">{T("ra_phase")}</div>', unsafe_allow_html=True)

            st.markdown(f"**{T('reservoir_phase')}** *(储存期)*")
            ra_grade_chip(ra_report.reservoir_grade, ra_report.ra_reservoir, T("ra_reservoir"))
            st.markdown("")

            st.markdown(f"**{T('conduit_phase')}** *(管道期)*")
            ra_grade_chip(ra_report.conduit_grade, ra_report.ra_conduit, T("ra_conduit"))
            st.markdown("")

            st.markdown(f"**{T('booster_phase')}** *(助推期)*")
            if ra_report.af_adjustment_applied:
                st.caption(T("af_booster_note"))
            ra_grade_chip(ra_report.booster_grade, ra_report.ra_booster, T("ra_booster"))
            st.markdown("")

            st.divider()
            st.markdown(f'<div class="section-header">{T("strain_consistency")}</div>',
                        unsafe_allow_html=True)
            cons_emoji = {"Consistent": "✅", "Inconsistent": "⚠️", "Cannot check": "❓"}.get(
                ra_report.strain_consistency, "❓"
            )
            st.markdown(f"{cons_emoji} **{ra_report.strain_consistency}**")
            if ra_report.consistency_note:
                st.caption(ra_report.consistency_note)

        with col2:
            st.markdown(f'<div class="section-header">{T("overall_ra")}</div>', unsafe_allow_html=True)
            fn_color, fn_bg = RA_GRADE_COLORS.get(ra_report.overall_ra_function, RA_GRADE_COLORS["Unknown"])
            st.markdown(f"""
<div style="background:{fn_bg};border:2px solid {fn_color};border-radius:10px;
            padding:14px 20px;margin-bottom:12px;text-align:center">
    <div style="font-size:20px;font-weight:700;color:{fn_color}">{ra_report.overall_ra_function}</div>
</div>
""", unsafe_allow_html=True)

            score_gauge(ra_report.ra_composite_score, "RA Composite Score / RA综合评分 (0–100)")

            st.markdown(f'<div class="section-header">{T("derived_metrics")}</div>',
                        unsafe_allow_html=True)
            d1, d2 = st.columns(2)
            d1.metric(T("rc_ratio"),
                      f"{ra_report.reservoir_conduit_ratio:.2f}" if ra_report.reservoir_conduit_ratio else "—",
                      help=T("rc_ratio_help"))
            d2.metric(T("booster_fraction"),
                      f"{ra_report.booster_fraction:.1f}%" if ra_report.booster_fraction else "—",
                      help=T("booster_frac_help"))

            if ra_report.af_adjustment_applied:
                st.info(f"**AF Adjustment Applied**\n\n{ra_report.booster_af_note}")
            if ra_report.ckd_flag and ra_report.ra_reservoir is not None and ra_report.ra_reservoir < 30:
                st.warning(
                    "CKD with impaired RA reservoir — volume overload likely contributing to RA remodelling."
                )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — AGENT C · RISK & ACTIONS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_c:
        st.markdown(
            f'<div class="agent-note-c">🧠 <strong>Agent C:</strong> {risk_rep.agent_c_notes}</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(f'<div class="section-header">{T("cox_output")}</div>', unsafe_allow_html=True)

            if risk_rep.cox_model_complete:
                st.success(T("cox_complete"))
            else:
                feats_str = ", ".join(risk_rep.missing_cox_features)
                st.warning(
                    f"⚠️ {T('cox_imputed', n=len(risk_rep.missing_cox_features), feats=feats_str)}"
                )

            cox_m1, cox_m2 = st.columns(2)
            cox_m1.metric(T("2yr_prob"), prob_str, help=T("2yr_prob_help"))
            cox_m2.metric(T("cox_label"), risk_rep.cox_risk_label)

            if risk_rep.cox_2yr_event_prob is not None:
                p = risk_rep.cox_2yr_event_prob
                p_color = ("#dc2626" if p >= 0.231 else "#ea580c" if p >= 0.121 else
                           "#d97706" if p >= 0.078 else "#16a34a")
                st.markdown(f"""
<div style="margin:8px 0 16px 0">
    <div style="font-size:11px;color:#6c757d;margin-bottom:4px">18-month event probability</div>
    <div style="background:#e9ecef;border-radius:6px;height:18px;overflow:hidden;position:relative">
        <div style="width:{p*100:.1f}%;background:{p_color};height:18px;border-radius:6px"></div>
        <div style="position:absolute;top:0;left:{min(p*100,85):.0f}%;font-size:11px;
                    font-weight:700;color:white;line-height:18px;padding-left:4px">{p:.1%}</div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:10px;color:#9ca3af;margin-top:2px">
        <span>Low (&lt;7.8%)</span><span>Moderate (7.8–12.1%)</span>
        <span>High (12.1–23.1%)</span><span>Critical (≥23.1%)</span>
    </div>
</div>
""", unsafe_allow_html=True)

            st.caption(T("cox_caption"))
            st.divider()

            st.markdown(f'<div class="section-header">{T("integrated_tier")}</div>',
                        unsafe_allow_html=True)
            int_color, int_bg = TIER_COLORS.get(risk_rep.integrated_risk_tier, TIER_COLORS["UNKNOWN"])
            int_emoji = TIER_EMOJI.get(risk_rep.integrated_risk_tier, "⚪")
            st.markdown(f"""
<div style="background:{int_bg};border:1px solid {int_color};border-radius:8px;
            padding:10px 16px;text-align:center">
    <span style="font-size:22px">{int_emoji}</span>
    <span style="font-size:18px;font-weight:700;color:{int_color};margin-left:8px">
        {RISK_TIER_ZH.get(risk_rep.integrated_risk_tier, risk_rep.integrated_risk_tier) if _is_zh else risk_rep.integrated_risk_tier}</span>
    <div style="font-size:11px;color:#6c757d;margin-top:4px">{T('integrated_note')}</div>
</div>
""", unsafe_allow_html=True)

            st.markdown(f"**{T('risk_drivers')}**")
            _c_drivers = risk_rep.risk_drivers_zh if _is_zh else risk_rep.risk_drivers
            for d in _c_drivers:
                st.markdown(f"• {d}")

        with col2:
            st.markdown(f'<div class="section-header">{T("decision_rules")}</div>',
                        unsafe_allow_html=True)
            for i, (code, priority, text) in enumerate(risk_rep.all_actions):
                p_icon     = ACTION_PRIORITY_ICONS.get(priority, "📌")
                is_primary = (i == 0)
                border_color = int_color if is_primary else "#e2e8f0"
                bg_color     = int_bg    if is_primary else "#ffffff"
                _action_text_c = DECISION_RULES_ZH.get(code, text) if _is_zh else text
                st.markdown(
                    f'<div style="background:{bg_color};border:1px solid {border_color};'
                    f'border-left:4px solid {border_color};border-radius:8px;padding:10px 14px;'
                    f'margin:6px 0;font-size:13px">'
                    + (f'<div style="font-size:10px;font-weight:700;color:{int_color};'
                       f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px">'
                       f'{T("primary_badge")}</div>' if is_primary else "")
                    + f'{p_icon} <strong>{code.replace("_"," ")}</strong><br>'
                    f'<span style="color:#6c757d">{_action_text_c}</span></div>',
                    unsafe_allow_html=True,
                )

        # ── LLM reasoning block ────────────────────────────────────────────────
        st.divider()
        st.markdown(f"### {T('ai_reasoning_title')}")

        if risk_rep.llm_available:
            _llm_caption = (
                f"Generated by **{risk_rep.llm_model_used}** via {risk_rep.llm_provider.title()} "
                f"· Reasoning: {risk_rep.llm_tokens_used} tokens"
                + (f" · Safety check: {risk_rep.llm_contradiction_tokens} tokens" if risk_rep.llm_contradiction_available else "")
                + f" · {T('simulation_note')}"
            )

            if risk_rep.llm_structured:
                # ── Structured 5-section display ──────────────────────────────
                sec1, sec2 = st.columns([3, 2])

                with sec1:
                    # Mechanism
                    st.markdown(f"**{T('llm_mechanism_title')}**")
                    st.markdown(
                        f"<div style='background:#f0f4ff;border-left:4px solid #3b82f6;"
                        f"border-radius:0 8px 8px 0;padding:14px 16px;font-size:13.5px;line-height:1.75'>"
                        f"{risk_rep.llm_mechanism.replace(chr(10), '<br>')}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("")

                    # Clinical Priority
                    st.markdown(f"**{T('llm_priority_title')}**")
                    st.markdown(
                        f"<div style='background:#fff7ed;border-left:4px solid #f97316;"
                        f"border-radius:0 8px 8px 0;padding:14px 16px;font-size:13.5px;line-height:1.75'>"
                        f"{risk_rep.llm_clinical_priority.replace(chr(10), '<br>')}</div>",
                        unsafe_allow_html=True,
                    )

                with sec2:
                    # Watchpoints
                    st.markdown(f"**{T('llm_watchpoints_title')}**")
                    for wp in risk_rep.llm_watchpoints:
                        st.markdown(
                            f"<div style='background:#f0fdf4;border-left:3px solid #22c55e;"
                            f"border-radius:0 6px 6px 0;padding:8px 12px;margin:4px 0;font-size:13px'>"
                            f"👁 {wp}</div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown("")

                    # Patient Note
                    st.markdown(f"**{T('llm_patient_note_title')}**")
                    st.markdown(
                        f"<div style='background:#fdf4ff;border:1px solid #d946ef;"
                        f"border-radius:8px;padding:12px 14px;font-size:13px;line-height:1.7;"
                        f"font-style:italic'>"
                        f"🗣 {risk_rep.llm_patient_note}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("")

                    # Model Caveat
                    if risk_rep.llm_model_caveat:
                        st.caption(f"{T('llm_caveat_title')}: {risk_rep.llm_model_caveat}")

            else:
                # Fallback: raw text output
                st.markdown(f"**{T('llm_raw_fallback')}**")
                st.markdown(
                    f"<div style='background:#0f1117;border:1px solid #1e40af;border-left:4px solid #3b82f6;"
                    f"border-radius:8px;padding:16px 20px;color:#e2e8f0;font-size:14px;line-height:1.8'>"
                    f"{risk_rep.llm_reasoning.replace(chr(10), '<br>')}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.caption(_llm_caption)

            # ── Contradiction / Safety Check ───────────────────────────────────
            st.divider()
            st.markdown(f"**{T('contradiction_title')}**")

            if risk_rep.llm_contradiction_available:
                _flag_colors = {
                    "warning": ("#dc2626", "#fef2f2", T("contradiction_flag_warn")),
                    "caution": ("#d97706", "#fffbeb", T("contradiction_flag_caution")),
                    "info":    ("#2563eb", "#eff6ff", T("contradiction_flag_info")),
                }
                if not risk_rep.llm_contradiction_flags:
                    st.success(T("contradiction_safe"))
                else:
                    st.warning(T("contradiction_unsafe"))
                    for flag in risk_rep.llm_contradiction_flags:
                        sev   = flag.get("severity", "caution").lower()
                        code  = flag.get("action_code", "")
                        concern = flag.get("concern", "")
                        color, bg, label = _flag_colors.get(sev, _flag_colors["caution"])
                        st.markdown(
                            f"<div style='background:{bg};border-left:4px solid {color};"
                            f"border-radius:0 8px 8px 0;padding:10px 14px;margin:6px 0;font-size:13px'>"
                            f"<span style='font-weight:700;color:{color}'>{label}</span>"
                            f"<span style='color:#6b7280;font-size:11px;margin-left:8px'>[{code}]</span><br>"
                            f"{concern}</div>",
                            unsafe_allow_html=True,
                        )
                if risk_rep.llm_contradiction_summary:
                    st.caption(f"{T('contradiction_summary')}: {risk_rep.llm_contradiction_summary}")
            elif risk_rep.llm_contradiction_error:
                st.caption(f"{T('contradiction_unavailable')} ({risk_rep.llm_contradiction_error})")

        elif risk_rep.llm_error:
            st.error(f"LLM error: {risk_rep.llm_error}")
        else:
            st.info(T("llm_enable_hint"))

        st.divider()
        render_report_card(snap, ra_report, risk_rep, _is_zh)

        # Plain-text versions for copy / archival
        with st.expander(("📄 中文纯文本报告" if _is_zh else "📄 Chinese plain-text report"),
                         expanded=False):
            st.code(risk_rep.summary_zh, language=None)
        with st.expander(("📄 英文纯文本报告" if _is_zh else "📄 English plain-text report"),
                         expanded=False):
            st.code(risk_rep.clinical_summary, language=None)

        st.divider()
        col_dl1, col_dl2 = st.columns([1, 3])
        with col_dl1:
            combined = {
                "patient_id":           snap.patient_id,
                "pipeline_timestamp":   snap.harvest_timestamp,
                "agent_a":              make_serializable(asdict(snap)),
                "agent_b":              make_serializable(asdict(ra_report)),
                "agent_c":              make_serializable(asdict(risk_rep)),
            }
            full_json = json.dumps(combined, indent=2, ensure_ascii=False)
            st.download_button(
                T("download_full"),
                data=full_json,
                file_name=f"care_agent_{snap.patient_id}_{snap.harvest_timestamp[:10]}.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_dl2:
            st.caption(T("full_json_caption"))
