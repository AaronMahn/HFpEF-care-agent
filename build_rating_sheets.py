"""
CARE-Agent — Human Rater Sheet Builder
=======================================
Takes the output of `compare_llm_outputs.py` (an `llm_run_<ts>/` folder) and
generates one self-contained, blinded, bilingual HTML rating sheet per doctor.

The sheet asks for six 0–2 scores per (patient, model) on the rubric below;
results are auto-saved to localStorage and exported as JSON for aggregation.

Rubric (per `_llm_reasoning` + `_llm_contradiction_check` output, full pair):
  1. 机制解释准确性 / Mechanism accuracy            (0–2)
  2. 风险驱动因素识别 / Risk-driver identification  (0–2)
  3. 治疗建议合理性 / Treatment-recommendation rationality (0–2)
  4. 安全标记质量 / Safety-flag quality              (0–2)
  5. 幻觉/错误推断 / Hallucination (REVERSE-SCORED)  (0–2)   ← higher = worse
  6. 临床可读性 / Clinical readability               (0–2)

  0 = poor / 错误或无关  ·  1 = adequate / 部分正确  ·  2 = excellent / 准确完整

Blinding strategy:
  - Per doctor: random model-letter map (A/B/C/D/E/F),
    re-rolled PER PATIENT so brand pattern recognition is impossible.
  - Per doctor: patients are also shuffled.
  - The un-blinding maps go into <output-dir>/blinding/<doctor>.json
    — keep these private until aggregation.

Usage
-----
    python build_rating_sheets.py --llm-run comparison_out/llm_run_<ts>
    python build_rating_sheets.py --llm-run comparison_out/llm_mock_run_<ts> \\
        --doctors "Rater B" "Rater C" "Rater A"
    python build_rating_sheets.py --llm-run <dir> --seed 7 --lang en

Output (default --output-dir = <llm-run-dir>/ratings):
    rating_sheet_<doctor>.html       — give one to each doctor
    blinding/<doctor>.json           — keep private; needed for un-blinding
    README_for_doctors.md            — instructions for the rater
    patient_information.xlsx         — un-blinded master table: every patient in
                                       the run with all clinical variables, tier,
                                       Cox probability, actions, risk drivers and
                                       the base report text. PRIVATE — contains
                                       real patient IDs; do NOT send to doctors.
                                       (disable with --no-excel)
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
import re
import sys
from datetime import datetime

# Vendor → short display name (used only inside the un-blinding map; never
# shown to the doctor).
VENDOR_DISPLAY = {
    "anthropic":  "Claude",
    "google":     "Gemini",
    "openai":     "GPT",
    "deepseek":   "DeepSeek",
    "moonshotai": "Kimi",
    "qwen":       "Qwen",
}

RUBRIC = [
    # (key, en_label, zh_label, reverse_scored?)
    ("mechanism_accuracy", "Mechanism accuracy",            "机制解释准确性",     False),
    ("risk_driver_id",     "Risk driver identification",    "风险驱动因素识别",   False),
    ("treatment_rationale","Treatment recommendation rationality", "治疗建议合理性", False),
    ("safety_flag_quality","Safety-flag quality",           "安全标记质量",       False),
    ("hallucination",      "Hallucination (reverse-scored — higher = worse)",
                                                             "幻觉 / 错误推断（反向计分 — 分越高越差）", True),
    ("clinical_readability","Clinical readability",          "临床可读性",        False),
]

# Score scale (inclusive). 0 = poor, SCORE_MAX = excellent (or worst, for the
# reverse-scored hallucination dimension). Change here to switch the scale.
SCORE_MAX = 2

TIER_COLOR = {
    "LOW":      ("#1b6a4a", "#dff0e6"),
    "MODERATE": ("#8a6a00", "#fff4cf"),
    "HIGH":     ("#b15a14", "#fde9d4"),
    "CRITICAL": ("#b71d3b", "#fde2e7"),
    "Unknown":  ("#475569", "#e2e8f0"),
}
TIER_ZH = {"LOW": "低危", "MODERATE": "中危", "HIGH": "高危", "CRITICAL": "极高危"}

# Severity order for sorting the master patient table (most severe first).
TIER_RANK = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3, "Unknown": 4}

# Source dataset (the full 84-column bio table Agent A reads). Resolved the same
# way as agent_a_data_harvester.DATA_PATH.
_HERE = os.path.dirname(os.path.abspath(__file__))
BASIC_DATA_PATH = (
    os.path.join(_HERE, "data", "basic_data.xlsx")
    if os.path.exists(os.path.join(_HERE, "data", "basic_data.xlsx"))
    else os.path.join(_HERE, "basic_data.xlsx")
)
BASIC_DATA_SHEET = "395全量HFpEF数据"

# Columns that encode a real identity (never shown to doctors) and follow-up
# OUTCOMES (hidden from the blinded sheet so they can't bias the rating; they
# DO go into the private master Excel).
IDENTITY_COLS = {"name"}
OUTCOME_COLS = ["followup_time", "event_all", "hf_rehos", "cv_death"]

# Binary 0/1 columns rendered as Yes/No (是/否).
BINARY_COLS = {
    "smoking", "ckd", "hypertension", "diabetes", "cad", "af",
    "SGLT-2i", "insulin", "bate_blockers", "acri_arb", "ccb", "aspirin",
    "diuretics", "statin", "tyg_high", "ra_myopathy",
    "G3a：eGFR 45–59", "G3b：eGFR 30–44", "G4–5：eGFR <30",
}

# Full clinical/biological data shown on the doctor sheet, grouped for
# readability. Each field: (source_column, EN label, ZH label, unit).
# Outcomes and `name` are deliberately excluded here.
FULL_BIO_GROUPS = [
    ("Demographics & vitals", "人口学与生命体征", [
        ("age", "Age", "年龄", ""), ("sex", "Sex", "性别", ""),
        ("weight", "Weight", "体重", "kg"), ("height", "Height", "身高", "cm"),
        ("bmi", "BMI", "BMI", "kg/m²"), ("bsa", "BSA", "BSA", "m²"),
        ("sbp", "SBP", "收缩压", "mmHg"), ("dbp", "DBP", "舒张压", "mmHg"),
        ("smoking", "Smoking", "吸烟", ""),
    ]),
    ("Comorbidities", "合并症", [
        ("hypertension", "Hypertension", "高血压", ""),
        ("diabetes", "Diabetes", "糖尿病", ""), ("cad", "CAD", "冠心病", ""),
        ("af", "Atrial fibrillation", "房颤", ""), ("ckd", "CKD", "慢性肾病", ""),
        ("tyg_high", "TyG high", "TyG 升高", ""),
    ]),
    ("Medications", "用药", [
        ("SGLT-2i", "SGLT-2i", "SGLT-2i", ""), ("insulin", "Insulin", "胰岛素", ""),
        ("bate_blockers", "Beta-blocker", "β受体阻滞剂", ""),
        ("acri_arb", "ACEI/ARB", "ACEI/ARB", ""), ("ccb", "CCB", "CCB", ""),
        ("aspirin", "Aspirin", "阿司匹林", ""), ("diuretics", "Diuretics", "利尿剂", ""),
        ("statin", "Statin", "他汀", ""),
    ]),
    ("Labs & metabolic", "实验室与代谢", [
        ("nyha_class", "NYHA class", "NYHA 分级", ""),
        ("egfr", "eGFR", "eGFR", "mL/min/1.73m²"), ("scr", "Creatinine", "血肌酐", "µmol/L"),
        ("KDIGO", "KDIGO", "KDIGO", ""), ("kdigo", "KDIGO stage", "KDIGO 分期", ""),
        ("G3a：eGFR 45–59", "G3a (eGFR 45–59)", "G3a (eGFR 45–59)", ""),
        ("G3b：eGFR 30–44", "G3b (eGFR 30–44)", "G3b (eGFR 30–44)", ""),
        ("G4–5：eGFR <30", "G4–5 (eGFR <30)", "G4–5 (eGFR <30)", ""),
        ("ntprobnp", "NT-proBNP", "NT-proBNP", "pg/mL"),
        ("logntprobnp", "log(NT-proBNP)", "log(NT-proBNP)", ""),
        ("hdl_c", "HDL-C", "HDL-C", "mmol/L"), ("ldl_c", "LDL-C", "LDL-C", "mmol/L"),
        ("tc", "Total cholesterol", "总胆固醇", "mmol/L"), ("tg", "Triglycerides", "甘油三酯", "mmol/L"),
        ("fbg", "Fasting glucose", "空腹血糖", "mmol/L"),
        ("tyg_index", "TyG index", "TyG 指数", ""), ("tyg_tertile", "TyG tertile", "TyG 三分位", ""),
    ]),
    ("Echo — LA/LV structure", "超声—左房左室结构", [
        ("lav", "LAV", "左房容积", "mL"), ("lavi", "LAVI", "左房容积指数", "mL/m²"),
        ("lvedd", "LVEDD", "左室舒末内径", "mm"), ("ivsd", "IVSd", "室间隔厚度", "mm"),
        ("lvpwd", "LVPWd", "左室后壁厚度", "mm"), ("lvmi", "LVMI", "左室质量指数", "g/m²"),
    ]),
    ("Diastolic function", "舒张功能", [
        ("mitral_e", "Mitral E", "二尖瓣 E", "cm/s"), ("mitral_a", "Mitral A", "二尖瓣 A", "cm/s"),
        ("mitral_ea", "E/A", "E/A", ""), ("septal_e", "Septal e'", "间隔 e'", "cm/s"),
        ("mitral_ee", "E/e'", "E/e'", ""), ("tri_e", "Tricuspid E", "三尖瓣 E", ""),
        ("tri_a", "Tricuspid A", "三尖瓣 A", ""), ("tri_ea", "Tricuspid E/A", "三尖瓣 E/A", ""),
    ]),
    ("LV systolic & pulmonary", "左室收缩与肺循环", [
        ("lvef", "LVEF", "左室射血分数", "%"), ("lvgls", "LV GLS", "左室整体长轴应变", "%"),
        ("tr", "TR grade", "三尖瓣反流", ""), ("pasp", "PASP", "肺动脉收缩压", "mmHg"),
        ("tapse", "TAPSE", "TAPSE", "mm"), ("tapse/pasp", "TAPSE/PASP", "TAPSE/PASP", ""),
        ("refwls/pasp", "RVFWLS/PASP", "RVFWLS/PASP", ""),
    ]),
    ("RV function & strain", "右室功能与应变", [
        ("rvs", "RV S'", "右室 S'", "cm/s"), ("rvfwls", "RV FWLS", "右室游离壁应变", "%"),
        ("BasalRVLS", "Basal RVLS", "基底段 RVLS", "%"), ("MedialRVLS", "Mid RVLS", "中间段 RVLS", "%"),
        ("ApicalRVLS", "Apical RVLS", "心尖段 RVLS", "%"),
        ("rvedv", "RVEDV", "右室舒末容积", "mL"), ("rvedvi", "RVEDVi", "右室舒末容积指数", "mL/m²"),
        ("rvesv", "RVESV", "右室收末容积", "mL"), ("rvesvi", "RVESVi", "右室收末容积指数", "mL/m²"),
        ("rvef", "RVEF", "右室射血分数", "%"),
    ]),
    ("RA function", "右房功能", [
        ("raa", "RA area", "右房面积", "cm²"), ("rav", "RA volume", "右房容积", "mL"),
        ("ravi", "RAVi", "右房容积指数", "mL/m²"), ("ra_myopathy", "RA myopathy", "右房心肌病", ""),
        ("ra_reservoir", "RA reservoir strain", "右房储器应变", "%"),
        ("ra_reservoir_tertile", "Reservoir tertile", "储器三分位", ""),
        ("ra_conduit", "RA conduit strain", "右房管道应变", "%"),
        ("ra_booster", "RA booster strain", "右房助力应变", "%"),
    ]),
]

# Derived (model/pipeline) columns that prefix the master Excel.
DERIVED_XLSX_COLUMNS = [
    ("patient_id",                  "Patient ID",           "患者ID"),
    ("original.integrated_tier",    "Integrated tier",      "综合风险分层"),
    ("original.cox_2yr_event_prob", "Cox 18-mo event prob", "Cox 18个月事件概率"),
    ("original.cox_risk_label",     "Cox risk label",       "Cox 风险标签"),
    ("original.primary_action",     "Primary action",       "主要建议"),
    ("original.actions",            "All actions",          "所有建议"),
    ("original.risk_drivers",       "Risk drivers",         "风险驱动因素"),
]

# Lazily-loaded {patient_id -> {column: value}} map of the full bio dataset.
_FULL_BIO_CACHE = None


def load_full_bio_records() -> dict:
    """Load basic_data.xlsx and key every patient by the SAME disambiguated id
    Agent A uses (`name`, with ' (k)' suffixes for repeats). Returns {} on any
    failure so the build degrades gracefully."""
    global _FULL_BIO_CACHE
    if _FULL_BIO_CACHE is not None:
        return _FULL_BIO_CACHE
    try:
        import math
        import pandas as pd
        df = pd.read_excel(BASIC_DATA_PATH, sheet_name=BASIC_DATA_SHEET)
        df = df.dropna(subset=["name"]).reset_index(drop=True)
        seen, pids = {}, []
        for nm in df["name"].astype(str):
            seen[nm] = seen.get(nm, 0) + 1
            pids.append(nm if seen[nm] == 1 else f"{nm} ({seen[nm]})")
        recs = {}
        for pid, (_, r) in zip(pids, df.iterrows()):
            d = {}
            for col in df.columns:
                v = r[col]
                if isinstance(v, float) and math.isnan(v):
                    v = None
                elif hasattr(v, "item"):
                    v = v.item()
                d[col] = v
            recs[pid] = d
        _FULL_BIO_CACHE = recs
        print(f"[ratings] Loaded full bio data: {len(recs)} patients × "
              f"{len(df.columns)} columns from {os.path.basename(BASIC_DATA_PATH)}")
    except Exception as e:
        print(f"  ⚠ Full bio data unavailable ({e}); falling back to model-input fields only.")
        _FULL_BIO_CACHE = {}
    return _FULL_BIO_CACHE


def _bio_disp(col: str, val, is_zh: bool):
    """Format a raw bio value for display (Yes/No for binaries, light rounding)."""
    if val is None or val == "":
        return "—"
    if isinstance(val, float):
        import math
        if math.isnan(val):
            return "—"
    if col in BINARY_COLS:
        try:
            iv = int(val)
            return ("是" if iv else "否") if is_zh else ("Yes" if iv else "No")
        except (TypeError, ValueError):
            pass
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        f = float(val)
        return int(f) if f.is_integer() else round(f, 2)
    return val


def _short_pid(real_id: str, idx: int) -> str:
    """Patient identifier shown to the doctor — index-based to keep them anonymous."""
    return f"P-{idx + 1:02d}"


def vendor_of(model: str) -> str:
    return VENDOR_DISPLAY.get(model.split("/")[0], model.split("/")[0])


# ─── HTML template ─────────────────────────────────────────────────────────────
def _safe_str(s) -> str:
    if s is None:
        return ""
    return str(s)


def _model_card_html(letter: str, pid: str, reasoning: dict, safety: dict, lang: str) -> str:
    """Render one anonymized model card with the LLM output and rating widgets.

    `pid` is the patient's display id (e.g. "P-03"); it is baked into every
    radio-group `name` so groups are unique PER (patient, model). Without this,
    radios for the same letter on different patients share one document-wide
    group and selecting a later patient silently clears the earlier ones.
    """
    is_zh = lang == "zh"

    def label(en, zh):
        return zh if is_zh else en

    mech = html.escape(_safe_str(reasoning.get("mechanism", "")))
    prio = html.escape(_safe_str(reasoning.get("clinical_priority", "")))
    note = html.escape(_safe_str(reasoning.get("patient_note", "")))
    caveat = html.escape(_safe_str(reasoning.get("model_caveat", "")))
    wps = reasoning.get("watchpoints", []) or []
    wp_html = "<ul>" + "".join(f"<li>{html.escape(_safe_str(w))}</li>" for w in wps) + "</ul>" if wps else f"<i>—</i>"
    s_summary = html.escape(_safe_str(safety.get("summary", "")))
    s_flags = safety.get("flags", []) or []
    if s_flags:
        flag_html = ""
        for fl in s_flags:
            if isinstance(fl, dict):
                sev = html.escape(_safe_str(fl.get("severity", "?")))
                concern = html.escape(_safe_str(fl.get("concern", "")))
                code = html.escape(_safe_str(fl.get("action_code", "")))
                flag_html += f'<div class="flag"><b>[{sev.upper()}]</b> <code>{code}</code><br>{concern}</div>'
            else:
                flag_html += f'<div class="flag">{html.escape(_safe_str(fl))}</div>'
    else:
        flag_html = "<i>—</i>"

    # Rating widgets — radios from 0 to SCORE_MAX inclusive.
    widget_rows = []
    for key, en_lbl, zh_lbl, reverse in RUBRIC:
        lbl = zh_lbl if is_zh else en_lbl
        reverse_tag = (" 🔁" if reverse else "")
        radios = "".join(
            f'<label class="opt"><input type="radio" '
            f'name="r-{pid}-{{LETTER}}-{key}" data-dim="{key}" value="{v}">'
            f'<span>{v}</span></label>'
            for v in range(0, SCORE_MAX + 1)
        )
        widget_rows.append(
            f'<div class="rrow"><div class="rname">{html.escape(lbl)}{reverse_tag}</div>'
            f'<div class="ropts">{radios}</div></div>'
        )
    widgets = "".join(widget_rows).replace("{LETTER}", letter)
    comment_placeholder = ("可选 — 写下你的具体顾虑或亮点" if is_zh
                           else "Optional — comments, concerns, highlights")

    return f"""
<div class="mcard" data-letter="{letter}">
  <div class="mcard-head">
    <span class="letter">{letter}</span>
    <span class="mhint">{label('Anonymous model', '匿名模型')}</span>
  </div>
  <div class="mcard-body">
    <div class="lbl">{label('Mechanism', '机制')}</div><div class="txt">{mech or '<i>—</i>'}</div>
    <div class="lbl">{label('Clinical priority', '临床优先级')}</div><div class="txt">{prio or '<i>—</i>'}</div>
    <div class="lbl">{label('Watchpoints', '监测点')}</div><div class="txt">{wp_html}</div>
    <div class="lbl">{label('Patient note', '患者解释')}</div><div class="txt note">{note or '<i>—</i>'}</div>
    <div class="lbl">{label('Model caveat', '模型局限')}</div><div class="txt caveat">{caveat or '<i>—</i>'}</div>
    <div class="lbl safety">{label('Safety flags', '安全核查')}</div><div class="txt">{flag_html}</div>
    <div class="lbl safety">{label('Safety summary', '安全总结')}</div><div class="txt">{s_summary or '<i>—</i>'}</div>
  </div>
  <div class="rating">
    <div class="rhead">{label('Rate this output (0–2 on each dimension)', '请对该输出评分（每维 0–2）')}</div>
    <div class="legend">{label(
        '0 = poor · 1 = adequate · 2 = excellent (for hallucination 🔁: 0 = none, 2 = many)',
        '0 = 错误或无关  ·  1 = 部分正确  ·  2 = 准确完整   （幻觉 🔁：0 = 无，2 = 多次）'
    )}</div>
    {widgets}
    <div class="rrow">
      <div class="rname">{label('Comments', '点评')}</div>
      <div class="ropts"><textarea name="c-{pid}-{letter}" data-letter="{letter}" rows="2"
                              placeholder="{html.escape(comment_placeholder)}"></textarea></div>
    </div>
  </div>
</div>
"""


def _fmt_num(v, dec: int = 1):
    """Round floats lightly for display; pass other values through unchanged."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        f = float(v)
        return int(f) if f.is_integer() else round(f, dec)
    return v


def _yesno(v, is_zh: bool) -> str:
    """0/1 (or bool) → localized Yes/No; fall back to the raw string."""
    try:
        iv = int(v)
    except (TypeError, ValueError):
        return _safe_str(v) or "—"
    if is_zh:
        return "是" if iv else "否"
    return "Yes" if iv else "No"


def _clinical_panel(sb: dict, orig: dict, lang: str) -> str:
    """Structured clinical-data panel shown to the doctor for each patient.

    Blinding-safe: shows clinical variables only — no real patient ID/name.
    """
    is_zh = lang == "zh"

    def L(en, zh):
        return zh if is_zh else en

    sex_raw = str(sb.get("sex") or "")
    sex = ("男" if sex_raw.lower().startswith("m") else "女") if is_zh else (sex_raw or "?")

    egfr = _fmt_num(sb.get("egfr"))
    ntp = _fmt_num(sb.get("ntprobnp"))
    ra = _fmt_num(sb.get("ra_reservoir"))

    a_tier = sb.get("agent_a_tier", "?")
    a_tier_disp = (TIER_ZH.get(a_tier, a_tier) if is_zh else a_tier)
    cox_label = orig.get("cox_risk_label", "?")
    cox_label_disp = (TIER_ZH.get(cox_label, cox_label) if is_zh else cox_label)
    prob = orig.get("cox_2yr_event_prob")
    prob_s = f"{prob:.1%}" if isinstance(prob, (int, float)) else "N/A"

    def unit(v, u):
        return f"{v} {u}" if v not in ("", None) else "—"

    items = [
        (L("Age", "年龄"),                    sb.get("age", "?")),
        (L("Sex", "性别"),                    sex),
        (L("NYHA class", "NYHA 分级"),         sb.get("nyha", "?")),
        (L("eGFR", "eGFR"),                   unit(egfr, "mL/min/1.73m²")),
        (L("NT-proBNP", "NT-proBNP"),         unit(ntp, "pg/mL")),
        (L("RA reservoir strain", "右房储器应变"), unit(ra, "%")),
        (L("Atrial fibrillation", "房颤"),     _yesno(sb.get("af"), is_zh)),
        (L("CKD", "慢性肾病"),                 _yesno(sb.get("ckd"), is_zh)),
        (L("Agent-A tier", "Agent A 分层"),    a_tier_disp),
        (L("Cox risk label", "Cox 风险标签"),  cox_label_disp),
        (L("Cox 18-mo event prob", "Cox 18个月事件概率"), prob_s),
    ]
    chips = "".join(
        f'<div class="citem"><span class="ck">{html.escape(_safe_str(k))}</span>'
        f'<span class="cv">{html.escape(_safe_str(v))}</span></div>'
        for k, v in items
    )

    drivers = orig.get("risk_drivers") or []
    if drivers:
        dr_html = ("<ul class='drv'>"
                   + "".join(f"<li>{html.escape(_safe_str(d))}</li>" for d in drivers)
                   + "</ul>")
    else:
        dr_html = "<i>—</i>"

    return (
        f'<div class="clin">'
        f'<div class="clin-title">{html.escape(L("Clinical data", "临床数据"))}</div>'
        f'<div class="cgrid">{chips}</div>'
        f'<div class="clin-sub">{html.escape(L("Risk drivers", "风险驱动因素"))}</div>'
        f'{dr_html}'
        f'</div>'
    )


def _full_bio_details(full_row: dict, lang: str) -> str:
    """Collapsible panel with ALL baseline bio variables, grouped (no name/outcomes)."""
    if not full_row:
        return ""
    is_zh = lang == "zh"
    groups_html = []
    for en_title, zh_title, fields in FULL_BIO_GROUPS:
        chips = []
        for col, en_lbl, zh_lbl, unit in fields:
            if col not in full_row:
                continue
            disp = _bio_disp(col, full_row.get(col), is_zh)
            if disp != "—" and unit:
                disp = f"{disp} {unit}"
            lbl = zh_lbl if is_zh else en_lbl
            chips.append(
                f'<div class="citem"><span class="ck">{html.escape(_safe_str(lbl))}</span>'
                f'<span class="cv">{html.escape(_safe_str(disp))}</span></div>'
            )
        if chips:
            groups_html.append(
                f'<div class="bgroup"><div class="bgroup-t">'
                f'{html.escape(zh_title if is_zh else en_title)}</div>'
                f'<div class="cgrid">{"".join(chips)}</div></div>'
            )
    summary = ("完整临床数据（全部基线指标）— 点击展开"
               if is_zh else "Full clinical data (all baseline variables) — click to expand")
    return (f'<details class="fullbio"><summary>{html.escape(summary)}</summary>'
            f'{"".join(groups_html)}</details>')


def _patient_block(doctor_idx: int, patient_idx: int, rec: dict,
                   letter_map: dict[str, str], lang: str,
                   full_recs: dict | None = None) -> str:
    """One patient section: header + collapsed original report + N model cards."""
    is_zh = lang == "zh"
    sb = rec["snapshot"]
    orig = rec["original"]
    short_pid = _short_pid(rec["patient_id"], patient_idx)
    tier = orig.get("integrated_tier", "Unknown")
    fg, bg = TIER_COLOR.get(tier, TIER_COLOR["Unknown"])
    tier_label = (TIER_ZH.get(tier, tier) if is_zh else tier)

    prob = orig.get("cox_2yr_event_prob")
    prob_s = f"{prob:.1%}" if isinstance(prob, (int, float)) else "N/A"

    sex_en = sb.get("sex") or "?"
    sex_zh = "男" if str(sex_en).lower().startswith("m") else "女"
    age = sb.get("age") or "?"
    nyha = sb.get("nyha") or "?"
    demo = (f"{age} 岁 · {sex_zh} · NYHA {nyha} 级" if is_zh
            else f"{age}y · {sex_en} · NYHA {nyha}")

    # Cards in the order dictated by the per-patient letter map
    # letter_map: real_model_string -> letter ("A".."F")  (one entry per model)
    # We need to display them in alphabetical letter order (A..F) so each card
    # gets a stable letter regardless of original order.
    inv = {letter: m for m, letter in letter_map.items()}
    sorted_letters = sorted(inv.keys())
    model_lookup = {m["model"]: m for m in rec["models"]}

    cards = []
    for letter in sorted_letters:
        real_model = inv[letter]
        m = model_lookup.get(real_model, {})
        cards.append(_model_card_html(letter, short_pid, m.get("reasoning", {}),
                                       m.get("safety", {}), lang))

    orig_box = (
        '<details class="orig" open><summary>'
        + (f'患者基础报告（无 LLM）— 点击折叠'
           if is_zh else 'Patient base report (no LLM) — click to collapse')
        + f'</summary><pre>{html.escape(_safe_str(orig.get("summary_text", "")))}</pre></details>'
    )

    clin = _clinical_panel(sb, orig, lang)
    full_row = (full_recs or {}).get(rec["patient_id"], {})
    fullbio = _full_bio_details(full_row, lang)

    return f"""
<section class="pblock" id="p{patient_idx}" data-pid="{html.escape(short_pid)}">
  <div class="phead">
    <div><span class="pid">{html.escape(short_pid)}</span>
         <span class="demo">{html.escape(demo)}</span></div>
    <div><span class="tierpill" style="color:{fg};background:{bg}">{html.escape(tier_label)}</span>
         <span class="cox">{('Cox 18 个月事件概率' if is_zh else 'Cox 18-mo')}: <b>{prob_s}</b></span></div>
  </div>
  {clin}
  {fullbio}
  {orig_box}
  <div class="cards">{''.join(cards)}</div>
</section>
"""


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CARE-Agent · {DOCTOR} · Rating Sheet</title>
<style>
  :root {
    --bg:#fbfaf7; --fg:#1f2330; --muted:#5a6072; --line:#e3e0d8;
    --accent:#7a3e1d; --accent-soft:#f0e6db; --code-bg:#f4f1ea;
    --font-zh:"PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif;
    --font-en:"Charter","Source Serif Pro",Georgia,serif;
    --font-mono:"JetBrains Mono","SF Mono",ui-monospace,monospace;
  }
  *{box-sizing:border-box}
  html,body{margin:0;background:var(--bg);color:var(--fg)}
  body{font-family:var(--font-en);line-height:1.55;font-size:14.5px}
  body.lang-zh{font-family:var(--font-zh);line-height:1.7}
  .wrap{max-width:1100px;margin:0 auto;padding:24px 22px 100px}
  header.bar{
    position:sticky;top:0;z-index:20;background:rgba(251,250,247,.97);
    backdrop-filter:saturate(150%) blur(8px);
    border-bottom:1px solid var(--line);
    display:flex;align-items:center;justify-content:space-between;
    padding:12px 22px;flex-wrap:wrap;gap:10px;
  }
  .barL{display:flex;gap:14px;align-items:baseline;flex-wrap:wrap}
  .barL .title{font-weight:700}
  .barL .who{color:var(--muted)}
  .barR{display:flex;gap:8px;align-items:center}
  .lang-toggle{display:inline-flex;border:1px solid var(--line);border-radius:999px;background:#fff;overflow:hidden}
  .lang-toggle button{border:0;background:transparent;padding:5px 14px;cursor:pointer;font:inherit;font-size:12.5px;color:var(--muted)}
  .lang-toggle button.active{background:var(--accent);color:#fff}
  .barR .btn{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:7px 14px;font:inherit;font-size:12.5px;cursor:pointer}
  .barR .btn.ghost{background:#fff;color:var(--accent);border:1px solid var(--accent)}
  .progress{font-size:12px;color:var(--muted);font-family:var(--font-mono)}
  .pblock{background:#fff;border:1px solid var(--line);border-radius:12px;margin:18px 0;overflow:hidden;box-shadow:0 2px 8px rgba(15,23,42,.04);scroll-margin-top:80px}
  .phead{padding:12px 16px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;border-bottom:1px solid var(--line)}
  .pid{font-size:20px;font-weight:700;font-family:var(--font-mono)}
  .demo{color:var(--muted);font-size:13px;margin-left:10px}
  .tierpill{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:700;letter-spacing:.03em}
  .cox{margin-left:10px;color:var(--muted);font-family:var(--font-mono);font-size:12.5px}
  .cox b{color:var(--fg)}
  details.orig{margin:6px 16px 0}
  details.orig summary{cursor:pointer;font-size:12.5px;color:var(--muted);font-weight:600;padding:6px 0}
  details.orig pre{background:#0f172a;color:#e2e8f0;padding:12px;border-radius:8px;white-space:pre-wrap;font-family:var(--font-mono);font-size:11.5px;line-height:1.55;max-height:280px;overflow:auto}
  .clin{margin:8px 16px 0;border:1px solid var(--line);border-radius:10px;background:#fcfbf8;padding:10px 12px}
  .clin-title{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--accent);font-weight:700;margin-bottom:6px}
  .cgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:2px 16px}
  .citem{display:flex;justify-content:space-between;gap:8px;border-bottom:1px dashed #ece7da;padding:4px 0;font-size:12.5px}
  .citem .ck{color:var(--muted)}
  .citem .cv{font-weight:600;font-family:var(--font-mono);font-size:12px;text-align:right;color:#1e293b}
  .clin-sub{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;color:#8a8a85;font-weight:700;margin:9px 0 2px}
  ul.drv{margin:2px 0 0;padding-left:18px;font-size:12.5px;color:#1e293b;line-height:1.5}
  ul.drv li{margin:1px 0}
  details.fullbio{margin:8px 16px 0;border:1px solid var(--line);border-radius:10px;background:#fcfbf8;overflow:hidden}
  details.fullbio>summary{cursor:pointer;font-size:12px;color:var(--accent);font-weight:700;padding:9px 12px;list-style:none}
  details.fullbio>summary::-webkit-details-marker{display:none}
  details.fullbio>summary::before{content:"▸ ";color:var(--accent)}
  details.fullbio[open]>summary::before{content:"▾ "}
  details.fullbio .bgroup{padding:4px 12px 8px;border-top:1px solid var(--line)}
  .bgroup-t{font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;color:#8a6a4a;font-weight:700;margin:6px 0 3px}
  .cards{padding:6px 14px 16px;display:grid;gap:12px}
  .mcard{border:1px solid var(--line);border-radius:10px;background:#fafaf6;overflow:hidden}
  .mcard-head{padding:8px 12px;background:#222;color:#fff;display:flex;align-items:baseline;gap:10px}
  .mcard-head .letter{font-weight:800;font-size:18px;font-family:var(--font-mono);letter-spacing:.05em}
  .mcard-head .mhint{font-size:11.5px;color:#cbd5e1}
  .mcard-body{padding:8px 12px 4px;font-size:13px}
  .lbl{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;color:#8a8a85;font-weight:700;margin:8px 0 2px}
  .lbl.safety{color:var(--accent)}
  .txt{line-height:1.55;color:#1e293b}
  .txt.note{color:#5b21b6}
  .txt.caveat{color:#64748b;font-size:12px}
  .txt i{color:#cbd5e1}
  .txt ul{margin:2px 0 0;padding-left:18px}
  .flag{background:#fff8e8;border-left:3px solid #b15a14;padding:5px 9px;margin:3px 0;border-radius:0 6px 6px 0;font-size:12.5px}
  .rating{border-top:1px solid var(--line);padding:10px 12px;background:#fff}
  .rhead{font-size:12.5px;color:var(--accent);font-weight:700;margin-bottom:4px}
  .legend{font-size:11.5px;color:var(--muted);margin-bottom:8px;line-height:1.5}
  .rrow{display:grid;grid-template-columns:230px 1fr;gap:10px;align-items:center;padding:4px 0;border-top:1px dashed #ece7da}
  .rrow:first-of-type{border-top:0}
  .rname{font-size:13px;color:#1e293b}
  .ropts{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
  .opt{display:inline-flex;align-items:center;gap:3px;padding:4px 9px;background:#fafaf6;border:1px solid var(--line);border-radius:8px;cursor:pointer;font-size:12.5px}
  .opt input{margin:0}
  .opt:has(input:checked){background:var(--accent-soft);border-color:var(--accent);color:var(--accent);font-weight:700}
  textarea{width:100%;font:inherit;font-size:13px;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:#fafaf6;resize:vertical}
  .toast{position:fixed;bottom:18px;right:18px;background:#1b6a4a;color:#fff;padding:8px 14px;border-radius:8px;font-size:13px;opacity:0;transition:opacity .25s;pointer-events:none;z-index:30}
  .toast.show{opacity:1}
  /* language toggle */
  :not(body).lang-en, :not(body).lang-zh { display:none }
  body.lang-en .lang-en { display:block }
  body.lang-zh .lang-zh { display:block }
  body.lang-en span.lang-en, body.lang-en label.lang-en { display:inline-block }
  body.lang-zh span.lang-zh, body.lang-zh label.lang-zh { display:inline-block }
</style></head>
<body class="lang-zh" data-doctor="{DOCTOR}" data-sheet-id="{SHEET_ID}">

<header class="bar">
  <div class="barL">
    <span class="title">CARE-Agent ·
      <span class="lang-en">Rating Sheet</span><span class="lang-zh">评分表</span></span>
    <span class="who">
      <span class="lang-en">Doctor:</span><span class="lang-zh">医生：</span>
      <b>{DOCTOR}</b>
    </span>
    <span class="progress" id="progress">0 / {TOTAL} done</span>
  </div>
  <div class="barR">
    <div class="lang-toggle">
      <button id="btn-en" type="button">English</button>
      <button id="btn-zh" type="button">中文</button>
    </div>
    <button class="btn" id="btn-link" type="button" style="display:none"
            title="Save progress directly to a file on your computer (auto-saves)">
      <span class="lang-en">Link file</span><span class="lang-zh">连接文件</span>
    </button>
    <span class="progress" id="fs-status" style="display:none"></span>
    <button class="btn ghost" id="btn-save" type="button" title="Auto-saved, but you can force a save here">
      <span class="lang-en">Save</span><span class="lang-zh">保存</span>
    </button>
    <button class="btn ghost" id="btn-import" type="button"
            title="Resume from a previously exported JSON file">
      <span class="lang-en">Import / Resume</span><span class="lang-zh">导入 / 继续</span>
    </button>
    <input id="file-import" type="file" accept="application/json,.json" style="display:none">
    <button class="btn" id="btn-export" type="button">
      <span class="lang-en">Export JSON</span><span class="lang-zh">导出 JSON</span>
    </button>
    <button class="btn ghost" id="btn-reset" type="button"
            title="Clear all answers on this sheet"
            style="color:#b71d3b;border-color:#e0a9b3">
      <span class="lang-en">Reset</span><span class="lang-zh">清空</span>
    </button>
  </div>
</header>

<div class="wrap">
  <div id="local-warning" style="display:none"></div>
  <div class="lang-en" style="color:var(--muted);font-size:13.5px;margin:6px 0 14px">
    Rate each anonymous model output (A/B/C/…) on six 0–2 dimensions.
    Outputs are blinded; the letter for the same model differs between patients.
    Hallucination is reverse-scored (higher = more hallucinations).
    All progress auto-saves in your browser. Click <b>Export JSON</b> when done
    and send the file back.
  </div>
  <div class="lang-zh" style="color:var(--muted);font-size:13.5px;margin:6px 0 14px">
    请对每位匿名模型（A/B/C/…）的输出在六个维度上进行 0–2 评分。
    模型已盲化，相同字母在不同患者间对应不同模型，避免品牌偏倚。
    "幻觉/错误推断" 为反向计分（分越高代表幻觉越多）。
    进度会自动保存在浏览器中。完成后点击 <b>导出 JSON</b> 并将文件发回。
  </div>
  {BODY}
</div>

<div class="toast" id="toast"></div>

<script>
(function(){
  var DOCTOR  = document.body.dataset.doctor;
  var SHEET   = document.body.dataset.sheetId;
  var KEY_LANG = 'careAgentRatingLang';
  var KEY_DATA = 'careAgentRating::' + SHEET + '::' + DOCTOR;
  var TOTAL   = parseInt(document.getElementById('progress').dataset.total || {TOTAL}, 10) || {TOTAL};

  // language toggle
  function applyLang(l){
    if(l!=='en') l='zh';
    document.body.classList.toggle('lang-en', l==='en');
    document.body.classList.toggle('lang-zh', l==='zh');
    document.getElementById('btn-en').classList.toggle('active', l==='en');
    document.getElementById('btn-zh').classList.toggle('active', l==='zh');
    document.documentElement.setAttribute('lang', l==='zh'?'zh-CN':'en');
    try{ localStorage.setItem(KEY_LANG, l); }catch(e){}
  }
  var savedLang = null; try{ savedLang = localStorage.getItem(KEY_LANG); }catch(e){}
  applyLang(savedLang || 'zh');
  document.getElementById('btn-en').onclick = function(){ applyLang('en'); };
  document.getElementById('btn-zh').onclick = function(){ applyLang('zh'); };

  // gather state
  function gather(){
    var out = {doctor: DOCTOR, sheet_id: SHEET,
               saved_at: new Date().toISOString(),
               ratings: {}};
    document.querySelectorAll('section.pblock').forEach(function(sec){
      var pid = sec.dataset.pid;
      out.ratings[pid] = {};
      sec.querySelectorAll('.mcard').forEach(function(card){
        var L = card.dataset.letter;
        var entry = {};
        card.querySelectorAll('input[type=radio]:checked').forEach(function(r){
          // dimension comes from data-dim (name is now patient-scoped: r-<pid>-<L>-<dim>)
          var dim = r.dataset.dim;
          entry[dim] = parseInt(r.value, 10);
        });
        var c = card.querySelector('textarea[data-letter="'+L+'"]');
        if(c && c.value.trim()) entry.comment = c.value.trim();
        out.ratings[pid][L] = entry;
      });
    });
    return out;
  }

  function progress(){
    var done = 0;
    document.querySelectorAll('.mcard').forEach(function(card){
      // a card is "done" once all 6 dims have a value
      var dims = new Set();
      card.querySelectorAll('input[type=radio]:checked').forEach(function(r){
        dims.add(r.name.split('-').slice(2).join('-'));
      });
      if(dims.size >= 6) done++;
    });
    var el = document.getElementById('progress');
    var txtZh = done + ' / ' + TOTAL + ' 已完成';
    var txtEn = done + ' / ' + TOTAL + ' done';
    el.textContent = document.body.classList.contains('lang-zh') ? txtZh : txtEn;
    el.dataset.total = TOTAL;
  }

  var fsWrite = null;     // assigned by the File System Access block below
  var serverSave = null;  // assigned by the local-server block below
  function save(){
    try{ localStorage.setItem(KEY_DATA, JSON.stringify(gather())); }catch(e){}
    if(fsWrite) fsWrite();         // auto-save to the linked on-disk file
    if(serverSave) serverSave();   // auto-save into the folder via local server
    progress();
  }

  // Apply a parsed ratings object to the form. Returns the number of
  // (patient, model) cells it managed to fill. Used by both localStorage
  // restore AND the Import/Resume button, so progress is portable across
  // machines, browsers and file paths (localStorage is none of those).
  function applyData(data){
    var filled = 0;
    Object.keys((data && data.ratings) || {}).forEach(function(pid){
      var entries = data.ratings[pid];
      var sec = document.querySelector('section.pblock[data-pid="'+pid+'"]');
      if(!sec) return;
      Object.keys(entries).forEach(function(letter){
        var entry = entries[letter];
        var card = sec.querySelector('.mcard[data-letter="'+letter+'"]');
        if(!card) return;
        var any = false;
        Object.keys(entry).forEach(function(k){
          if(k === 'comment'){
            var c = card.querySelector('textarea[data-letter="'+letter+'"]');
            if(c) { c.value = entry[k]; any = true; }
            return;
          }
          // card-scoped lookup by dimension — independent of the name format
          var r = card.querySelector('input[type=radio][data-dim="'+k+'"][value="'+entry[k]+'"]');
          if(r) { r.checked = true; any = true; }
        });
        if(any) filled++;
      });
    });
    return filled;
  }

  function restore(){
    var raw = null;
    try{ raw = localStorage.getItem(KEY_DATA); }catch(e){}
    if(!raw) { progress(); return; }
    try{ applyData(JSON.parse(raw)); }
    catch(e){ console.warn('restore failed', e); }
    progress();
  }

  // auto-save on any change
  document.addEventListener('change', save);
  document.addEventListener('input', function(e){
    if(e.target.tagName === 'TEXTAREA') save();
  });

  // manual save with toast
  document.getElementById('btn-save').onclick = function(){
    save();
    var t = document.getElementById('toast');
    t.textContent = document.body.classList.contains('lang-zh') ? '已保存' : 'Saved';
    t.classList.add('show'); setTimeout(function(){ t.classList.remove('show'); }, 1200);
  };

  // export JSON
  var exportedOnce = false;
  document.getElementById('btn-export').onclick = function(){
    save();
    var data = gather();
    var blob = new Blob([JSON.stringify(data, null, 2)], {type:'application/json;charset=utf-8'});
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'ratings_' + DOCTOR.replace(/\s+/g,'_') + '_' + SHEET + '.json';
    document.body.appendChild(a); a.click(); a.remove();
    exportedOnce = true;
  };

  // reset — clear every answer on this sheet (and persist the cleared state to
  // localStorage + linked file + server, so nothing lingers anywhere).
  document.getElementById('btn-reset').onclick = function(){
    var zh = document.body.classList.contains('lang-zh');
    if(!confirm(zh ? '确定清空本表的全部评分吗？此操作不可撤销。'
                   : 'Clear ALL answers on this sheet? This cannot be undone.')) return;
    document.querySelectorAll('.mcard input[type=radio]:checked').forEach(function(r){ r.checked = false; });
    document.querySelectorAll('.mcard textarea').forEach(function(t){ t.value = ''; });
    exportedOnce = true;        // nothing to nag about after a reset
    save();                     // writes the now-empty state everywhere
    var t = document.getElementById('toast');
    t.textContent = zh ? '已清空' : 'Cleared';
    t.classList.add('show'); setTimeout(function(){ t.classList.remove('show'); }, 1200);
  };

  // import / resume from a previously exported JSON — makes progress portable
  // and independent of which computer / browser / folder the sheet sits in.
  document.getElementById('btn-import').onclick = function(){
    document.getElementById('file-import').click();
  };
  document.getElementById('file-import').onchange = function(ev){
    var file = ev.target.files && ev.target.files[0];
    if(!file) return;
    var reader = new FileReader();
    reader.onload = function(){
      var zh = document.body.classList.contains('lang-zh');
      try{
        var data = JSON.parse(reader.result);
        if(data.doctor && DOCTOR && data.doctor !== DOCTOR){
          var msg = zh
            ? ('该文件属于医生「'+data.doctor+'」，与本表（'+DOCTOR+'）不一致。仍要导入吗？')
            : ('This file is for doctor "'+data.doctor+'", not this sheet ('+DOCTOR+'). Import anyway?');
          if(!confirm(msg)){ ev.target.value=''; return; }
        }
        var n = applyData(data);
        save();
        var t = document.getElementById('toast');
        t.textContent = zh ? ('已导入 '+n+' 个模型卡的进度') : ('Imported progress for '+n+' model cards');
        t.classList.add('show'); setTimeout(function(){ t.classList.remove('show'); }, 1800);
      }catch(e){
        alert(zh ? '导入失败：文件不是有效的评分 JSON。' : 'Import failed: not a valid ratings JSON file.');
      }
      ev.target.value = '';   // allow re-importing the same file
    };
    reader.readAsText(file);
  };

  // Warn before closing if there is work that has not been exported to a file.
  // localStorage alone is fragile (lost when the file is moved, re-unzipped,
  // or opened in another browser/computer), so we nudge the user to Export.
  window.addEventListener('beforeunload', function(e){
    var hasAny = document.querySelector('.mcard input[type=radio]:checked');
    // No need to nag if work is continuously saved (linked file or local server).
    if(hasAny && !exportedOnce && !(fileHandle && fsPerm) && !serverMode){
      e.preventDefault(); e.returnValue = '';
      return '';
    }
  });

  // ─── Robust storage: auto-save directly to a real file on disk ───────────
  // Uses the File System Access API. The rater clicks "Link file" once and
  // picks a .json; every change is then written straight to that file, so
  // progress survives cache clears, folder moves and browser restarts — and
  // the file itself is the deliverable to send back. Falls back silently to
  // Export/Import when the browser lacks the API (Safari/Firefox).
  var fileHandle = null, fsPerm = false, fsTimer = null;
  var FS_SUPPORTED = (typeof window.showSaveFilePicker === 'function');
  var linkBtn = document.getElementById('btn-link');
  var fsStatus = document.getElementById('fs-status');
  var FS_KEY = 'fh::' + SHEET + '::' + DOCTOR;

  function fsSetStatus(connected, name){
    if(!fsStatus) return;
    var zh = document.body.classList.contains('lang-zh');
    fsStatus.style.display = 'inline-block';
    fsStatus.textContent = connected ? ((zh?'已保存到文件: ':'Saving to: ') + (name||''))
                                     : (zh?'未连接文件':'No file linked');
    fsStatus.style.color = connected ? '#1b6a4a' : 'var(--muted)';
  }
  function idbOpen(){ return new Promise(function(res,rej){
    var r = indexedDB.open('careAgentFS', 1);
    r.onupgradeneeded = function(){ r.result.createObjectStore('handles'); };
    r.onsuccess = function(){ res(r.result); };
    r.onerror   = function(){ rej(r.error); };
  }); }
  function idbGet(k){ return idbOpen().then(function(db){ return new Promise(function(res,rej){
    var t = db.transaction('handles','readonly').objectStore('handles').get(k);
    t.onsuccess=function(){res(t.result);}; t.onerror=function(){rej(t.error);};
  }); }); }
  function idbSet(k,v){ return idbOpen().then(function(db){ return new Promise(function(res,rej){
    var t = db.transaction('handles','readwrite').objectStore('handles').put(v,k);
    t.onsuccess=function(){res();}; t.onerror=function(){rej(t.error);};
  }); }); }

  function fsVerifyPerm(handle){
    var opts = {mode:'readwrite'};
    return handle.queryPermission(opts).then(function(s){
      if(s==='granted') return true;
      return handle.requestPermission(opts).then(function(s2){ return s2==='granted'; });
    });
  }
  function fsReadInto(handle){
    return handle.getFile().then(function(f){ return f.text(); }).then(function(txt){
      if(txt && txt.trim()){
        var d = JSON.parse(txt);
        if(d && d.ratings) applyData(d);
      }
    }).catch(function(){});
  }
  function fsWriteNow(){
    if(!fileHandle || !fsPerm) return;
    fileHandle.createWritable().then(function(w){
      return w.write(new Blob([JSON.stringify(gather(), null, 2)],
                              {type:'application/json'})).then(function(){ return w.close(); });
    }).catch(function(e){ console.warn('file write failed', e); });
  }
  fsWrite = function(){ if(!fileHandle || !fsPerm) return;
                        clearTimeout(fsTimer); fsTimer = setTimeout(fsWriteNow, 400); };

  function fsLinkNew(){
    window.showSaveFilePicker({
      suggestedName: 'ratings_' + DOCTOR.replace(/\s+/g,'_') + '_' + SHEET + '.json',
      types: [{description:'JSON', accept:{'application/json':['.json']}}]
    }).then(function(h){
      fileHandle = h;
      return fsVerifyPerm(h).then(function(ok){
        fsPerm = ok;
        return fsReadInto(h);            // resume if the chosen file already had data
      }).then(function(){
        save(); fsSetStatus(true, h.name); progress();
        // Persist the handle for auto-reconnect next time — best-effort only,
        // so a failure here (e.g. private mode) never blocks saving.
        idbSet(FS_KEY, h).catch(function(){});
      });
    }).catch(function(){ /* user cancelled picker */ });
  }

  function fsReconnect(){
    if(!FS_SUPPORTED){ return; }
    linkBtn.style.display = 'inline-block';
    idbGet(FS_KEY).then(function(h){
      // Guard against a race: if the rater already linked a file manually while
      // this async lookup was in flight, don't clobber the connected status.
      if(!h){ if(!fileHandle) fsSetStatus(false); return; }
      if(fileHandle) return;            // already linked in this session
      fileHandle = h;
      return h.queryPermission({mode:'readwrite'}).then(function(s){
        if(s==='granted'){
          fsPerm = true;
          return fsReadInto(h).then(function(){ fsSetStatus(true, h.name); progress(); });
        } else {
          // Needs a user click to re-grant permission after reopening.
          fsSetStatus(false);
          var zh = document.body.classList.contains('lang-zh');
          linkBtn.querySelector('.lang-zh').textContent = '重新连接文件';
          linkBtn.querySelector('.lang-en').textContent = 'Reconnect file';
        }
      });
    }).catch(function(){ fsSetStatus(false); });
  }

  // ─── Local-server mode (best): served by the bundled launcher over http ───
  // When the sheet is opened through the "Start rating" launcher, it is served
  // over http://127.0.0.1 — so every change auto-saves straight into the folder
  // and reopening auto-loads it, with zero file pickers or clicks.
  var serverMode = (location.protocol === 'http:' || location.protocol === 'https:')
                   && (typeof window.fetch === 'function');
  var srvTimer = null, srvBusy = false, srvPending = false;
  function srvStatus(ok){
    if(!fsStatus) return;
    var zh = document.body.classList.contains('lang-zh');
    fsStatus.style.display = 'inline-block';
    fsStatus.textContent = ok ? (zh?'已自动保存到文件夹 ✓':'Auto-saved to folder ✓')
                              : (zh?'保存失败，请保持启动器窗口开着':'Save failed — keep the launcher window open');
    fsStatus.style.color = ok ? '#1b6a4a' : '#b71d3b';
  }
  function srvPost(){
    if(srvBusy){ srvPending = true; return; }
    srvBusy = true;
    fetch('save', {method:'POST', headers:{'Content-Type':'application/json'},
                   body: JSON.stringify(gather())})
      .then(function(r){ return r.json(); })
      .then(function(j){ srvBusy=false; srvStatus(j && j.ok);
                         if(srvPending){ srvPending=false; srvPost(); } })
      .catch(function(){ srvBusy=false; srvStatus(false); });
  }
  function srvLoad(){
    fetch('load?doctor=' + encodeURIComponent(DOCTOR) + '&sheet=' + encodeURIComponent(SHEET))
      .then(function(r){ return r.json(); })
      .then(function(j){ if(j && j.ok && j.data && j.data.ratings) applyData(j.data);
                         srvStatus(true); progress(); })
      .catch(function(){ srvStatus(false); });
  }
  serverSave = function(){ if(!serverMode) return;
                           clearTimeout(srvTimer); srvTimer = setTimeout(srvPost, 400); };

  if(serverMode){
    // Folder auto-save is active; the file-picker and the file:// warning are
    // unnecessary, and there is nothing unsaved to nag about on close.
    if(linkBtn) linkBtn.style.display = 'none';
  } else {
    if(FS_SUPPORTED){
      linkBtn.onclick = function(){
        // If a handle exists but lost permission (after reopen), re-grant & resume.
        if(fileHandle && !fsPerm){
          fsVerifyPerm(fileHandle).then(function(ok){
            fsPerm = ok;
            if(ok){ return fsReadInto(fileHandle).then(function(){
              fsSetStatus(true, fileHandle.name); save(); progress();
              linkBtn.querySelector('.lang-zh').textContent = '连接文件';
              linkBtn.querySelector('.lang-en').textContent = 'Link file';
            }); }
            else fsLinkNew();
          });
        } else {
          fsLinkNew();
        }
      };
      fsReconnect();
    }

    // If opened as a local file (file://), localStorage is tied to this exact
    // path + browser. Tell the rater plainly so they don't lose work.
    if(location.protocol === 'file:'){
      var w = document.getElementById('local-warning');
      if(w){
        w.style.cssText = 'background:#fff4e8;border:1px solid #e0b483;color:#7a3e1d;'
          + 'border-radius:8px;padding:10px 14px;margin:6px 0 14px;font-size:13px;line-height:1.6';
        w.innerHTML =
          '<span class="lang-zh"><b>⚠ 重要：</b>请先把压缩包<b>完整解压</b>到一个固定文件夹，'
          + '不要直接在压缩包里打开本文件。进度只保存在<b>当前这台电脑的这个浏览器</b>里——'
          + '换电脑、换浏览器或移动/重新解压文件后将无法自动恢复。'
          + '更稳妥：用随附的<b>启动器</b>打开（自动存到文件夹）；'
          + '或每做一部分就点右上角 <b>导出 JSON</b>，下次用 <b>导入 / 继续</b> 载入。</span>'
          + '<span class="lang-en"><b>⚠ Important:</b> First <b>fully extract</b> the zip to a fixed folder; '
          + 'do not open this file from inside the zip. Progress is saved only in <b>this browser on this computer</b>. '
          + 'Safer: open via the bundled <b>launcher</b> (auto-saves to the folder), '
          + 'or click <b>Export JSON</b> as you go and <b>Import / Resume</b> next time.</span>';
        w.style.display = 'block';
      }
    }
  }

  restore();
  if(serverMode) srvLoad();   // server copy is authoritative; overlay it last
})();
</script>
</body></html>
"""


def _dig(rec: dict, dotted: str):
    """Fetch a value from a patient record by 'snapshot.x' / 'original.x' / 'x'."""
    if "." in dotted:
        head, tail = dotted.split(".", 1)
        return (rec.get(head) or {}).get(tail)
    return rec.get(dotted)


def _xlsx_cell_value(dotted: str, raw):
    """Normalise a raw value for an Excel cell (join lists, keep numbers)."""
    if raw is None:
        return ""
    if isinstance(raw, (list, tuple)):
        # risk_drivers / actions: one item per line so the cell stays readable
        sep = "\n" if dotted.endswith("risk_drivers") else "; "
        return sep.join(str(x) for x in raw)
    return raw


def write_patient_excel(out_dir: str, llm_data: dict, lang: str,
                        full_recs: dict | None = None) -> str | None:
    """Write an un-blinded master Excel with every patient's COMPLETE information.

    One row per patient in the run: pipeline-derived fields (tier, Cox, actions,
    risk drivers) + ALL source bio columns (incl. follow-up OUTCOMES and the real
    name) + the base report text. PRIVATE — keep with the blinding/ folder.
    Returns the path (or None on failure).
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
    except Exception as e:  # openpyxl missing — don't crash the whole build
        print(f"  ⚠ patient_information.xlsx skipped (openpyxl unavailable: {e})")
        return None

    patients = list(llm_data.get("patients", []))
    if not patients:
        return None
    full_recs = full_recs or {}

    # Most-severe tier first, then highest Cox probability within a tier.
    def _sort_key(rec):
        tier = (rec.get("original") or {}).get("integrated_tier", "Unknown")
        prob = (rec.get("original") or {}).get("cox_2yr_event_prob")
        prob = prob if isinstance(prob, (int, float)) else -1
        return (TIER_RANK.get(tier, 99), -prob)
    patients = sorted(patients, key=_sort_key)

    is_zh = lang == "zh"

    # Source column order (full bio incl. name + outcomes) from the first patient
    # that has a matched record; raw dataset names are used as headers.
    src_cols = []
    for rec in patients:
        fr = full_recs.get(rec.get("patient_id"))
        if fr:
            src_cols = list(fr.keys())
            break

    derived_headers = [(zh if is_zh else en) for _, en, zh in DERIVED_XLSX_COLUMNS]
    report_header = "基础报告（无 LLM）" if is_zh else "Base report (no LLM)"
    headers = ["#"] + derived_headers + src_cols + [report_header]

    wb = Workbook()
    ws = wb.active
    ws.title = "患者信息" if is_zh else "Patients"

    head_fill = PatternFill("solid", fgColor="7A3E1D")
    head_font = Font(bold=True, color="FFFFFF", size=11)
    thin = Side(style="thin", color="D9D2C5")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    top_align = Alignment(vertical="top", wrap_text=True)

    for c, title in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=c, value=title)
        cell.fill = head_fill
        cell.font = head_font
        cell.alignment = Alignment(vertical="center", horizontal="center", wrap_text=True)
        cell.border = border

    tier_fill = {"LOW": "DFF0E6", "MODERATE": "FFF4CF", "HIGH": "FDE9D4", "CRITICAL": "FDE2E7"}
    # Integrated tier sits at: 1 (#) + its position within DERIVED_XLSX_COLUMNS.
    tier_col_idx = 1 + 1 + next(i for i, (jp, *_) in enumerate(DERIVED_XLSX_COLUMNS)
                                if jp == "original.integrated_tier")
    n_derived = len(DERIVED_XLSX_COLUMNS)

    for r, rec in enumerate(patients, start=2):
        ws.cell(row=r, column=1, value=r - 1).alignment = top_align
        ws.cell(row=r, column=1).border = border
        # derived block
        for c, (dotted, _en, _zh) in enumerate(DERIVED_XLSX_COLUMNS, start=2):
            val = _xlsx_cell_value(dotted, _dig(rec, dotted))
            cell = ws.cell(row=r, column=c, value=val)
            cell.alignment = top_align
            cell.border = border
            if dotted == "original.cox_2yr_event_prob" and isinstance(val, (int, float)):
                cell.number_format = "0.0%"
            if c == tier_col_idx:
                fill = tier_fill.get(str(val))
                if fill:
                    cell.fill = PatternFill("solid", fgColor=fill)
                    cell.font = Font(bold=True)
        # full bio block (raw source values)
        fr = full_recs.get(rec.get("patient_id"), {})
        for j, col in enumerate(src_cols):
            v = fr.get(col)
            if v is None:
                v = ""
            cell = ws.cell(row=r, column=2 + n_derived + j, value=v)
            cell.alignment = top_align
            cell.border = border
        # base report (last column)
        rep = _safe_str((rec.get("original") or {}).get("summary_text", ""))
        rep_cell = ws.cell(row=r, column=len(headers), value=rep)
        rep_cell.alignment = top_align
        rep_cell.border = border

    # Column widths: wide for free-text, narrow default for the many bio columns.
    wide = {"患者ID", "Patient ID", "name", "风险驱动因素", "Risk drivers",
            "所有建议", "All actions", "主要建议", "Primary action",
            report_header}
    for c, title in enumerate(headers, start=1):
        if title == "#":
            w = 5
        elif title in (report_header,):
            w = 70
        elif title in wide:
            w = 30
        else:
            w = 14
        ws.column_dimensions[get_column_letter(c)].width = w

    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{len(patients) + 1}"

    path = os.path.join(out_dir, "patient_information.xlsx")
    wb.save(path)
    n_cols = len(headers)
    note = "full bio + outcomes" if src_cols else "model-input fields only (source data not found)"
    print(f"  ✓ wrote {path}  ({len(patients)} patients × {n_cols} cols · {note} · "
          f"un-blinded — keep private)")
    return path


def build_one_doctor(doctor: str, doctor_idx: int, llm_data: dict, args, sheet_id: str,
                     out_dir: str, full_recs: dict | None = None) -> dict:
    """Build the HTML sheet + un-blinding map for one doctor. Returns the
    un-blinding map (so the caller can also write a combined summary)."""
    patients = llm_data.get("patients", [])
    models   = list((llm_data.get("meta") or {}).get("models")
                    or sorted({m["model"] for p in patients for m in p.get("models", [])}))

    # Per-doctor RNG (deterministic given --seed + doctor name)
    seed_str = f"{args.seed}::{doctor}::{sheet_id}"
    rng = random.Random(int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**31))

    # Letters A..Z up to the number of models
    letters = [chr(ord("A") + i) for i in range(len(models))]
    if len(models) > 26:
        raise ValueError("More than 26 models; rating sheet would run out of letters.")

    # Shuffle patient order for this doctor
    patient_order = list(range(len(patients)))
    rng.shuffle(patient_order)

    # Per-patient letter map: real_model -> letter (re-rolled per patient)
    blinding_per_patient = []  # list aligned to patient_order
    body_chunks = []
    for pidx_display, pidx_real in enumerate(patient_order):
        rec = patients[pidx_real]
        shuffled_letters = letters[:]
        rng.shuffle(shuffled_letters)
        letter_map = dict(zip(models, shuffled_letters))
        blinding_per_patient.append({
            "doctor": doctor,
            "display_pid":  _short_pid(rec["patient_id"], pidx_display),
            "real_pid":     rec["patient_id"],
            "integrated_tier": rec["original"]["integrated_tier"],
            "letter_to_model": {letter_map[m]: m for m in models},
        })
        body_chunks.append(_patient_block(doctor_idx, pidx_display, rec, letter_map,
                                          args.lang, full_recs))

    total_ratings = len(patients) * len(models)
    html_page = HTML_PAGE
    html_page = html_page.replace("{DOCTOR}", html.escape(doctor))
    html_page = html_page.replace("{SHEET_ID}", html.escape(sheet_id))
    html_page = html_page.replace("{TOTAL}", str(total_ratings))
    html_page = html_page.replace("{BODY}", "".join(body_chunks))

    # Write the HTML
    safe_doctor = re.sub(r"[^\w一-鿿]+", "_", doctor).strip("_") or "doctor"
    html_path = os.path.join(out_dir, f"rating_sheet_{safe_doctor}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_page)

    # Write the blinding map (private — needed for aggregation)
    bl_dir = os.path.join(out_dir, "blinding")
    os.makedirs(bl_dir, exist_ok=True)
    blinding_path = os.path.join(bl_dir, f"{safe_doctor}.json")
    with open(blinding_path, "w", encoding="utf-8") as f:
        json.dump({
            "doctor":   doctor,
            "sheet_id": sheet_id,
            "seed":     args.seed,
            "patient_display_order": [b["display_pid"] for b in blinding_per_patient],
            "per_patient": blinding_per_patient,
        }, f, ensure_ascii=False, indent=2)

    print(f"  ✓ {doctor:>10s}  →  {html_path}  ({total_ratings} ratings)")
    return {"html": html_path, "blinding": blinding_path}


def write_readme(out_dir: str, doctors: list[str], total: int, n_models: int, n_patients: int, lang: str):
    """A short instructions file for the doctors."""
    last_pid = f"P-{n_patients:02d}" if n_patients else "P-??"
    en = f"""# CARE-Agent — Rating Sheet (Instructions)

You'll receive a single HTML file: `rating_sheet_<your_name>.html`.

1. Double-click to open it in any modern browser (Chrome, Safari, Edge).
2. For each patient (P-01 to {last_pid}), you'll see **{n_models} anonymous model
   outputs** labelled **Model A / B / C / …**. The labels are re-randomised per
   patient so brand pattern recognition is impossible.
3. Read the patient's *base report* (click the "Show patient base report" disclosure
   if you need the clinical context), then for **each** model output rate the six
   dimensions on a 0–2 scale:
   - Mechanism accuracy
   - Risk-driver identification
   - Treatment-recommendation rationality
   - Safety-flag quality
   - Hallucination *(reverse-scored — higher = worse)*
   - Clinical readability
4. Add a comment if useful (optional).
5. **Saving & resuming (important).** First **fully extract** the zip to a fixed
   folder; do not open the HTML from inside the zip. Three ways, best first:
   - **Easiest (recommended, needs Python installed):** double-click the bundled
     launcher — **`Start_rating.command`** on macOS, **`Start_rating.bat`** on
     Windows. It opens your sheet and **auto-saves every answer into this
     folder** (`ratings_<your_name>_*.json`); next time just double-click the
     launcher again and it **auto-loads** — no file pickers, no buttons. Keep
     the launcher's black window **open** while rating; close it when done.
   - **Chrome / Edge by double-clicking the HTML:** click **Link file** (top
     right) once, choose a location, and answers auto-write to it. Next time
     click **Link file** again and pick the same file to continue.
   - **Fallback (Safari / Firefox):** progress stays only in this browser and
     won't restore elsewhere. Click **Export JSON** as you go and **Import /
     Resume** next time.
6. When finished, send back `ratings_<your_name>_*.json` from the folder (or the
   **Export JSON** download).

Thank you.
"""

    zh = f"""# CARE-Agent — 评分说明

你会收到一个 HTML 文件：`rating_sheet_<你的姓名>.html`。

1. 在浏览器中双击打开（Chrome、Safari、Edge 均可）。
2. 每位患者（P-01 至 {last_pid}）下方会展示 **{n_models} 个匿名模型**（标记为
   Model A / B / C / …），同一字母在不同患者间对应不同模型，避免品牌偏倚。
3. 如需临床背景，可点击"查看患者基础报告（无 LLM）"展开。对**每个**模型输出，按以下
   六个维度 0–2 评分：
   - 机制解释准确性
   - 风险驱动因素识别
   - 治疗建议合理性
   - 安全标记质量
   - 幻觉/错误推断（**反向计分** — 分越高代表幻觉越多）
   - 临床可读性
4. 可选填写"点评"。
5. **保存与续做（重要）。** 请先把压缩包**完整解压**到一个固定文件夹，不要直接在
   压缩包里打开 HTML。按推荐程度，有三种保存方式：
   - **最省心（推荐，需已装 Python）：** 双击随附的启动器——macOS 用
     **`Start_rating.command`**，Windows 用 **`Start_rating.bat`**。它会自动打开
     评分表，之后**每次打分都自动存进这个文件夹**（生成 `ratings_<姓名>_*.json`），
     下次再双击启动器即可**自动读回**，全程无需选文件、无需点任何按钮。评分时请
     **保持启动器的黑色窗口开着**，做完再关。
   - **Chrome / Edge 直接双击 HTML：** 打开后点右上角 **连接文件**，选一个保存位置，
     之后自动写入该文件；下次打开再点一次 **连接文件**、选同一个文件即可继续。
   - **兜底（Safari / Firefox）：** 进度暂存在当前浏览器里，换电脑/浏览器/移动文件后
     无法自动恢复。请每做一部分点 **导出 JSON**，下次用 **导入 / 继续** 载入。
6. 全部完成后，把文件夹里的 `ratings_<姓名>_*.json`（或 **导出 JSON** 下载的文件）回传。

谢谢。
"""

    path = os.path.join(out_dir, "README_for_doctors.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write((zh if lang == "zh" else en) + "\n\n---\n\n" + (en if lang == "zh" else zh))
    print(f"  ✓ wrote {path}")


# ── Local-server assets: let a doctor double-click a launcher so the sheet is
# served over http://127.0.0.1 and every change auto-saves into the folder. ──
SERVE_SCRIPT = r'''#!/usr/bin/env python3
"""CARE-Agent rating server (zero-dependency, local only).

Double-click the matching launcher (Start_rating.command on macOS,
Start_rating.bat on Windows) — it runs this and opens your rating sheet in the
browser. Every answer is then saved straight into THIS folder as
ratings_<doctor>_<sheet>.json, and reopening auto-loads it. Keep the launcher
window open while you rate; close it when you are done.
"""
import http.server, socketserver, json, os, re, sys, glob, threading, webbrowser
import urllib.parse

# When bundled into a standalone .exe by PyInstaller, __file__ points inside the
# temporary unpack dir; serve the folder the .exe actually lives in (next to the
# rating sheet) instead.
if getattr(sys, "frozen", False):
    HERE = os.path.dirname(os.path.abspath(sys.executable))
else:
    HERE = os.path.dirname(os.path.abspath(__file__))
START_PORT = int(os.environ.get("CARE_PORT", "8753"))


def safe(name):
    return re.sub(r"[^\w一-鿿]+", "_", str(name)).strip("_") or "doctor"


def ratings_path(doctor, sheet):
    return os.path.join(HERE, "ratings_%s_%s.json" % (safe(doctor), safe(sheet)))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=HERE, **k)

    def _send_json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/load":
            q = urllib.parse.parse_qs(u.query)
            fn = ratings_path((q.get("doctor") or [""])[0], (q.get("sheet") or [""])[0])
            if os.path.exists(fn):
                try:
                    with open(fn, encoding="utf-8") as f:
                        return self._send_json(200, {"ok": True, "data": json.load(f)})
                except Exception as e:
                    return self._send_json(200, {"ok": False, "error": str(e)})
            return self._send_json(200, {"ok": True, "data": None})
        # Security: only ever serve the rating sheets themselves. This keeps any
        # private files that might sit in the folder (blinding/, patient_*.xlsx)
        # from being exposed over the local server.
        name = os.path.basename(u.path)
        if u.path in ("/", "") or name == "":
            sheets = sorted(os.path.basename(p) for p in glob.glob(os.path.join(HERE, "rating_sheet_*.html")))
            if len(sheets) == 1:
                self.send_response(302); self.send_header("Location", "/" + sheets[0]); self.end_headers(); return
            if len(sheets) == 0:
                msg = (
                    "<!doctype html><meta charset=utf-8>"
                    "<div style='font-family:system-ui,sans-serif;max-width:720px;margin:40px auto;"
                    "line-height:1.7;color:#1f2330'>"
                    "<h2 style='color:#b15a14'>没找到评分表 / No rating sheet found</h2>"
                    "<p>请把 <b>Start_rating.exe</b> 和某位医生的 "
                    "<b>rating_sheet_X.html</b> 放在<b>同一个文件夹</b>里，"
                    "再双击 exe。</p>"
                    "<p>Put <b>Start_rating.exe</b> in the <b>same folder</b> as a doctor's "
                    "<b>rating_sheet_X.html</b>, then double-click the exe again.</p>"
                    "<p style='color:#5a6072;font-size:13px'>当前查找的文件夹 / "
                    "Looking in:<br><code>%s</code></p></div>"
                    % HERE.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                )
                b = msg.encode("utf-8")
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
            links = "".join('<li><a href="/%s">%s</a></li>' % (s, s) for s in sheets)
            html = "<!doctype html><meta charset=utf-8><h3>CARE-Agent rating sheets</h3><ul>%s</ul>" % links
            b = html.encode("utf-8")
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
        if re.fullmatch(r"rating_sheet_[\w一-鿿\-]+\.html", name):
            return super().do_GET()
        self.send_response(404); self.end_headers()

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/save":
            try:
                n = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(n) or b"{}")
                fn = ratings_path(data.get("doctor", "doctor"), data.get("sheet_id", "sheet"))
                tmp = fn + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp, fn)
                return self._send_json(200, {"ok": True, "file": os.path.basename(fn)})
            except Exception as e:
                return self._send_json(200, {"ok": False, "error": str(e)})
        self.send_response(404); self.end_headers()

    def log_message(self, *a):
        pass


def main():
    os.chdir(HERE)
    port = START_PORT
    httpd = None
    for _ in range(30):
        try:
            httpd = socketserver.TCPServer(("127.0.0.1", port), Handler); break
        except OSError:
            port += 1
    if httpd is None:
        print("Could not find a free port."); sys.exit(1)
    sheets = sorted(glob.glob(os.path.join(HERE, "rating_sheet_*.html")))
    url = "http://127.0.0.1:%d/" % port
    if len(sheets) == 1:
        url += os.path.basename(sheets[0])
    print("=" * 60)
    print("CARE-Agent rating server is running.")
    print("  Open:", url)
    print("  Your answers auto-save into this folder.")
    print("  KEEP THIS WINDOW OPEN while rating; close it when finished.")
    print("=" * 60)
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
'''

LAUNCH_COMMAND = (
    "#!/bin/bash\n"
    "# Double-click to start rating (macOS). Keep the window open while rating.\n"
    'cd "$(dirname "$0")"\n'
    "python3 serve_ratings.py || python serve_ratings.py\n"
)

PY_INSTALLER_VER = "3.12.7"   # official python.org build downloaded if Python is missing

LAUNCH_BAT = (
    "@echo off\r\n"
    "chcp 65001 >nul\r\n"
    "setlocal enableextensions\r\n"
    'cd /d "%~dp0"\r\n'
    "rem ---- 1) Already have a real Python? (-c test skips the MS Store stub) ----\r\n"
    'py -3 -c "import sys" >nul 2>nul && ( py -3 serve_ratings.py & goto end )\r\n'
    'python -c "import sys" >nul 2>nul && ( python serve_ratings.py & goto end )\r\n'
    'python3 -c "import sys" >nul 2>nul && ( python3 serve_ratings.py & goto end )\r\n'
    "\r\n"
    "echo ============================================================\r\n"
    "echo  No Python detected on this PC / 这台电脑没有检测到 Python\r\n"
    "echo  This tool can install it automatically for the current user\r\n"
    "echo  (no admin needed, needs internet).\r\n"
    "echo  可为你自动安装（仅当前用户、无需管理员、需要联网）。\r\n"
    "echo ============================================================\r\n"
    'choice /C YN /M "Install Python now / 现在自动安装吗 (Y=yes N=no)"\r\n'
    "if errorlevel 2 goto manual\r\n"
    "\r\n"
    "rem ---- 2) Download the official per-user installer to TEMP ----\r\n"
    'set "PYVER=' + PY_INSTALLER_VER + '"\r\n'
    'set "PYEXE=%TEMP%\\python-%PYVER%-amd64.exe"\r\n'
    "echo Downloading Python %PYVER% ... / 正在下载 ...\r\n"
    "powershell -NoProfile -ExecutionPolicy Bypass -Command "
    "\"[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; "
    "try{ Invoke-WebRequest -UseBasicParsing -Uri "
    "'https://www.python.org/ftp/python/%PYVER%/python-%PYVER%-amd64.exe' "
    "-OutFile '%PYEXE%' }catch{ exit 1 }\"\r\n"
    'if not exist "%PYEXE%" goto dlfail\r\n'
    "\r\n"
    "rem ---- 3) Silent per-user install + add to PATH ----\r\n"
    "echo Installing ... / 正在安装（请稍候，勿关闭窗口）...\r\n"
    '"%PYEXE%" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_test=0\r\n'
    "\r\n"
    "rem ---- 4) Locate the freshly installed python and start ----\r\n"
    'py -3 -c "import sys" >nul 2>nul && ( py -3 serve_ratings.py & goto end )\r\n'
    'for /d %%P in ("%LocalAppData%\\Programs\\Python\\Python3*") do '
    'if exist "%%P\\python.exe" ( "%%P\\python.exe" serve_ratings.py & goto end )\r\n'
    "echo.\r\n"
    "echo Installed, but PATH not refreshed yet. Please double-click this file\r\n"
    "echo again (or restart the PC). / 已安装，请再次双击本文件，或重启电脑。\r\n"
    "goto manual\r\n"
    "\r\n"
    ":dlfail\r\n"
    "echo.\r\n"
    "echo Download failed (no internet or blocked). / 下载失败（无网络或被拦截）。\r\n"
    "\r\n"
    ":manual\r\n"
    "echo.\r\n"
    "echo ------------------------------------------------------------\r\n"
    "echo  No problem — open the rating sheet directly instead:\r\n"
    "echo  没关系，直接打开评分表即可：\r\n"
    "echo   1) Double-click rating_sheet_*.html in Chrome or Edge\r\n"
    "echo      用 Chrome / Edge 双击打开 rating_sheet_*.html\r\n"
    'echo   2) Click "Link file" / 点右上角“连接文件”, it then auto-saves.\r\n'
    "echo   (or use Export JSON / Import to save manually)\r\n"
    "echo ------------------------------------------------------------\r\n"
    "echo.\r\n"
    ":end\r\n"
    "pause\r\n"
)


# One-click EXE builder (run once on any Windows PC that has Python) — produces
# Start_rating.exe, a standalone launcher doctors can double-click with NO Python.
BUILD_EXE_BAT = (
    "@echo off\r\n"
    "chcp 65001 >nul\r\n"
    'cd /d "%~dp0"\r\n'
    "echo Building Start_rating.exe (needs Python + internet, one time) ...\r\n"
    "echo 正在构建 Start_rating.exe（需要 Python + 联网，仅一次）...\r\n"
    'py -3 -c "import sys" >nul 2>nul && (set PY=py -3 & goto build)\r\n'
    'python -c "import sys" >nul 2>nul && (set PY=python & goto build)\r\n'
    "echo No Python found. Install from https://www.python.org/downloads/ first.\r\n"
    "echo 未找到 Python，请先安装：https://www.python.org/downloads/\r\n"
    "pause & exit /b 1\r\n"
    ":build\r\n"
    "%PY% -m pip install --user --upgrade pyinstaller || (echo pip/pyinstaller install failed & pause & exit /b 1)\r\n"
    '%PY% -m PyInstaller --onefile --console --name Start_rating serve_ratings.py\r\n'
    "if exist dist\\Start_rating.exe (\r\n"
    "  echo.\r\n"
    "  echo Done. Your exe is: dist\\Start_rating.exe\r\n"
    "  echo 完成！exe 在 dist\\Start_rating.exe —— 把它和 rating_sheet_*.html 放同一文件夹发给医生即可。\r\n"
    ") else ( echo Build failed. Check the messages above. )\r\n"
    "pause\r\n"
)

BUILD_EXE_README = (
    "# 生成免安装的 Start_rating.exe（一次性，在装了 Python 的 Windows 上做）\n\n"
    "目的：让医生**双击 exe 即用、完全不需要装 Python**。\n\n"
    "步骤：\n"
    "1. 在一台装了 Python 的 Windows 电脑上，把 `serve_ratings.py` 和本文件夹里的\n"
    "   `build_exe.bat` 放在一起。\n"
    "2. 双击 `build_exe.bat`，等它跑完（会自动安装 PyInstaller 并打包）。\n"
    "3. 生成的 `dist\\Start_rating.exe` 就是成品。把这个 `Start_rating.exe` 跟某位医生的\n"
    "   `rating_sheet_X.html` 放在**同一个文件夹**里发给他。\n"
    "4. 医生双击 `Start_rating.exe` → 自动打开评分表、自动存进该文件夹、下次自动读回，\n"
    "   全程不需要 Python。评分时保持那个黑色窗口开着。\n\n"
    "说明：exe 只能在 Windows 上构建；构建机和医生机都用 64 位 Windows 即可。\n"
    "杀毒/SmartScreen 可能对自制 exe 报警，点“仍要运行”即可（这是未签名 exe 的正常提示）。\n\n"
    "---\n\n"
    "# Build a no-install Start_rating.exe (one-time, on a Windows PC with Python)\n\n"
    "1. Put `serve_ratings.py` and `build_exe.bat` together on a Windows PC that has Python.\n"
    "2. Double-click `build_exe.bat`; it installs PyInstaller and packages everything.\n"
    "3. The result is `dist\\Start_rating.exe`. Put that exe in the SAME folder as a\n"
    "   doctor's `rating_sheet_X.html` and send both.\n"
    "4. The doctor double-clicks `Start_rating.exe` — opens the sheet, auto-saves into\n"
    "   the folder, auto-loads next time, no Python needed. Keep the black window open.\n"
)


def write_server_assets(out_dir: str):
    """Write the local-server script + double-click launchers into out_dir, plus
    a build_exe/ kit for producing a no-Python Start_rating.exe on Windows."""
    sp = os.path.join(out_dir, "serve_ratings.py")
    with open(sp, "w", encoding="utf-8") as f:
        f.write(SERVE_SCRIPT)
    os.chmod(sp, 0o755)
    cmd = os.path.join(out_dir, "Start_rating.command")
    with open(cmd, "w", encoding="utf-8") as f:
        f.write(LAUNCH_COMMAND)
    os.chmod(cmd, 0o755)
    bat = os.path.join(out_dir, "Start_rating.bat")
    with open(bat, "w", encoding="utf-8", newline="") as f:
        f.write(LAUNCH_BAT)
    # EXE build kit (optional, for Windows-without-Python doctors)
    be = os.path.join(out_dir, "build_exe")
    os.makedirs(be, exist_ok=True)
    import shutil as _sh
    _sh.copyfile(sp, os.path.join(be, "serve_ratings.py"))
    with open(os.path.join(be, "build_exe.bat"), "w", encoding="utf-8", newline="") as f:
        f.write(BUILD_EXE_BAT)
    with open(os.path.join(be, "HOW_TO_BUILD_EXE.md"), "w", encoding="utf-8") as f:
        f.write(BUILD_EXE_README)
    print(f"  ✓ wrote {sp}, Start_rating.command, Start_rating.bat, build_exe/")


def run(args) -> int:
    llm_run = os.path.abspath(args.llm_run)
    src = os.path.join(llm_run, "llm_comparison.json")
    if not os.path.exists(src):
        print(f"[ratings] ERROR: not found: {src}")
        print("[ratings]   Pass --llm-run pointing at an llm_run_<ts>/ folder.")
        return 2
    with open(src, "r", encoding="utf-8") as f:
        llm_data = json.load(f)

    n_patients = len(llm_data.get("patients", []))
    n_models = len(llm_data.get("meta", {}).get("models", []))
    if n_patients == 0 or n_models == 0:
        print(f"[ratings] ERROR: empty LLM comparison: {n_patients} patients × {n_models} models")
        return 2

    out_dir = args.output_dir or os.path.join(llm_run, "ratings")
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sheet_id = f"{os.path.basename(llm_run.rstrip('/'))}__{ts}"

    print(f"[ratings] Source: {src}")
    print(f"[ratings] Patients: {n_patients}  ·  Models: {n_models}  ·  Per doctor: {n_patients * n_models} ratings")
    print(f"[ratings] Doctors: {', '.join(args.doctors)}")
    print(f"[ratings] Output: {out_dir}\n")

    full_recs = load_full_bio_records()

    for i, doc in enumerate(args.doctors):
        build_one_doctor(doc, i, llm_data, args, sheet_id, out_dir, full_recs)

    write_readme(out_dir, args.doctors, n_patients * n_models, n_models, n_patients, args.lang)
    write_server_assets(out_dir)

    if not args.no_excel:
        write_patient_excel(out_dir, llm_data, args.lang, full_recs)

    print(f"\n[ratings] Done. Sheets in: {out_dir}")
    print(f"[ratings] Hand each doctor their own .html file; keep the blinding/ folder")
    print(f"[ratings]   and patient_information.xlsx private (real patient IDs).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build blinded human-rater HTML sheets from an LLM comparison run.")
    p.add_argument("--llm-run", required=True, dest="llm_run",
                   help="path to an llm_run_<ts>/ folder produced by compare_llm_outputs.py")
    p.add_argument("--doctors", nargs="+",
                   default=["医生1", "医生2", "医生3"],
                   help="doctor names (default: 医生1 医生2 医生3)")
    p.add_argument("--output-dir", default=None, dest="output_dir",
                   help="defaults to <llm-run>/ratings/")
    p.add_argument("--seed", type=int, default=42,
                   help="seed for the per-doctor letter & patient shuffles")
    p.add_argument("--lang", default="zh", choices=["en", "zh"],
                   help="primary language for the HTML sheet (toggleable in the UI)")
    p.add_argument("--no-excel", action="store_true", dest="no_excel",
                   help="skip writing patient_information.xlsx (the un-blinded master table)")
    return p


def main():
    args = build_parser().parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n[ratings] interrupted."); return 130


if __name__ == "__main__":
    sys.exit(main())
