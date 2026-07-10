"""
CARE-Agent — Agent C: Risk Synthesizer
=======================================
Receives a PatientSnapshot (Agent A) and RaImagingReport (Agent B),
loads the pre-trained Cox proportional-hazards model, computes the
individual 18-month event probability, applies clinical decision rules,
and outputs a structured ClinicalRiskReport with recommended actions
and a natural-language summary for the clinician.

Cox model variables (must match training):
    age, sex, af, ckd, nyha_class, egfr, logntprobnp,
    mitral_ee, pasp, rvef, ra_reservoir
"""

import json
import os
import pickle
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from agent_a_data_harvester import PatientSnapshot
from agent_b_imaging_analyst import RaImagingReport
from llm_client import LLMClient, LLMResponse
from translations import (
    RISK_TIER_ZH, RA_FUNCTION_ZH, RA_GRADE_ZH, DECISION_RULES_ZH,
)


# ─── Paths ────────────────────────────────────────────────────────────────────
# Cox model lives in data/; legacy root location is a fallback.
_HERE = os.path.dirname(os.path.abspath(__file__))
COX_MODEL_PATH = (
    os.path.join(_HERE, "data", "cox_model.pkl")
    if os.path.exists(os.path.join(_HERE, "data", "cox_model.pkl"))
    else os.path.join(_HERE, "cox_model.pkl")
)

# ─── Model feature list (must match training order) ───────────────────────────
COX_FEATURES = [
    "age", "sex", "af", "ckd", "nyha_class",
    "egfr", "logntprobnp", "mitral_ee", "pasp", "rvef", "ra_reservoir",
]

# ─── Risk probability thresholds → risk label ─────────────────────────────────
# Recalibrated 2026-06: tiers = cohort quartiles of the 18-MONTH predicted
# event probability. The Cox model's duration is in MONTHS; absolute risk is reported
# at 18 months (cohort median follow-up, adequate risk set). Q1/Q2/Q3 = 7.79/12.05/23.14%.
PROB_THRESHOLDS = [
    (0.2314, "CRITICAL"),
    (0.1205, "HIGH"),
    (0.0779, "MODERATE"),
    (0.00, "LOW"),
]

# ─── Clinical decision rules ─────────────────────────────────────────────────
#  Each rule: (condition_fn, action_code, priority, action_text)
DECISION_RULES = [
    # --- Critical / urgent ---
    (
        lambda s, r: r.overall_ra_function == "Severely impaired (RA myopathy)",
        "URGENT_CARDIOLOGY",
        1,
        "Urgent cardiology referral — RA myopathy confirmed. "
        "Consider advanced imaging (cardiac MRI) and specialist review within 2 weeks.",
    ),
    (
        lambda s, r: s.risk_tier == "CRITICAL" and r.overall_ra_function in
                     ("Severely impaired", "Severely impaired (RA myopathy)"),
        "MDT_REVIEW",
        1,
        "Multidisciplinary team (MDT) review recommended — "
        "critical risk tier combined with severe RA dysfunction.",
    ),
    # --- Cardiorenal syndrome ---
    (
        lambda s, r: r.ckd_flag and r.ra_composite_score is not None and r.ra_composite_score < 50,
        "CARDIORENAL_MANAGEMENT",
        2,
        "Cardiorenal syndrome management — CKD with impaired RA function. "
        "Optimise diuresis carefully (target euvolaemia without AKI risk); "
        "avoid nephrotoxins; nephrology co-management advised. "
        "Consider SGLT2i if eGFR ≥20 mL/min/1.73m² — "
        "cardiorenal protection demonstrated in EMPEROR-Reduced and DAPA-HF trials.",
    ),
    # --- AF + RA impairment ---
    (
        lambda s, r: s.af == 1 and r.overall_ra_function in
                     ("Moderately impaired", "Severely impaired", "Severely impaired (RA myopathy)"),
        "AF_RHYTHM_REVIEW",
        2,
        "AF rhythm/rate control review — atrial fibrillation with moderate-severe RA impairment. "
        "Consider rhythm control strategy; anticoagulation status review.",
    ),
    # --- High NT-proBNP ---
    (
        lambda s, r: s.ntprobnp is not None and s.ntprobnp > 2000,
        "DIURETIC_OPTIMISATION",
        3,
        "Diuretic optimisation indicated — NT-proBNP >2000 pg/mL suggests persistent congestion. "
        "Review current loop diuretic dose and fluid status.",
    ),
    # --- Metabolic (TyG) ---
    (
        lambda s, r: s.tyg_index is not None and s.tyg_index > 9.5,
        "METABOLIC_INTERVENTION",
        3,
        "Metabolic intervention — elevated TyG index (insulin resistance surrogate; TyG >9.5 "
        "associated with HR 2.09 for all-cause death and 2.31 for cardiovascular death "
        "in HF+CKD: Chen et al. ESC Heart Fail 2024). "
        "Lifestyle modification; consider SGLT2i (dual metabolic and cardiorenal benefit); "
        "endocrinology referral if T2DM suboptimally controlled.",
    ),
    # --- Pulmonary hypertension ---
    (
        lambda s, r: s.pasp is not None and s.pasp > 50,
        "PULM_HTN_WORKUP",
        3,
        "Pulmonary hypertension workup — PASP >50 mmHg. "
        "Right heart catheterisation consideration; pulmonology/cardiology review.",
    ),
    # --- Routine follow-up ---
    (
        lambda s, r: True,   # always fires as baseline
        "ROUTINE_FOLLOWUP",
        4,
        "Routine HFpEF follow-up — 3-monthly clinical review, repeat echo at 12 months, "
        "optimise RAAS and SGLT2i therapy per current guidelines.",
    ),
]


# ─── Clinical Risk Report dataclass ───────────────────────────────────────────
@dataclass
class ClinicalRiskReport:
    patient_id:             str
    agent_c_status:         str = "PENDING"   # PASS / WARN / FAIL

    # ── Cox model output ──────────────────────────────────────────────────────
    cox_linear_predictor:   Optional[float] = None   # raw log-hazard
    cox_2yr_event_prob:     Optional[float] = None   # 0–1
    cox_risk_label:         str  = "Unknown"          # LOW/MODERATE/HIGH/CRITICAL
    cox_model_complete:     bool = False              # all features present?
    missing_cox_features:   list = field(default_factory=list)

    # ── Integrated risk assessment ─────────────────────────────────────────────
    integrated_risk_tier:   str  = "Unknown"   # combines Agent A tier + Cox + RA function
    risk_drivers:           list = field(default_factory=list)   # English
    risk_drivers_zh:        list = field(default_factory=list)   # Chinese

    # ── Clinical actions ───────────────────────────────────────────────────────
    primary_action:         str  = ""
    all_actions:            list = field(default_factory=list)   # list of (code, priority, en_text)

    # ── Natural-language summary ───────────────────────────────────────────────
    clinical_summary:       str  = ""
    summary_zh:             str  = ""   # Full Chinese clinical report

    # ── LLM reasoning (Agent C AI layer) — primary call ──────────────────────
    llm_reasoning:          str  = ""    # Raw LLM output (fallback if JSON parse fails)
    llm_model_used:         str  = ""
    llm_provider:           str  = ""
    llm_available:          bool = False
    llm_error:              str  = ""
    llm_tokens_used:        int  = 0

    # ── Structured LLM sections (parsed from JSON response) ───────────────────
    llm_mechanism:          str  = ""    # Mechanistic explanation of risk interaction
    llm_clinical_priority:  str  = ""    # Single most important clinical decision
    llm_watchpoints:        list = field(default_factory=list)  # 2–3 specific monitoring points
    llm_patient_note:       str  = ""    # Plain-language patient-friendly summary
    llm_model_caveat:       str  = ""    # AI model limitations for this specific patient
    llm_structured:         bool = False # True if JSON was parsed successfully

    # ── Contradiction check (second LLM call) ─────────────────────────────────
    llm_contradiction_flags:     list = field(default_factory=list)  # [{action_code, concern, severity}]
    llm_contradiction_summary:   str  = ""
    llm_contradiction_available: bool = False
    llm_contradiction_error:     str  = ""
    llm_contradiction_tokens:    int  = 0

    # ── Metadata ──────────────────────────────────────────────────────────────
    agent_c_notes:          str  = ""


# ─── Agent C ──────────────────────────────────────────────────────────────────
class RiskSynthesizerAgent:
    """
    Agent C — integrates Agent A + B outputs with Cox model to produce
    a patient-specific clinical risk report and recommended actions.
    """

    def __init__(self, cox_model_path: str = COX_MODEL_PATH):
        print(f"[Agent C] Loading Cox model from: {cox_model_path}")
        with open(cox_model_path, "rb") as f:
            self._cox = pickle.load(f)
        print("[Agent C] Cox model loaded — C-index 0.741 (validation cohort).")

    # ── Main entry point ───────────────────────────────────────────────────────
    def synthesise(
        self,
        snapshot: PatientSnapshot,
        ra_report: RaImagingReport,
        llm_client: Optional[LLMClient] = None,
        lang: str = "en",
    ) -> ClinicalRiskReport:
        print(f"\n[Agent C] ── Risk synthesis for: {snapshot.patient_id} (lang={lang}) ──")

        report = ClinicalRiskReport(patient_id=snapshot.patient_id)

        self._run_cox(report, snapshot)
        self._integrate_risk(report, snapshot, ra_report)
        self._apply_decision_rules(report, snapshot, ra_report)
        self._write_summary(report, snapshot, ra_report, lang=lang)

        # LLM reasoning + contradiction check — only if a client is provided
        if llm_client is not None:
            self._llm_reasoning(report, snapshot, ra_report, llm_client, lang=lang)
            self._llm_contradiction_check(report, snapshot, ra_report, llm_client, lang=lang)

        print(
            f"[Agent C] Integrated risk: {report.integrated_risk_tier} | "
            f"18-mo event prob: "
            + (f"{report.cox_2yr_event_prob:.1%}" if report.cox_2yr_event_prob is not None else "N/A")
            + f" | Primary action: {report.primary_action}"
            + (f" | LLM: {report.llm_model_used}" if report.llm_available else "")
        )
        return report

    # ── Step 1: Cox model prediction ───────────────────────────────────────────
    def _run_cox(self, report: ClinicalRiskReport, snap: PatientSnapshot):
        # Convert sex string to numeric (Female=0, Male=1)
        sex_numeric = None
        if snap.sex is not None:
            sex_numeric = 1 if str(snap.sex).strip().lower() in ("male", "男", "1") else 0

        feature_map = {
            "age":         snap.age,
            "sex":         sex_numeric,
            "af":          snap.af,
            "ckd":         snap.ckd,
            "nyha_class":  snap.nyha_class,
            "egfr":        snap.egfr,
            "logntprobnp": snap.log_ntprobnp,
            "mitral_ee":   snap.mitral_ee,
            "pasp":        snap.pasp,
            "rvef":        snap.rvef,
            "ra_reservoir":snap.ra_reservoir,
        }

        missing = [k for k, v in feature_map.items() if v is None]
        report.missing_cox_features = missing
        report.cox_model_complete = len(missing) == 0

        if missing:
            report.agent_c_status = "WARN"

        # Impute missing with cohort medians from training data
        medians = {
            "age": 67.0, "sex": 0.0, "af": 0.0, "ckd": 0.0,
            "nyha_class": 2.0, "egfr": 72.0, "logntprobnp": 6.8,
            "mitral_ee": 14.5, "pasp": 38.0, "rvef": 55.0, "ra_reservoir": 30.0,
        }
        row = {k: (v if v is not None else medians[k]) for k, v in feature_map.items()}

        df = pd.DataFrame([row])[COX_FEATURES]

        # Predict cumulative hazard at 18 months and convert to event probability.
        # NOTE: Cox model duration is in MONTHS (timeline ~10.7-25.9). Horizon = 18
        # months (cohort median follow-up; risk set adequate). The earlier times=[730]
        # clamped to the unstable tail of the baseline hazard and grossly over-predicted.
        try:
            cum_haz = self._cox.predict_cumulative_hazard(df, times=[18])
            h = float(cum_haz.values[0][0])
            prob = 1.0 - np.exp(-h)
            report.cox_linear_predictor = float(self._cox.predict_log_partial_hazard(df).values[0])
            report.cox_2yr_event_prob = round(float(prob), 4)
        except Exception as e:
            print(f"[Agent C] Cox prediction error: {e}")
            report.cox_2yr_event_prob = None
            report.agent_c_status = "WARN"
            return

        # Map probability to label
        for thresh, label in PROB_THRESHOLDS:
            if report.cox_2yr_event_prob >= thresh:
                report.cox_risk_label = label
                break

        if report.agent_c_status == "PENDING":
            report.agent_c_status = "PASS"

    # ── Step 2: Integrated risk tier ───────────────────────────────────────────
    def _integrate_risk(
        self,
        report: ClinicalRiskReport,
        snap: PatientSnapshot,
        ra: RaImagingReport,
    ):
        drivers_en, drivers_zh = [], []

        if report.cox_2yr_event_prob is not None:
            drivers_en.append(
                f"Cox model 18-month event probability: {report.cox_2yr_event_prob:.1%} ({report.cox_risk_label})"
            )
            drivers_zh.append(
                f"Cox 模型 18 个月事件概率：{report.cox_2yr_event_prob:.1%}"
                f"（{RISK_TIER_ZH.get(report.cox_risk_label, report.cox_risk_label)}）"
            )

        if snap.risk_tier in ("CRITICAL", "HIGH"):
            drivers_en.append(f"Agent A risk tier: {snap.risk_tier} ({len(snap.risk_flags)} flags)")
            drivers_zh.append(
                f"Agent A 风险层级：{RISK_TIER_ZH.get(snap.risk_tier, snap.risk_tier)}"
                f"（{len(snap.risk_flags)} 项标记）"
            )

        if ra.overall_ra_function not in ("Normal", "Unknown"):
            drivers_en.append(f"RA function: {ra.overall_ra_function} (composite {ra.ra_composite_score}/100)")
            drivers_zh.append(
                f"RA 功能：{RA_FUNCTION_ZH.get(ra.overall_ra_function, ra.overall_ra_function)}"
                f"（综合评分 {ra.ra_composite_score}/100）"
            )

        if ra.af_adjustment_applied:
            drivers_en.append("AF present — booster strain unreliable")
            drivers_zh.append("合并房颤 — 助推期应变不可靠，以储存期和管道期为主要参考")

        if ra.ckd_flag and ra.ra_composite_score is not None and ra.ra_composite_score < 50:
            drivers_en.append("CKD with impaired RA function — cardiorenal syndrome risk")
            drivers_zh.append("CKD 合并 RA 功能受损 — 心肾综合征风险")

        report.risk_drivers    = drivers_en
        report.risk_drivers_zh = drivers_zh

        # Integrate: take the highest severity across all sources
        tier_rank = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3, "Unknown": -1}
        tiers = [
            snap.risk_tier,
            report.cox_risk_label,
        ]
        # RA function → tier mapping
        ra_to_tier = {
            "Normal": "LOW",
            "Mildly impaired": "MODERATE",
            "Moderately impaired": "HIGH",
            "Severely impaired": "CRITICAL",
            "Severely impaired (RA myopathy)": "CRITICAL",
            "Unknown": "Unknown",
        }
        tiers.append(ra_to_tier.get(ra.overall_ra_function, "Unknown"))

        valid_tiers = [t for t in tiers if t in tier_rank and t != "Unknown"]
        if valid_tiers:
            report.integrated_risk_tier = max(valid_tiers, key=lambda t: tier_rank[t])
        else:
            report.integrated_risk_tier = snap.risk_tier  # fallback

    # ── Step 3: Decision rules ─────────────────────────────────────────────────
    def _apply_decision_rules(
        self,
        report: ClinicalRiskReport,
        snap: PatientSnapshot,
        ra: RaImagingReport,
    ):
        actions = []
        for condition_fn, code, priority, text in DECISION_RULES:
            try:
                if condition_fn(snap, ra):
                    actions.append((code, priority, text))
            except Exception:
                pass

        # Sort by priority (1 = highest)
        actions.sort(key=lambda x: x[1])
        report.all_actions = actions

        # Primary action = first (highest priority)
        if actions:
            report.primary_action = actions[0][0]

    # ── Step 4: Natural-language summaries ─────────────────────────────────────
    def _write_summary(
        self,
        report: ClinicalRiskReport,
        snap: PatientSnapshot,
        ra: RaImagingReport,
        lang: str = "en",
    ):
        # ── Shared helpers ──────────────────────────────────────────────────────
        # snap.sex is "Male"/"Female" (set by Agent A); normalise robustly so the
        # report never silently defaults to the wrong sex.
        is_male = str(snap.sex).strip().lower() in ("male", "m", "男", "1")
        age_sex = (
            f"{int(snap.age)}-year-old {'male' if is_male else 'female'}"
            if snap.age is not None else "patient"
        )
        prob_str = f"{report.cox_2yr_event_prob:.1%}" if report.cox_2yr_event_prob is not None else "unavailable"
        missing_note = (
            f" (Note: {len(report.missing_cox_features)} Cox features imputed: "
            f"{', '.join(report.missing_cox_features[:3])}{'...' if len(report.missing_cox_features) > 3 else ''})"
            if report.missing_cox_features else ""
        )

        comorbidities = []
        if snap.af == 1:
            comorbidities.append("AF")
        if snap.ckd == 1:
            comorbidities.append("CKD")
        if snap.diabetes == 1:
            comorbidities.append("DM")
        if snap.hypertension == 1:
            comorbidities.append("hypertension")
        comorbid_str = ", ".join(comorbidities) if comorbidities else "no major comorbidities flagged"

        action_lines = "\n".join(
            f"  [{i+1}] ({code}) {text}"
            for i, (code, _, text) in enumerate(report.all_actions[:4])
        )

        # --- English summary ---
        report.clinical_summary = (
            f"CARE-Agent Risk Report — {snap.patient_id}\n"
            f"{'─' * 55}\n"
            f"Patient: {age_sex}, NYHA class {int(snap.nyha_class) if snap.nyha_class else 'N/A'}, "
            f"comorbidities: {comorbid_str}.\n\n"
            f"Integrated risk tier: {report.integrated_risk_tier}\n"
            f"Cox 18-month event probability: {prob_str}{missing_note}\n\n"
            f"RA function summary:\n"
            f"  Reservoir {ra.ra_reservoir:.1f}% ({ra.reservoir_grade}) | "
            f"Conduit {ra.ra_conduit:.1f}% ({ra.conduit_grade}) | "
            f"Booster {ra.ra_booster:.1f}% ({ra.booster_grade})\n"
            f"  Overall: {ra.overall_ra_function} | Composite score: {ra.ra_composite_score}/100\n"
            + (f"  AF note: {ra.booster_af_note}\n" if ra.af_adjustment_applied else "")
            + (f"  QC flags: {', '.join(ra.imaging_quality_flags)}\n" if ra.imaging_quality_flags else "")
            + f"\nKey risk drivers:\n"
            + "\n".join(f"  • {d}" for d in report.risk_drivers)
            + f"\n\nRecommended actions:\n{action_lines}\n"
            f"{'─' * 55}\n"
            f"Generated by CARE-Agent | Simulation study — not for clinical use."
        )

        # --- Full Chinese summary ---
        tier_zh = RISK_TIER_ZH.get(report.integrated_risk_tier, report.integrated_risk_tier)

        ra_fn_zh = RA_FUNCTION_ZH.get(ra.overall_ra_function, ra.overall_ra_function)

        res_grade_zh = RA_GRADE_ZH.get(ra.reservoir_grade, ra.reservoir_grade)
        con_grade_zh = RA_GRADE_ZH.get(ra.conduit_grade, ra.conduit_grade)
        bst_grade_zh = RA_GRADE_ZH.get(ra.booster_grade, ra.booster_grade)

        # Comorbidities in Chinese
        comorbid_zh_parts = []
        if snap.af == 1:          comorbid_zh_parts.append("心房颤动")
        if snap.ckd == 1:         comorbid_zh_parts.append(f"慢性肾病{('（' + snap.ckd_stage + '期）') if snap.ckd_stage else ''}")
        if snap.diabetes == 1:    comorbid_zh_parts.append("2型糖尿病")
        if snap.hypertension == 1:comorbid_zh_parts.append("高血压")
        if snap.cad == 1:         comorbid_zh_parts.append("冠状动脉疾病")
        comorbid_zh = "、".join(comorbid_zh_parts) if comorbid_zh_parts else "无主要合并症"

        sex_zh = "男" if str(snap.sex).strip().lower() in ("male", "m", "男", "1") else "女"
        nyha_zh = str(int(snap.nyha_class)) if snap.nyha_class else "未知"
        age_zh = str(int(snap.age)) if snap.age is not None else "未知"

        # Missing features note in Chinese
        if report.missing_cox_features:
            _mf_list = "、".join(report.missing_cox_features[:3])
            _mf_etc  = "等" if len(report.missing_cox_features) > 3 else ""
            missing_note_zh = f"（注：{len(report.missing_cox_features)} 项 Cox 特征已用队列中位数填补：{_mf_list}{_mf_etc}）"
        else:
            missing_note_zh = ""

        # Actions in Chinese
        action_lines_zh = "\n".join(
            f"  [{i+1}] {DECISION_RULES_ZH.get(code, text)}"
            for i, (code, _, text) in enumerate(report.all_actions[:4])
        )

        # RA myopathy note
        ra_myopathy_zh = "阳性" if snap.ra_myopathy == 1 else "阴性"

        # QC / AF notes
        af_note_zh = f"\n  房颤备注：{ra.booster_af_note}" if ra.af_adjustment_applied else ""
        qc_note_zh = f"\n  质控标记：{', '.join(ra.imaging_quality_flags)}" if ra.imaging_quality_flags else ""

        # eGFR / creatinine
        egfr_s  = f"{snap.egfr:.1f}" if snap.egfr is not None else "未知"
        creat_s = f"{snap.creatinine:.0f}" if snap.creatinine is not None else "未知"
        ntpro_s = f"{snap.ntprobnp:.0f}" if snap.ntprobnp is not None else "未知"
        ckd_stage_s = snap.ckd_stage or "未分期"
        lvef_s  = f"{snap.lvef:.1f}" if snap.lvef is not None else "未知"
        lavi_s  = f"{snap.lavi:.1f}" if snap.lavi is not None else "未知"
        pasp_s  = str(snap.pasp) if snap.pasp is not None else "未知"
        rvef_s  = f"{snap.rvef:.1f}" if snap.rvef is not None else "未知"
        tapse_s = str(snap.tapse) if snap.tapse is not None else "未知"
        ee_s    = f"{snap.mitral_ee:.1f}" if snap.mitral_ee is not None else "未知"

        report.summary_zh = (
            f"【CARE-Agent 风险评估报告】{snap.patient_id}\n"
            f"{'─' * 50}\n"
            f"患者信息：{age_zh} 岁，性别：{sex_zh}，NYHA 心功能分级：{nyha_zh} 级\n"
            f"合并症：{comorbid_zh}\n\n"
            f"▌综合风险等级：{tier_zh}\n"
            f"▌Cox 模型 18 个月事件概率：{prob_str}{missing_note_zh}\n\n"
            f"──── 右心房应变分析（Agent B） ────\n"
            f"  储存期应变：{ra.ra_reservoir:.1f}%（{res_grade_zh}）\n"
            f"  管道期应变：{ra.ra_conduit:.1f}%（{con_grade_zh}）\n"
            f"  助推期应变：{ra.ra_booster:.1f}%（{bst_grade_zh}）{af_note_zh}\n"
            f"  整体 RA 功能：{ra_fn_zh}　综合评分：{ra.ra_composite_score}/100\n"
            f"  RA 心肌病：{ra_myopathy_zh}{qc_note_zh}\n\n"
            f"──── 关键生化与影像指标 ────\n"
            f"  eGFR：{egfr_s} mL/min/1.73m²　肌酐：{creat_s} μmol/L　慢性肾病分期：{ckd_stage_s}\n"
            f"  NT-proBNP：{ntpro_s} pg/mL\n"
            f"  LVEF：{lvef_s}%　LAVI：{lavi_s} mL/m²　E/e'：{ee_s}\n"
            f"  PASP：{pasp_s} mmHg　RVEF：{rvef_s}%　TAPSE：{tapse_s} mm\n\n"
            f"──── 主要风险驱动因素 ────\n"
            + "\n".join(f"  • {d}" for d in report.risk_drivers_zh)
            + f"\n\n──── 推荐临床行动方案 ────\n{action_lines_zh}\n"
            f"{'─' * 50}\n"
            f"由 CARE-Agent 生成 | 本报告为模拟研究，仅供科研参考，不作为临床诊疗依据。"
        )

        # Agent C handoff note
        report.agent_c_notes = (
            f"Risk synthesis complete. Integrated tier: {report.integrated_risk_tier}. "
            f"Cox 18-mo prob: {prob_str}. "
            f"Primary action: {report.primary_action}. "
            f"Model complete: {report.cox_model_complete}."
        )

    # ── Shared helper: extract JSON from LLM response ─────────────────────────
    @staticmethod
    def _parse_llm_json(text: str) -> Optional[dict]:
        """Strip markdown fences and parse JSON, tolerantly. Returns None on failure.

        Handles: code-fence wrappers, prose before/after the object, trailing
        commas, and a first-balanced-object scan (so a valid object followed by
        extra text still parses)."""
        if not text:
            return None
        cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

        def _try(s: str):
            try:
                return json.loads(s)
            except Exception:
                # tolerate trailing commas:  {..,}  or  [..,]
                try:
                    return json.loads(re.sub(r",\s*([}\]])", r"\1", s))
                except Exception:
                    return None

        # 1) whole string
        out = _try(cleaned)
        if isinstance(out, dict):
            return out

        # 2) first balanced { ... } object (string-aware brace matching)
        start = cleaned.find("{")
        if start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(cleaned)):
                ch = cleaned[i]
                if in_str:
                    if esc:           esc = False
                    elif ch == "\\":  esc = True
                    elif ch == '"':   in_str = False
                    continue
                if ch == '"':   in_str = True
                elif ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        out = _try(cleaned[start:i + 1])
                        if isinstance(out, dict):
                            return out
                        break

        # 3) greedy fallback: first '{' to last '}'
        m = re.search(r"\{[\s\S]+\}", cleaned)
        if m:
            out = _try(m.group())
            if isinstance(out, dict):
                return out
        return None

    # ── Shared helper: build patient data block ────────────────────────────────
    def _build_data_block(
        self,
        report: ClinicalRiskReport,
        snap: PatientSnapshot,
        ra: RaImagingReport,
    ) -> str:
        """Return a compact structured data block to include in LLM prompts."""
        prob_str    = f"{report.cox_2yr_event_prob:.1%}" if report.cox_2yr_event_prob is not None else "unavailable"
        missing_note = f" ({len(report.missing_cox_features)} features imputed)" if report.missing_cox_features else ""

        comorbidities = []
        if snap.af == 1:          comorbidities.append("atrial fibrillation")
        if snap.ckd == 1:         comorbidities.append(f"CKD {snap.ckd_stage or ''}")
        if snap.diabetes == 1:    comorbidities.append("type 2 diabetes")
        if snap.hypertension == 1:comorbidities.append("hypertension")
        if snap.cad == 1:         comorbidities.append("CAD")
        comorbid_str = ", ".join(comorbidities) if comorbidities else "none documented"

        actions_str = "\n".join(
            f"  - [{code}] {text}" for code, _, text in report.all_actions[:5]
        )

        bmi_s    = f"{snap.bmi:.1f}"       if snap.bmi       is not None else "N/A"
        sbp_s    = str(snap.sbp)            if snap.sbp       is not None else "N/A"
        dbp_s    = str(snap.dbp)            if snap.dbp       is not None else "N/A"
        egfr_s   = f"{snap.egfr:.1f}"      if snap.egfr      is not None else "N/A"
        creat_s  = f"{snap.creatinine:.0f}" if snap.creatinine is not None else "N/A"
        ntpro_s  = f"{snap.ntprobnp:.0f}"  if snap.ntprobnp  is not None else "N/A"
        tyg_s    = f"{snap.tyg_index:.2f}" if snap.tyg_index is not None else "N/A"
        fbg_s    = str(snap.fbg)            if snap.fbg       is not None else "N/A"
        lvef_s   = f"{snap.lvef:.1f}"      if snap.lvef      is not None else "N/A"
        lavi_s   = f"{snap.lavi:.1f}"      if snap.lavi      is not None else "N/A"
        mitral_s = f"{snap.mitral_ee:.1f}" if snap.mitral_ee is not None else "N/A"
        pasp_s   = str(snap.pasp)           if snap.pasp      is not None else "N/A"
        rvef_s   = f"{snap.rvef:.1f}"      if snap.rvef      is not None else "N/A"
        tapse_s  = str(snap.tapse)          if snap.tapse     is not None else "N/A"
        ra_r_s   = f"{ra.ra_reservoir:.1f}" if ra.ra_reservoir is not None else "N/A"
        ra_c_s   = f"{ra.ra_conduit:.1f}"  if ra.ra_conduit  is not None else "N/A"
        ra_b_s   = f"{ra.ra_booster:.1f}"  if ra.ra_booster  is not None else "N/A"
        age_s    = str(int(snap.age))       if snap.age       is not None else "N/A"
        ckd_stage_s = snap.ckd_stage or "N/A"
        tyg_tert_s  = snap.tyg_tertile or "N/A"
        af_note_line = f"AF Note: {ra.booster_af_note}" if ra.af_adjustment_applied else ""

        return f"""=== PATIENT ===
ID: {snap.patient_id}  |  Age: {age_s}  |  Sex: {snap.sex}  |  NYHA: {snap.nyha_class}
Comorbidities: {comorbid_str}
BMI: {bmi_s}  |  BP: {sbp_s}/{dbp_s} mmHg

=== RENAL ===
eGFR: {egfr_s} mL/min/1.73m²  |  CKD Stage: {ckd_stage_s}  |  Creatinine: {creat_s} μmol/L

=== BIOMARKERS ===
NT-proBNP: {ntpro_s} pg/mL  |  TyG: {tyg_s} (tertile: {tyg_tert_s})  |  FBG: {fbg_s} mmol/L

=== ECHO ===
LVEF: {lvef_s}%  |  LAVI: {lavi_s} mL/m²  |  E/e': {mitral_s}
PASP: {pasp_s} mmHg  |  RVEF: {rvef_s}%  |  TAPSE: {tapse_s} mm

=== RA IMAGING (Agent B) ===
Reservoir: {ra_r_s}% ({ra.reservoir_grade})  |  Conduit: {ra_c_s}% ({ra.conduit_grade})  |  Booster: {ra_b_s}% ({ra.booster_grade})
Overall: {ra.overall_ra_function}  |  Composite: {ra.ra_composite_score}/100  |  Consistency: {ra.strain_consistency}
{af_note_line}
RA Myopathy: {'Positive' if snap.ra_myopathy == 1 else 'Negative'}

=== RISK MODEL (Agent C) ===
Cox 18-month event probability: {prob_str}{missing_note}
Cox risk label: {report.cox_risk_label}  |  Integrated tier: {report.integrated_risk_tier}
Risk drivers: {'; '.join(report.risk_drivers)}
Agent A flags: {'; '.join(snap.risk_flags) if snap.risk_flags else 'none'}

=== RULE-BASED ACTIONS ===
{actions_str}"""

    # ── Step 5: Structured LLM Reasoning ──────────────────────────────────────
    def _llm_reasoning(
        self,
        report: ClinicalRiskReport,
        snap: PatientSnapshot,
        ra: RaImagingReport,
        llm_client: LLMClient,
        lang: str = "en",
    ):
        """
        Primary LLM call — returns structured JSON with 5 clinical sections.
        The LLM is explicitly told NOT to restate numbers (already shown in the UI)
        and to focus on synthesis, mechanism, and patient-level nuance.
        """
        is_zh = lang == "zh"

        if is_zh:
            system_prompt = """你是一位专精于射血分数保留型心力衰竭（HFpEF）与心肾综合征的资深心脏科专家，正在审阅 CARE-Agent 流程的结构化分析结果。

重要规则：
• 不要重复数字和检查结果——这些已在界面中显示
• 你的价值在于提供规则系统无法给出的临床洞察：机制分析、综合判断、患者个体化考量
• 针对该患者的具体情况，切忌泛泛而论
• 若存在心肾综合征：请明确评估 SGLT2i 适应症（eGFR ≥20 mL/min/1.73m² 为阈值，EMPEROR/DAPA-HF 证据），并解释其心肾保护机制
• 对于 TyG 指数升高：应解读为胰岛素抵抗放大心肾风险，而非单纯代谢异常

请以 JSON 格式返回以下内容（不含 markdown 代码块）：
{
  "mechanism": "150–200字：解释右心房功能、肾功能与心脏血流动力学在该患者中如何相互作用，形成当前风险格局",
  "clinical_priority": "60–100字：最重要的一个临床决策是什么，以及为什么它排在其他措施之前",
  "watchpoints": ["监测点1（具体指标或情况）", "监测点2", "监测点3"],
  "patient_note": "50–75字：用患者能理解的通俗语言解释病情现状和最重要的注意事项",
  "model_caveat": "40–60字：AI模型对该患者分析的具体局限性（例如缺失特征、数据质量、模型外推范围等）"
}
请全程使用简体中文。"""
        else:
            system_prompt = """You are an expert cardiologist specialising in HFpEF and cardiorenal syndrome, reviewing structured output from the CARE-Agent AI pipeline.

CRITICAL RULES:
• Do NOT restate the numbers or grades — they are already displayed in the dashboard
• Your value is clinical SYNTHESIS that the rule-based pipeline cannot provide: mechanism, interaction, patient-specific nuance
• Be specific to THIS patient — avoid generic statements that would apply to any HFpEF patient
• When cardiorenal syndrome is present: explicitly consider SGLT2i eligibility (eGFR ≥20 threshold per EMPEROR/DAPA-HF) and explain mechanistic rationale
• For elevated TyG index: interpret as insulin resistance amplifying cardiorenal risk, not merely a metabolic incidental finding

Return your analysis as JSON (no markdown code fences):
{
  "mechanism": "150–200 words: explain HOW this patient's RA dysfunction, renal status, and hemodynamics interact to produce their current risk profile — the pathophysiological chain",
  "clinical_priority": "60–100 words: the single most important clinical decision and WHY it takes precedence over others for this specific patient",
  "watchpoints": ["specific thing to monitor 1", "specific thing to monitor 2", "specific thing to monitor 3"],
  "patient_note": "50–75 words: plain language explanation a patient can understand — what is happening and the one most important thing they should know",
  "model_caveat": "40–60 words: specific limitations of the AI model for THIS patient (missing features, data quality, extrapolation concerns, etc.)"
}"""

        data_block = self._build_data_block(report, snap, ra)
        lang_instruction = "请全程使用简体中文。" if is_zh else ""

        user_prompt = f"""{'请分析以下 HFpEF 患者并返回 JSON 格式结果。' if is_zh else 'Analyse this HFpEF patient and return structured JSON.'}

{data_block}

{'请记住：只返回 JSON，不要重复数据，聚焦于规则系统无法捕捉的临床洞察。' if is_zh else 'Remember: return JSON only. Do not restate the data. Focus on clinical insights the rule-based system cannot provide.'}
{lang_instruction}"""

        print(f"[Agent C] Calling LLM for structured reasoning ({llm_client.provider}/{llm_client.model})...")
        llm_resp: LLMResponse = llm_client.complete(system_prompt, user_prompt, force_json=True)

        report.llm_model_used  = llm_resp.model
        report.llm_provider    = llm_resp.provider
        report.llm_tokens_used = llm_resp.tokens_used

        if not llm_resp.success:
            report.llm_available = False
            report.llm_error     = llm_resp.error
            print(f"[Agent C] LLM call failed: {llm_resp.error}")
            return

        report.llm_available = True
        report.llm_reasoning = llm_resp.content  # always store raw output

        # Attempt structured JSON parse
        parsed = self._parse_llm_json(llm_resp.content)
        if parsed and isinstance(parsed, dict):
            report.llm_mechanism         = parsed.get("mechanism", "")
            report.llm_clinical_priority = parsed.get("clinical_priority", "")
            report.llm_watchpoints       = parsed.get("watchpoints", [])
            report.llm_patient_note      = parsed.get("patient_note", "")
            report.llm_model_caveat      = parsed.get("model_caveat", "")
            report.llm_structured        = True
            print(f"[Agent C] Structured LLM output parsed OK ({llm_resp.tokens_used} tokens).")
        else:
            report.llm_structured = False
            print(f"[Agent C] JSON parse failed — storing raw output ({llm_resp.tokens_used} tokens).")

    # ── Step 6: Contradiction / Safety Check ──────────────────────────────────
    def _llm_contradiction_check(
        self,
        report: ClinicalRiskReport,
        snap: PatientSnapshot,
        ra: RaImagingReport,
        llm_client: LLMClient,
        lang: str = "en",
    ):
        """
        Second LLM call — checks whether the recommended actions are safe and
        appropriate given this patient's specific data. Flags contradictions,
        contraindications, and dosing concerns.
        """
        is_zh = lang == "zh"

        if is_zh:
            system_prompt = """你是一位临床安全核查专家，专门审查 AI 生成的 HFpEF 治疗建议是否存在矛盾、禁忌症或潜在危害。

你的任务是：
1. 逐条核查推荐行动方案是否与该患者具体指标存在矛盾：
   - SGLT2i：eGFR <20 mL/min/1.73m² 时禁用；eGFR 20–45 时慎用（参考 EMPEROR-Reduced/DAPA-HF 证据）
   - 袢利尿剂：CKD 4–5期（eGFR <30）大剂量利尿时需警惕急性肾损伤（AKI）风险
   - 心律/心率控制：合并心房颤动（AF）且 RA 功能严重受损时，需评估抗凝充分性（心房淤滞加剧卒中风险）
   - 代谢干预：若 T2DM 患者正在使用胰岛素或磺脲类，需注意低血糖风险
2. 识别规则系统可能遗漏的、需要临床医生特别关注的安全隐患
3. 检查 Cox 模型风险等级与 Agent A 临床风险等级之间是否存在显著不一致，并解释原因

以 JSON 格式返回（不含 markdown 代码块）：
{
  "flags": [
    {"action_code": "行动代码", "concern": "具体安全顾虑或矛盾点（1–2句）", "severity": "warning 或 caution 或 info"}
  ],
  "model_discordance": "如果 Cox 概率与临床风险等级显著不一致，请解释原因；若一致则写 null",
  "overall_safe": true 或 false,
  "summary": "30–50字：总体安全评估"
}
请全程使用简体中文。"""
        else:
            system_prompt = """You are a clinical safety checker reviewing AI-generated HFpEF treatment recommendations for contradictions, contraindications, and inconsistencies.

Your tasks:
1. Check each recommended action against the patient's specific labs for contradictions:
   - SGLT2i: contraindicated if eGFR <20 mL/min/1.73m²; use with caution eGFR 20–45; renal benefit threshold per EMPEROR-Reduced/DAPA-HF
   - Loop diuretics: flag AKI risk if CKD stage 4-5 (eGFR <30) with high-dose diuresis
   - Rate/rhythm control: check anticoagulation adequacy in AF with impaired RA (stroke risk amplified by atrial stasis)
   - Metabolic agents: check for hypoglycaemia risk if T2DM on insulin/sulfonylurea
2. Identify safety concerns the rule-based system may have missed
3. Flag significant discordance between the Cox model risk label and Agent A clinical tier — and explain why they might differ

Return JSON (no markdown fences):
{
  "flags": [
    {"action_code": "ACTION_CODE", "concern": "specific contradiction or safety concern in 1-2 sentences", "severity": "warning or caution or info"}
  ],
  "model_discordance": "explanation if Cox probability and clinical tier disagree significantly, otherwise null",
  "overall_safe": true or false,
  "summary": "30-50 word overall safety assessment"
}"""

        data_block = self._build_data_block(report, snap, ra)
        lang_instruction = "请全程使用简体中文。" if is_zh else ""

        user_prompt = f"""{'请对以下 HFpEF 患者的推荐治疗方案进行安全性核查，返回 JSON 格式结果。' if is_zh else 'Perform a safety check on the recommended actions for this HFpEF patient. Return JSON.'}

{data_block}

{'仅返回 JSON，不要重复数据。' if is_zh else 'Return JSON only. Do not restate the data.'}
{lang_instruction}"""

        print(f"[Agent C] Calling LLM for contradiction check ({llm_client.provider}/{llm_client.model})...")
        llm_resp: LLMResponse = llm_client.complete(system_prompt, user_prompt, force_json=True)
        report.llm_contradiction_tokens = llm_resp.tokens_used

        if not llm_resp.success:
            report.llm_contradiction_available = False
            report.llm_contradiction_error     = llm_resp.error
            print(f"[Agent C] Contradiction check failed: {llm_resp.error}")
            return

        parsed = self._parse_llm_json(llm_resp.content)
        if parsed and isinstance(parsed, dict):
            report.llm_contradiction_flags     = parsed.get("flags", [])
            report.llm_contradiction_summary   = parsed.get("summary", "")
            report.llm_contradiction_available = True
            discordance = parsed.get("model_discordance")
            if discordance:
                report.llm_contradiction_summary = f"{report.llm_contradiction_summary} | {discordance}"
            print(f"[Agent C] Contradiction check: {len(report.llm_contradiction_flags)} flag(s) ({llm_resp.tokens_used} tokens).")
        else:
            report.llm_contradiction_available = False
            report.llm_contradiction_error     = "JSON parse failed"
            print("[Agent C] Contradiction check JSON parse failed.")
