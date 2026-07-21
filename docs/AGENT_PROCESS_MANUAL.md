# CARE-Agent — Process Manual

> *How a Cox model number becomes a severity label and a clinical action list.*
>
> Companion document to the source code in `agent_a_data_harvester.py`,
> `agent_b_imaging_analyst.py`, `agent_c_risk_synthesizer.py`, and `llm_client.py`.
> Verified against the code as of 2026‑05‑26.

---

## 0. The pipeline at a glance

```
┌─────────────┐    PatientSnapshot    ┌─────────────┐   RaImagingReport   ┌──────────────┐
│   Agent A   │ ────────────────────► │   Agent B   │ ──────────────────► │   Agent C    │
│ Data        │                       │ RA strain   │                     │ Risk         │
│ harvester   │                       │ analyst     │                     │ synthesizer  │
│ (Excel/EHR) │                       │             │                     │  + LLM       │
└─────────────┘                       └─────────────┘                     └──────────────┘
       │                                     │                                     │
       │ raw labs, echo, comorbidities       │ reservoir / conduit / booster       │ Cox 18-mo P(event)
       │ + Agent-A risk tier (flag-count)    │ + composite score 0–100             │ + integrated tier
       │                                     │ + overall RA function grade         │ + ranked action list
       │                                     │                                     │ + LLM narrative
       │                                     │                                     │ + LLM safety check
       └─────────────────────────────────────┴─────────────────────────────────────┘
```

Each agent **does not overwrite** the previous one — it adds a layer of judgement.
Agent C is where “numbers” become “severity + advice”.

---

## 1. Agent A — Data Harvester (and the first severity opinion)

`agent_a_data_harvester.py`

### 1.1 What it does

1. Loads `basic_data.xlsx` (sheet `395全量HFpEF数据`, 395 patients) and any user‑uploaded extension.
   Patient names are **not** unique (9 repeat in this cohort); Agent A assigns a unique `patient_id` (repeats disambiguated as \"name (2)\"/\"name (3)\") so every row is processed exactly once.
2. For one `patient_id` it builds a `PatientSnapshot` containing demographics, comorbidities, meds, labs, echo, RA strain and known outcomes.
3. Derives `ckd_stage` from eGFR (G1 ≥90 / G2 60–89 / G3a 45–59 / G3b 30–44 / G4 15–29 / G5 <15) and recomputes `tyg_index` if missing (`ln(TG·FBG / 2)`, with mmol→mg/dL).
4. Runs `_validate_quality()` — flags `WARN` if any of `egfr / ntprobnp / lvef / ra_reservoir / tyg_index` are missing, or if values are out of physiological range.

### 1.2 Risk flags (clinical thresholds → boolean flags)

| Threshold (`THRESHOLDS` dict)         | Flag emitted                               |
| ------------------------------------- | ------------------------------------------ |
| eGFR < 30                             | `SEVERE_CKD`                               |
| 30 ≤ eGFR < 60                        | `CKD_CONFIRMED`                            |
| NT‑proBNP > 2000                      | `CRITICAL_NT-proBNP`                       |
| 900 < NT‑proBNP ≤ 2000                | `ELEVATED_NT-proBNP`                       |
| TyG > 9.5                             | `HIGH_INSULIN_RESISTANCE`                  |
| RA reservoir < 20                     | `SEVERELY_IMPAIRED_RA`                     |
| 20 ≤ RA reservoir < 30                | `IMPAIRED_RA_RESERVOIR`                    |
| PASP > 35                             | `PULMONARY_HYPERTENSION`                   |
| RVEF < 45                             | `RV_DYSFUNCTION`                           |
| E/e′ > 14                             | `ELEVATED_FILLING_PRESSURE`                |
| AF = 1                                | `ATRIAL_FIBRILLATION`                      |
| `ra_myopathy` = 1                     | `RA_MYOPATHY_POSITIVE`                     |

### 1.3 Agent A’s severity tier — *flag‑count rule*

`_assign_risk_tier()` (lines 362–373):

```python
critical_flags = [f for f in flags if 'CRITICAL' in f or 'SEVERE' in f or 'MYOPATHY' in f]

if critical_flags or n_flags >= 4: tier = 'CRITICAL'
elif n_flags >= 3:                  tier = 'HIGH'
elif n_flags >= 1:                  tier = 'MODERATE'
else:                               tier = 'LOW'
```

This is the **clinical / heuristic** severity — purely from thresholds, no model.
It is one of the three inputs Agent C will later combine.

---

## 2. Agent B — RA Imaging Analyst

`agent_b_imaging_analyst.py`

### 2.1 Phase grading (`RA_GRADES`)

| Phase     | Normal   | Mildly impaired | Moderately impaired | Severely impaired |
| --------- | -------- | --------------- | ------------------- | ----------------- |
| Reservoir | ≥ 35 %   | 25 – 35         | 20 – 25             | < 20              |
| Conduit   | ≥ 18 %   | 12 – 18         | 8 – 12              | < 8               |
| Booster   | ≥ 14 %   | 10 – 14         | 6 – 10              | < 6               |

### 2.2 Composite score (0–100, higher = better)

```python
score = ( 0.50 · min(reservoir/60, 1)
        + 0.30 · min(conduit/40,   1)
        + 0.20 · min(booster/25,   1) ) · 100
```

(`WEIGHTS = {"reservoir": 0.50, "conduit": 0.30, "booster": 0.20}`, normalisation
maxima 60 / 40 / 25.)

### 2.3 Overall RA function grade (`_overall_grade`)

```
ra_myopathy == 1  → "Severely impaired (RA myopathy)"     # short-circuits
else, by composite score:
  ≥ 65   → Normal
  50–64  → Mildly impaired
  35–49  → Moderately impaired
  < 35   → Severely impaired
```

### 2.4 Contextual flags Agent C will use later

* **AF present** → sets `af_adjustment_applied = True`; booster strain is reinterpreted as passive recoil (not active contraction). Reservoir + conduit become the reliable phases.
* **CKD + reservoir < 30** → adds a `CKD_CONTEXT` quality flag suggesting SGLT2i eligibility check.
* **Consistency**: reservoir should ≈ conduit + booster within 25 %; otherwise an `Inconsistent` flag fires (image‑quality issue).

---

## 3. Agent C — Risk Synthesizer (the heart of the manual)

`agent_c_risk_synthesizer.py` · `RiskSynthesizerAgent.synthesise()` runs six numbered steps. Steps 1–4 are deterministic; steps 5–6 are the optional LLM layer.

### 3.1 Step 1 — Cox model prediction (`_run_cox`, lines 226–284)

#### 3.1.1 Inputs

The model uses 11 features in this fixed order:

```python
COX_FEATURES = ["age", "sex", "af", "ckd", "nyha_class",
                "egfr", "logntprobnp", "mitral_ee", "pasp", "rvef", "ra_reservoir"]
```

`sex` is encoded numerically (Female = 0, Male = 1; robust to "男" / "1" strings).

#### 3.1.2 Missing data → cohort‑median imputation

If any feature is missing, Agent A’s status is downgraded to `WARN` and the value is filled from training‑cohort medians:

```python
medians = {"age":67, "sex":0, "af":0, "ckd":0, "nyha_class":2,
           "egfr":72, "logntprobnp":6.8, "mitral_ee":14.5,
           "pasp":38, "rvef":55, "ra_reservoir":30}
```

Each imputed feature is listed in `report.missing_cox_features` and surfaced
in the patient report as “Cox features imputed: …”.

#### 3.1.3 From the model to a probability

The model is a `lifelines.CoxPHFitter` loaded from `cox_model.pkl`
(apparent C‑index 0.741, optimism-corrected 0.579, in the 395-patient development
cohort — the model is used as a fixed risk-ranking component and has not been
externally validated).

```python
cum_haz = self._cox.predict_cumulative_hazard(df, times=[18])    # 18 months
H       = float(cum_haz.values[0][0])
prob    = 1.0 - np.exp(-H)              # 18-month event probability
report.cox_linear_predictor = self._cox.predict_log_partial_hazard(df)
report.cox_2yr_event_prob   = round(prob, 4)
```

So the “number” you see in the dashboard is **the patient‑specific 18-month probability
of the composite event** (HF rehospitalisation or CV death — same outcome the Cox
model was trained on).

#### 3.1.4 Probability → ordinal risk label

```python
PROB_THRESHOLDS = [
    (0.2314, "CRITICAL"),
    (0.1205, "HIGH"),
    (0.0779, "MODERATE"),
    (0.00, "LOW"),
]
```

The first threshold the probability beats wins. So:

| 18-mo event probability | Cox risk label (`cox_risk_label`) |
| ----------------------:| :-------------------------------- |
| ≥ 23.1 %                 | **CRITICAL** / 极高危              |
| 12.1 – 23.0 %              | **HIGH** / 高危                    |
| 7.8 – 12.0 %              | **MODERATE** / 中危                |
| < 7.8 %                 | **LOW** / 低危                     |

This is the **statistical** severity. By itself it is not the final severity shown to
the clinician — that comes next.

### 3.2 Step 2 — Integrated severity tier (`_integrate_risk`, lines 286–350)

Agent C now has **three independent severity opinions**:

1. `snap.risk_tier`           — Agent A’s flag‑count tier (clinical heuristic).
2. `report.cox_risk_label`    — the Cox 18-mo probability tier (statistical).
3. RA‑function‑mapped tier    — from Agent B’s overall RA grade:

   ```
   Normal               → LOW
   Mildly impaired      → MODERATE
   Moderately impaired  → HIGH
   Severely impaired    → CRITICAL
   Severely impaired (RA myopathy) → CRITICAL
   ```

It then **takes the most severe** of the three (using the ranking
`LOW=0 < MODERATE=1 < HIGH=2 < CRITICAL=3`):

```python
report.integrated_risk_tier = max(valid_tiers, key=lambda t: tier_rank[t])
```

The reasoning is intentionally conservative: a patient flagged HIGH by any single
pathway is treated as HIGH overall. The picked tier is what populates
**`▌综合风险等级 / Integrated risk tier`** in the report.

While building this tier the function also compiles a `risk_drivers` list (in EN and
ZH) — the human‑readable reasons (e.g. *“Cox model 18-mo event probability: 47 %
(HIGH)”*, *“CKD with impaired RA function — cardiorenal syndrome risk”*) that are
printed under *Key risk drivers* in the summary.

### 3.3 Step 3 — Clinical advice (`_apply_decision_rules`, lines 352–373)

Severity is now decided; **what to do about it** is a separate rule engine.
`DECISION_RULES` is an ordered list of 8 lambdas — each fires independently
and adds a recommendation with a priority (1 = most urgent, 4 = baseline).

| Priority | Action code             | Condition (predicate on `snapshot`/`ra_report`)                                                            | What it recommends (EN, abbreviated)                                                                  |
| -------: | ----------------------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
|        1 | `URGENT_CARDIOLOGY`     | `overall_ra_function == "Severely impaired (RA myopathy)"`                                                 | Urgent cardiology referral; cardiac MRI; specialist review within 2 weeks                             |
|        1 | `MDT_REVIEW`            | `risk_tier == CRITICAL` AND RA function is `Severely impaired*`                                            | Multidisciplinary team (MDT) review                                                                   |
|        2 | `CARDIORENAL_MANAGEMENT`| CKD flag AND `ra_composite_score < 50`                                                                     | Careful diuresis, avoid nephrotoxins, nephrology co‑management, SGLT2i if eGFR ≥ 20 (EMPEROR / DAPA)  |
|        2 | `AF_RHYTHM_REVIEW`      | AF = 1 AND RA function is Moderately / Severely impaired                                                   | Rhythm‑control review, re‑audit anticoagulation                                                       |
|        3 | `DIURETIC_OPTIMISATION` | `ntprobnp > 2000`                                                                                          | Loop‑diuretic optimisation                                                                            |
|        3 | `METABOLIC_INTERVENTION`| `tyg_index > 9.5`                                                                                          | Lifestyle, SGLT2i for dual metabolic + cardiorenal benefit, endocrinology if T2DM uncontrolled       |
|        3 | `PULM_HTN_WORKUP`       | `pasp > 50`                                                                                                | Right‑heart catheterisation, pulmonology/cardiology review                                            |
|        4 | `ROUTINE_FOLLOWUP`      | always `True` — baseline                                                                                   | 3‑monthly review, repeat echo at 12 m, optimise RAAS + SGLT2i per guideline                            |

All firing rules are stored in `report.all_actions`; the list is sorted by priority,
and the first one becomes `report.primary_action`.

The full Chinese text for each action lives in `DECISION_RULES_ZH` in
`translations.py` (lines 425–458). For example:

```
URGENT_CARDIOLOGY  → "紧急心脏科转诊 — 确认右心房心肌病。建议 2 周内完成心脏核磁共振…"
CARDIORENAL_MANAGEMENT → "心肾综合征管理 — CKD 合并 RA 功能受损，存在心肾综合征风险…"
```

This is **why a low Cox number can still produce a long action list** — the rules
are largely orthogonal to the Cox model. They look at the *raw clinical facts*
(NT‑proBNP, TyG, PASP, RA myopathy flag, etc.), not at the tier label.

### 3.4 Step 4 — Natural‑language report (`_write_summary`, lines 375–522)

`clinical_summary` (English) and `summary_zh` (Chinese) are templated text blocks
that compose:

* Demographics (`{age}-year-old {male|female}`, NYHA class, comorbidities)
* The integrated tier and Cox 18-mo probability (plus *imputed‑feature note* if any)
* Agent B’s three RA phases with grade, composite score, AF / QC notes
* Key labs and echo (eGFR, creatinine, NT‑proBNP, LVEF, LAVI, E/e′, PASP, RVEF, TAPSE)
* `risk_drivers` bullet list
* Top 4 actions, in priority order, in the requested language

> **Past bug, worth remembering.** Sex is normalised against `("male", "m", "男", "1")`
> in *both* the EN and ZH branches so the report never silently shows the wrong sex
> or prints NYHA class in place of age.

### 3.5 Step 5 — LLM clinical reasoning (`_llm_reasoning`, lines 656–747)

If a `LLMClient` is passed in, Agent C makes a structured call asking for:

```jsonc
{
  "mechanism":          "150–200 words: pathophysiological chain for this patient",
  "clinical_priority":  "60–100 words: the single most important decision and WHY",
  "watchpoints":        ["specific monitor 1", "…2", "…3"],
  "patient_note":       "50–75 words plain language",
  "model_caveat":       "40–60 words on AI limits FOR THIS PATIENT"
}
```

Key system‑prompt rules (verbatim):

* **Do NOT restate the numbers** — they’re already in the dashboard.
* Synthesis only; be *specific to this patient*.
* If cardiorenal syndrome present → explicitly assess SGLT2i eligibility
  (eGFR ≥ 20 per EMPEROR‑Reduced / DAPA‑HF).
* Elevated TyG → interpret as insulin resistance *amplifying* cardiorenal risk,
  not an incidental finding.

The patient data block sent to the LLM is built by `_build_data_block`
(lines 580–653) and is a compact, sectioned dump (PATIENT / RENAL / BIOMARKERS /
ECHO / RA IMAGING / RISK MODEL / RULE‑BASED ACTIONS).

The reply is parsed by `_parse_llm_json` (lines 524–578), which is tolerant of
markdown fences, trailing commas, and prose around the JSON object. If parsing
fails, the raw text is preserved in `report.llm_reasoning` and
`llm_structured = False` so the UI can fall back.

Output fields populated:
`llm_mechanism`, `llm_clinical_priority`, `llm_watchpoints`,
`llm_patient_note`, `llm_model_caveat`, `llm_model_used`, `llm_provider`,
`llm_tokens_used`.

### 3.6 Step 6 — LLM contradiction / safety check (`_llm_contradiction_check`, lines 749–841)

A second, independent LLM call audits the rule‑based action list against this
specific patient’s data. It is asked to look for:

* **SGLT2i**: contraindicated if eGFR < 20; caution 20–45.
* **Loop diuretics**: AKI risk if CKD 4–5 with aggressive diuresis.
* **Rate / rhythm control**: anticoagulation adequacy in AF + impaired RA.
* **Metabolic agents**: hypoglycaemia risk if T2DM on insulin / sulfonylurea.
* **Model discordance**: significant disagreement between the Cox tier and Agent A’s
  clinical tier, with an explanation.

It returns:

```jsonc
{
  "flags": [
    {"action_code": "...", "concern": "1-2 sentences", "severity": "warning|caution|info"}
  ],
  "model_discordance": "explanation or null",
  "overall_safe": true|false,
  "summary": "30-50 word overall safety assessment"
}
```

These populate `llm_contradiction_flags` and `llm_contradiction_summary`.
**Importantly, this layer can override or down‑weight rule‑based recommendations
in the UI without modifying the underlying rules** — it’s a safety net, not a
re‑planner.

---

## 4. Worked example (end‑to‑end)

Patient P_127: 71y female, AF = 1, CKD = 1, NYHA 3, eGFR 38, NT‑proBNP 3 400,
TyG 9.8, PASP 52, RVEF 41, RA reservoir 18 %, RA myopathy = 0.

| Step                                | Result                                                                                          |
| ----------------------------------- | ----------------------------------------------------------------------------------------------- |
| Agent A flags                       | SEVERE_CKD, CRITICAL_NT-proBNP, HIGH_INSULIN_RESISTANCE, SEVERELY_IMPAIRED_RA, PULMONARY_HYPERTENSION, RV_DYSFUNCTION, ATRIAL_FIBRILLATION → 7 flags |
| Agent A `risk_tier`                 | **CRITICAL** (critical_flags present, n ≥ 4)                                                    |
| Agent B composite score             | e.g. ~32 → overall RA function **Severely impaired**                                            |
| Agent C Cox H(t=18)                | computed from feature vector                                                                    |
| Agent C 18-mo probability            | e.g. 0.71 → **CRITICAL** label                                                                  |
| Integrated tier                     | max(CRITICAL, CRITICAL, CRITICAL) = **CRITICAL**                                                |
| Actions firing                      | MDT_REVIEW (p=1), CARDIORENAL_MANAGEMENT (p=2), AF_RHYTHM_REVIEW (p=2), DIURETIC_OPTIMISATION (p=3), METABOLIC_INTERVENTION (p=3), PULM_HTN_WORKUP (p=3), ROUTINE_FOLLOWUP (p=4) |
| `primary_action`                    | `MDT_REVIEW`                                                                                    |
| LLM reasoning (typical)             | Mechanistic link between AF‑driven loss of atrial kick, RA myopathy‑like remodelling, congestion, and eGFR decline; **clinical priority** = optimise volume status before considering rhythm control; **patient_note** in plain language. |
| LLM safety check (typical)          | `warning` on `METABOLIC_INTERVENTION` if patient on insulin (hypoglycaemia); `caution` on `DIURETIC_OPTIMISATION` (AKI risk at eGFR 38 with aggressive loops). |

Contrast with patient P_004: 62y male, no AF/CKD/DM, NT‑proBNP 250, eGFR 88,
RA reservoir 38 %, RA myopathy 0 → 0 flags, Cox prob ≈ 5 %, integrated tier **LOW**,
only `ROUTINE_FOLLOWUP` fires.

---

## 5. Why the same Cox number can produce different advice

Two patients can share the same `cox_2yr_event_prob` (say 0.35, “MODERATE”) and
still receive very different reports, because:

* The **integrated tier** also looks at Agent A flag count and Agent B’s RA function —
  one of them can escalate the tier above MODERATE.
* The **rule engine** reads the *raw* labs, not the probability. A patient at 35 %
  with NT‑proBNP 2 500 still triggers `DIURETIC_OPTIMISATION`; a patient at 35 % with
  NT‑proBNP 600 does not.
* The **LLM layer** personalises the narrative to *this* patient (mechanism, plain‑
  language note, AI caveat).
* The **safety LLM** can warn or down‑rank an action if a contraindication exists
  even though the rule fired.

So the agent does *not* answer “severity = f(Cox prob)”. It answers:

> *severity = max(clinical heuristic, statistical model, imaging grade); advice =
> rules over raw clinical facts, audited by a safety LLM and explained by a
> reasoning LLM.*

---

## 6. Where to look in the code (quick map)

| Concern                                      | File · function                                                  | Lines              |
| -------------------------------------------- | ---------------------------------------------------------------- | ------------------ |
| Clinical thresholds & flag generation        | `agent_a_data_harvester.py` · `_flag_risks`, `THRESHOLDS`        | 25–36, 319–359     |
| Agent A tier rule                            | `agent_a_data_harvester.py` · `_assign_risk_tier`                | 362–373            |
| RA grading & composite score                 | `agent_b_imaging_analyst.py` · `RA_GRADES`, `_derive_metrics`    | 18–48, 200–222     |
| Overall RA function mapping                  | `agent_b_imaging_analyst.py` · `_overall_grade`                  | 244–262            |
| Cox features, prediction, imputation         | `agent_c_risk_synthesizer.py` · `_run_cox`                       | 38–49, 226–284     |
| Probability → label thresholds               | `agent_c_risk_synthesizer.py` · `PROB_THRESHOLDS`                | 44–49              |
| Integrated tier (max of three opinions)      | `agent_c_risk_synthesizer.py` · `_integrate_risk`                | 286–350            |
| Decision rules (8 lambdas + priorities)      | `agent_c_risk_synthesizer.py` · `DECISION_RULES`                 | 51–125             |
| Chinese action text                          | `translations.py` · `DECISION_RULES_ZH`                          | 425–458            |
| Patient data block sent to the LLM           | `agent_c_risk_synthesizer.py` · `_build_data_block`              | 580–653            |
| LLM reasoning call (5‑section JSON)          | `agent_c_risk_synthesizer.py` · `_llm_reasoning`                 | 656–747            |
| LLM safety / contradiction check             | `agent_c_risk_synthesizer.py` · `_llm_contradiction_check`       | 749–841            |
| Tolerant JSON parser for LLM replies         | `agent_c_risk_synthesizer.py` · `_parse_llm_json`                | 524–578            |
| Provider registry (12 LLM providers)         | `llm_client.py` · `PROVIDERS`                                    | 38 + …             |
| Reasoning‑model parameter fallbacks          | `llm_client.py` · `_safe_create`                                 | 512–537            |

---

## 7. Caveats & known limits

1. **Median imputation can be aggressive.** A patient missing several Cox features
   gets a probability close to the cohort baseline. `report.missing_cox_features`
   should always be inspected before quoting the probability.
2. **Decision rules are simple lambdas with hard thresholds** (NT‑proBNP > 2000,
   TyG > 9.5, PASP > 50, etc.). They don’t account for trend, dose, or recent change.
3. **Sex encoding** treats anything not in `("male","m","男","1")` as female. Audit
   if you ingest non‑standard EHR fields.
4. **LLM reasoning is advisory.** The contradiction check catches the obvious
   contraindications listed in §3.6 — it cannot replace clinical review.
5. **`.care_agent_settings.json` stores the OpenRouter API key in plaintext.**
   Treat the file as a secret; don’t commit it.
6. **Simulation only.** Every report ends with “Simulation Study · Not for Clinical
   Use” / “由 CARE-Agent 生成 | 本报告为模拟研究…” — keep it.

---

## 8. 一页中文速览（Chinese one‑pager）

* **Agent A**：从 Excel/EHR 抽取 11 项 Cox 特征 + 其他临床/影像数据，按阈值打"风险标记"，按标记数量给出初版风险层级（LOW/MODERATE/HIGH/CRITICAL，详见 `THRESHOLDS` 与 `_assign_risk_tier`）。
* **Agent B**：对 RA 三相（储存/管道/助推）按阈值分级，按 0.50/0.30/0.20 加权得到 0–100 综合评分，再映射为整体 RA 功能（≥65 正常 / 50–64 轻度 / 35–49 中度 / <35 重度；若 `ra_myopathy=1` 则直接判为"重度（右心房心肌病）"）。
* **Agent C**：
  1. 用 `lifelines` Cox 模型预测 18 个月累积风险 `H`，得到 18 个月事件概率 `P = 1 − e^(−H)`；缺失项用队列中位数填补。
  2. 概率 → 标签：≥23.1% 极高危，≥12.1% 高危，≥7.8% 中危，其余低危。
  3. **综合风险等级 = max(Agent A 层级, Cox 标签, RA 功能映射)**（取最严重者，保守策略）。
  4. **临床建议** 来自 8 条独立规则（按优先级 1–4 排序），不是从概率推导出来的——它们直接读取 NT‑proBNP、TyG、PASP、CKD、AF、RA 心肌病等原始变量。`primary_action` 是优先级最高的那一条。
  5. 若启用 LLM：先做"结构化推理"返回 5 段 JSON（机制 / 临床优先级 / 监测点 / 患者通俗解释 / 模型局限性），再做"矛盾/安全核查"逐条审查推荐行动是否与该患者的指标冲突。
* **同样的 Cox 数字，不同患者得到不同建议**：层级取三路最大值，建议来自原始变量而非概率，因此 NT‑proBNP、TyG、PASP、AF、CKD 等具体值才是决定建议清单的关键。
