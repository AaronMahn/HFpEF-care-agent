"""
CARE-Agent — UI Translations
Supports: English (en) · Chinese Simplified (zh)
"""

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        # App-wide
        "app_title":            "CARE-Agent · Patient Risk Dashboard",
        "app_caption":          "HFpEF Cardiorenal Risk Pipeline  ·  Agent A → B → C",
        "page_title":           "🫀 CARE-Agent — Patient Risk Dashboard",
        "simulation_note":      "Simulation Study · Not for Clinical Use",
        "language":             "Language / 语言",

        # Sidebar — dataset
        "sidebar_title":        "CARE-Agent",
        "sidebar_caption":      "Cardiorenal Autonomous Risk Evaluation",
        "dataset_section":      "Dataset",
        "data_source":          "Data source",
        "use_builtin":          "Use built-in dataset",
        "upload_merge":         "Upload & merge with built-in",
        "builtin_caption":      "Using: `basic_data.xlsx` (395 patients, built-in)",
        "merged_caption":       "Built-in + uploaded data merged",
        "requirements":         "📋 Dataset requirements",
        "upload_label":         "Upload Excel file (.xlsx)",
        "download_template":    "⬇️ Download column template",
        "no_file":              "No file uploaded yet.",
        "upload_valid":         "Please upload a valid dataset to continue.",
        "missing_cols":         "Missing required columns",
        "read_error":           "Could not read file",
        "merge_success":        "✓ Merged: {n_new} new + {n_builtin} built-in = {n_total} patients",
        "merge_updated":        "({n_updated} existing patients updated with new data)",
        "merge_ambiguous":      "({n_amb} uploaded row(s) had a non-unique name and were added as NEW patients; include a unique patient_id column to update instead)",

        # Sidebar — LLM
        "ai_reasoning":         "🤖 AI Reasoning (Agent C)",
        "enable_llm":           "Enable LLM clinical reasoning",
        "provider":             "Provider",
        "model":                "Model",
        "model_name":           "Model name",
        "server_url":           "Server URL",
        "save_key":             "💾 Save",
        "clear_key":            "🗑 Clear",
        "key_saved_msg":        "✓ Key saved — persists across sessions",
        "no_key_saved":         "No saved key for this provider",
        "provider_guide":       "📖 Provider guide",

        # Sidebar — patient selector
        "select_patient":       "Select Patient",
        "search_by_name":       "Search by name",
        "select_placeholder":   "— Select a patient —",
        "pipeline_section":     "Pipeline",
        "pipeline_desc":        (
            "**Agent A** — Data Harvester\n"
            "Retrieves data, flags risks, assigns tier.\n\n"
            "**Agent B** — RA Imaging Analyst\n"
            "Classifies RA strain phases, computes composite score.\n\n"
            "**Agent C** — Risk Synthesizer\n"
            "Runs Cox model, integrates outputs, generates recommendations."
        ),

        # Home screen — cohort overview
        "cohort_overview":      "Cohort Overview",
        "total_patients":       "Total Patients",
        "event_rate":           "Composite Event Rate",
        "ckd_prev":             "CKD Prevalence",
        "mean_followup":        "Mean Follow-up",
        "dataset_label":        "Dataset",
        "builtin_label":        "built-in",
        "merged_label":         "merged",
        "how_it_works":         "How the pipeline works",
        "select_patient_hint":  "👈  Select a patient from the sidebar to run the full three-agent pipeline.",

        # Pipeline runner
        "running":              "Running CARE-Agent pipeline for",
        "llm_enabled_note":     "(LLM reasoning enabled)",
        "pipeline_error":       "Pipeline error",
        "init_error":           "Initialisation error",
        "no_patients":          "No patients found in the dataset.",

        # Risk banner
        "risk":                 "RISK",
        "integrated":           "Integrated · Agent A + B + C",
        "primary_action_label": "Primary action",

        # Tabs
        "tab_overview":         "📊 Overview",
        "tab_a":                "🔬 Agent A · Data",
        "tab_b":                "🖼️ Agent B · RA Imaging",
        "tab_c":                "🧠 Agent C · Risk & Actions",

        # Tab — Overview
        "pipeline_summary":     "Pipeline Summary",
        "agent_a_risk_tier":    "Agent A Risk Tier",
        "ra_function":          "RA Function",
        "ra_composite_score":   "RA Composite Score",
        "cox_2yr_prob":         "Cox 18-mo Prob",
        "integrated_risk":      "Integrated Risk",
        "risk_drivers":         "Risk Drivers",
        "agent_a_flags":        "Agent A Flags",
        "recommended_actions":  "Recommended Actions",
        "zh_summary":           "Chinese Summary (中文摘要)",

        # Tab A
        "clinical_risk_flags":  "Clinical Risk Flags",
        "no_flags":             "No clinical risk flags triggered.",
        "data_quality":         "Data Quality",
        "status":               "Status",
        "completeness":         "Completeness",
        "missing_critical":     "Missing critical fields",
        "implausible_vals":     "Implausible values",
        "demographics":         "Demographics & Severity",
        "age":                  "Age",
        "sex":                  "Sex",
        "nyha_class":           "NYHA Class",
        "bmi":                  "BMI",
        "af":                   "AF",
        "renal_function":       "Renal Function",
        "egfr":                 "eGFR",
        "ckd_stage":            "CKD Stage",
        "creatinine":           "Creatinine",
        "biomarkers":           "Key Biomarkers",
        "ra_strain":            "RA Strain (raw → Agent B)",
        "ra_reservoir":         "RA Reservoir",
        "ra_conduit":           "RA Conduit",
        "ra_booster":           "RA Booster",
        "ra_myopathy":          "RA Myopathy",
        "right_heart":          "Right Heart Echo",
        "comorbidities_meds":   "Comorbidities & Medications",
        "comorbidities":        "Comorbidities",
        "medications":          "Medications",
        "left_heart":           "Left Heart Echo Parameters",
        "download_agent_a":     "⬇️ Download Agent A JSON",
        "agent_a_json_caption": "Structured JSON snapshot — handoff payload for Agent B and C.",
        "yes":                  "Yes ⚠️",
        "no":                   "No",
        "positive":             "🔴 Positive",
        "negative":             "✓ Negative",
        "low_delta":            "⚠️ Low",
        "normal_delta":         "✓ Normal",

        # Tab B
        "imaging_qc":           "Imaging QC Status",
        "ra_phase":             "RA Phase Classification",
        "reservoir_phase":      "Reservoir Phase",
        "conduit_phase":        "Conduit Phase",
        "booster_phase":        "Booster Phase",
        "af_booster_note":      "⚠️ AF patient — booster reflects passive recoil, not active contraction",
        "strain_consistency":   "Strain Consistency Check",
        "overall_ra":           "Overall RA Function",
        "derived_metrics":      "Derived Metrics",
        "rc_ratio":             "Reservoir/Conduit Ratio",
        "rc_ratio_help":        "Reservoir ÷ Conduit. Reflects passive vs active function balance.",
        "booster_fraction":     "Booster Fraction",
        "booster_frac_help":    "Booster ÷ Reservoir × 100. Active contribution to total RA function.",

        # Tab C
        "cox_output":           "Cox Model Output",
        "cox_complete":         "✓ All 11 Cox model features present — prediction is complete",
        "cox_imputed":          "{n} feature(s) imputed with cohort medians: {feats}",
        "2yr_prob":             "18-Month Event Probability",
        "2yr_prob_help":        "Predicted probability of composite outcome (death or HF hospitalisation) at 18 months",
        "cox_label":            "Cox Risk Label",
        "integrated_tier":      "Integrated Risk Tier",
        "integrated_note":      "Highest severity across Agent A tier, Cox model, and RA function",
        "decision_rules":       "Clinical Decision Rules",
        "primary_badge":        "PRIMARY ACTION",
        "ai_reasoning_title":   "🤖 AI Clinical Reasoning",
        "llm_enable_hint":      "💡 Enable **AI Reasoning** in the sidebar and enter your API key to get LLM-powered clinical interpretation.",
        "full_report":          "📄 Full Clinical Report (English)",
        "zh_report":            "中文摘要",
        # LLM structured sections
        "llm_mechanism_title":       "⚙️ Pathophysiological Mechanism",
        "llm_priority_title":        "🎯 Clinical Priority",
        "llm_watchpoints_title":     "👁 Watchpoints",
        "llm_patient_note_title":    "🗣 Patient-Friendly Explanation",
        "llm_caveat_title":          "⚠️ Model Limitations",
        "llm_raw_fallback":          "📝 AI Reasoning (raw output)",
        # Contradiction check
        "contradiction_title":       "🔍 Safety & Contradiction Check",
        "contradiction_safe":        "✅ No major contradictions identified.",
        "contradiction_unsafe":      "⚠️ Potential contradictions flagged — review below.",
        "contradiction_summary":     "Summary",
        "contradiction_flag_warn":   "⛔ Warning",
        "contradiction_flag_caution":"⚠️ Caution",
        "contradiction_flag_info":   "ℹ️ Note",
        "contradiction_unavailable": "Contradiction check not available.",
        "download_full":        "⬇️ Download Full Pipeline JSON",
        "full_json_caption":    "Combined JSON output from all three agents.",
        "cox_caption":          "Cox model trained on 395 HFpEF patients (11 predictors incl. RA reservoir). C-index = 0.741. Simulation study — not for clinical use.",
    },

    "zh": {
        # App-wide
        "app_title":            "CARE-Agent · 患者风险评估看板",
        "app_caption":          "HFpEF 心肾风险评估流水线  ·  Agent A → B → C",
        "page_title":           "🫀 CARE-Agent — 患者风险评估看板",
        "simulation_note":      "模拟研究 · 仅供科研参考，不作临床决策依据",
        "language":             "Language / 语言",

        # Sidebar — dataset
        "sidebar_title":        "CARE-Agent",
        "sidebar_caption":      "心肾自主风险评估系统",
        "dataset_section":      "数据集",
        "data_source":          "数据来源",
        "use_builtin":          "使用内置数据集",
        "upload_merge":         "上传并与内置数据合并",
        "builtin_caption":      "当前使用：`basic_data.xlsx`（内置 395 例患者）",
        "merged_caption":       "内置数据集 + 上传数据（已合并）",
        "requirements":         "📋 数据集格式要求",
        "upload_label":         "上传 Excel 文件（.xlsx）",
        "download_template":    "⬇️ 下载列格式模板",
        "no_file":              "尚未上传文件。",
        "upload_valid":         "请上传有效数据集以继续。",
        "missing_cols":         "缺少必要列",
        "read_error":           "无法读取文件",
        "merge_success":        "✓ 合并成功：{n_new} 例新增 + {n_builtin} 例内置 = 共 {n_total} 例",
        "merge_updated":        "（{n_updated} 例已有患者数据已用新数据更新）",
        "merge_ambiguous":      "（{n_amb} 例上传记录姓名不唯一，已作为新患者添加；如需更新请提供唯一 patient_id 列）",

        # Sidebar — LLM
        "ai_reasoning":         "🤖 AI 推理（Agent C）",
        "enable_llm":           "启用 LLM 临床推理",
        "provider":             "服务商",
        "model":                "模型",
        "model_name":           "模型名称",
        "server_url":           "服务器地址",
        "save_key":             "💾 保存",
        "clear_key":            "🗑 清除",
        "key_saved_msg":        "✓ 密钥已保存，页面刷新后仍有效",
        "no_key_saved":         "该服务商暂无已保存密钥",
        "provider_guide":       "📖 服务商说明",

        # Sidebar — patient selector
        "select_patient":       "选择患者",
        "search_by_name":       "按姓名搜索",
        "select_placeholder":   "— 请选择患者 —",
        "pipeline_section":     "流水线说明",
        "pipeline_desc":        (
            "**Agent A** — 数据采集器\n"
            "采集数据、标记风险、分配风险层级。\n\n"
            "**Agent B** — RA 影像分析师\n"
            "分类 RA 应变各期，计算综合评分。\n\n"
            "**Agent C** — 风险综合器\n"
            "运行 Cox 模型，整合所有输出，生成临床建议。"
        ),

        # Home screen — cohort overview
        "cohort_overview":      "队列概览",
        "total_patients":       "总患者数",
        "event_rate":           "复合终点发生率",
        "ckd_prev":             "CKD 患病率",
        "mean_followup":        "平均随访时间",
        "dataset_label":        "数据集",
        "builtin_label":        "内置",
        "merged_label":         "合并",
        "how_it_works":         "流水线工作原理",
        "select_patient_hint":  "👈  请从左侧边栏选择患者，启动完整三智能体评估流水线。",

        # Pipeline runner
        "running":              "正在为患者运行 CARE-Agent 流水线：",
        "llm_enabled_note":     "（已启用 LLM 推理）",
        "pipeline_error":       "流水线错误",
        "init_error":           "初始化错误",
        "no_patients":          "数据集中未找到患者。",

        # Risk banner
        "risk":                 "风险",
        "integrated":           "综合评估 · Agent A + B + C",
        "primary_action_label": "首要建议",

        # Tabs
        "tab_overview":         "📊 总览",
        "tab_a":                "🔬 Agent A · 数据",
        "tab_b":                "🖼️ Agent B · RA 影像",
        "tab_c":                "🧠 Agent C · 风险与建议",

        # Tab — Overview
        "pipeline_summary":     "流水线摘要",
        "agent_a_risk_tier":    "Agent A 风险层级",
        "ra_function":          "RA 功能",
        "ra_composite_score":   "RA 综合评分",
        "cox_2yr_prob":         "Cox 18个月事件概率",
        "integrated_risk":      "综合风险",
        "risk_drivers":         "风险驱动因素",
        "agent_a_flags":        "Agent A 标记",
        "recommended_actions":  "推荐处理意见",
        "zh_summary":           "中文摘要",

        # Tab A
        "clinical_risk_flags":  "临床风险标记",
        "no_flags":             "未触发任何临床风险标记。",
        "data_quality":         "数据质量",
        "status":               "状态",
        "completeness":         "完整性",
        "missing_critical":     "缺失关键字段",
        "implausible_vals":     "异常值",
        "demographics":         "人口学资料与病情严重程度",
        "age":                  "年龄",
        "sex":                  "性别",
        "nyha_class":           "NYHA 分级",
        "bmi":                  "BMI",
        "af":                   "心房颤动",
        "renal_function":       "肾功能",
        "egfr":                 "eGFR",
        "ckd_stage":            "CKD 分期",
        "creatinine":           "肌酐",
        "biomarkers":           "关键生物标志物",
        "ra_strain":            "RA 应变（原始值 → Agent B）",
        "ra_reservoir":         "RA 储存期",
        "ra_conduit":           "RA 管道期",
        "ra_booster":           "RA 助推期",
        "ra_myopathy":          "RA 心肌病",
        "right_heart":          "右心超声参数",
        "comorbidities_meds":   "合并症与用药",
        "comorbidities":        "合并症",
        "medications":          "用药",
        "left_heart":           "左心超声参数",
        "download_agent_a":     "⬇️ 下载 Agent A JSON",
        "agent_a_json_caption": "结构化 JSON 快照 — 传递给 Agent B 和 C 的载荷。",
        "yes":                  "是 ⚠️",
        "no":                   "否",
        "positive":             "🔴 阳性",
        "negative":             "✓ 阴性",
        "low_delta":            "⚠️ 偏低",
        "normal_delta":         "✓ 正常",

        # Tab B
        "imaging_qc":           "影像质量控制状态",
        "ra_phase":             "RA 各期分类",
        "reservoir_phase":      "储存期",
        "conduit_phase":        "管道期",
        "booster_phase":        "助推期",
        "af_booster_note":      "⚠️ 房颤患者 — 助推期反映被动回弹，非主动收缩",
        "strain_consistency":   "应变一致性检验",
        "overall_ra":           "RA 整体功能",
        "derived_metrics":      "衍生指标",
        "rc_ratio":             "储存期/管道期比值",
        "rc_ratio_help":        "储存期 ÷ 管道期，反映被动与主动功能平衡。",
        "booster_fraction":     "助推分数",
        "booster_frac_help":    "助推期 ÷ 储存期 × 100，主动贡献占比。",

        # Tab C
        "cox_output":           "Cox 模型输出",
        "cox_complete":         "✓ 所有 11 个 Cox 模型特征均已获取 — 预测完整",
        "cox_imputed":          "{n} 个特征已用队列中位数填补：{feats}",
        "2yr_prob":             "18 个月事件概率",
        "2yr_prob_help":        "Cox 模型预测的 18 个月内复合终点（死亡或心衰再住院）发生概率",
        "cox_label":            "Cox 风险标签",
        "integrated_tier":      "综合风险层级",
        "integrated_note":      "取 Agent A 层级、Cox 模型和 RA 功能中的最高严重级别",
        "decision_rules":       "临床决策规则",
        "primary_badge":        "首要建议",
        "ai_reasoning_title":   "🤖 AI 临床推理",
        "llm_enable_hint":      "💡 请在左侧边栏启用 **AI 推理** 并输入 API 密钥，以获取本患者的 LLM 临床解读。",
        "full_report":          "📄 完整临床报告（英文）",
        "zh_report":            "中文摘要",
        # LLM structured sections
        "llm_mechanism_title":       "⚙️ 病理生理机制分析",
        "llm_priority_title":        "🎯 临床首要优先事项",
        "llm_watchpoints_title":     "👁 重点监测指标",
        "llm_patient_note_title":    "🗣 患者通俗说明",
        "llm_caveat_title":          "⚠️ 模型局限性提示",
        "llm_raw_fallback":          "📝 AI 推理（原始输出）",
        # Contradiction check
        "contradiction_title":       "🔍 安全性与矛盾检查",
        "contradiction_safe":        "✅ 未发现重大矛盾或禁忌症。",
        "contradiction_unsafe":      "⚠️ 发现潜在矛盾，请核查以下内容。",
        "contradiction_summary":     "总体评估",
        "contradiction_flag_warn":   "⛔ 警告",
        "contradiction_flag_caution":"⚠️ 注意",
        "contradiction_flag_info":   "ℹ️ 提示",
        "contradiction_unavailable": "安全核查结果不可用。",
        "download_full":        "⬇️ 下载完整流水线 JSON",
        "full_json_caption":    "包含三个智能体全部输出的综合 JSON 文件。",
        "cox_caption":          "Cox 模型基于 395 例 HFpEF 患者训练（11 个预测因子含 RA 储存期），C 指数 = 0.741。模拟研究，仅供科研参考。",
    },
}


def get_translator(lang: str = "en"):
    """Return a T(key, **fmt) function for the given language."""
    strings = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    en_strings = TRANSLATIONS["en"]

    def T(key: str, **kwargs) -> str:
        text = strings.get(key, en_strings.get(key, key))
        if kwargs:
            try:
                text = text.format(**kwargs)
            except (KeyError, ValueError):
                pass
        return text

    return T


# ── Clinical content translation tables ───────────────────────────────────────
# Used directly by agent_c_risk_synthesizer.py and app.py

RISK_TIER_ZH: dict[str, str] = {
    "LOW":      "低危",
    "MODERATE": "中危",
    "HIGH":     "高危",
    "CRITICAL": "极高危",
    "Unknown":  "未知",
}

RA_FUNCTION_ZH: dict[str, str] = {
    "Normal":              "正常",
    "Mildly impaired":     "轻度受损",
    "Moderately impaired": "中度受损",
    "Severely impaired":   "重度受损",
    "Severely impaired (RA myopathy)": "重度受损（右心房心肌病）",
    "Unknown":             "未知",
}

RA_GRADE_ZH: dict[str, str] = {
    "Normal":              "正常",
    "Mildly impaired":     "轻度受损",
    "Moderately impaired": "中度受损",
    "Severely impaired":   "重度受损",
    "Unknown":             "未知",
}

# Full Chinese text for each clinical decision rule action code
DECISION_RULES_ZH: dict[str, str] = {
    "URGENT_CARDIOLOGY": (
        "紧急心脏科转诊 — 确认右心房心肌病。建议 2 周内完成心脏核磁共振（CMR）"
        "检查及心脏专科会诊，评估是否存在心肌炎症或浸润性病变。"
    ),
    "MDT_REVIEW": (
        "多学科团队（MDT）会诊 — 极高危风险等级合并重度 RA 功能障碍，"
        "需心脏科、肾内科、影像科协同评估，制定个体化综合管理方案。"
    ),
    "CARDIORENAL_MANAGEMENT": (
        "心肾综合征管理 — CKD 合并 RA 功能受损，存在心肾综合征风险。"
        "谨慎滴定利尿方案，避免肾毒性药物，建议肾内科协同管理及定期监测肾功能。"
    ),
    "AF_RHYTHM_REVIEW": (
        "房颤心律／心率控制方案评估 — 心房颤动合并中重度 RA 功能受损。"
        "考虑节律控制策略（如复律或消融），复查抗凝治疗方案（CHA₂DS₂-VASc评分）。"
    ),
    "DIURETIC_OPTIMISATION": (
        "利尿剂方案优化 — NT-proBNP > 2000 pg/mL，提示持续容量超负荷。"
        "复查袢利尿剂剂量及给药频率，评估容量状态，必要时考虑静脉利尿治疗。"
    ),
    "METABOLIC_INTERVENTION": (
        "代谢干预 — TyG 指数升高，提示胰岛素抵抗。建议强化生活方式干预（饮食、运动），"
        "如合并 T2DM 且血糖控制不佳，考虑内分泌科转诊及优化降糖方案（优先 SGLT-2i）。"
    ),
    "PULM_HTN_WORKUP": (
        "肺动脉高压进一步检查 — PASP > 50 mmHg，考虑毛细血管后性肺动脉高压。"
        "建议右心导管检查明确血流动力学分型；呼吸科／心脏科联合会诊。"
    ),
    "ROUTINE_FOLLOWUP": (
        "HFpEF 常规随访 — 每 3 个月门诊复查（症状、体重、肾功能、NT-proBNP），"
        "12 个月复查超声心动图（含 RA 应变）；依现行指南优化 RAAS 及 SGLT-2i 治疗。"
    ),
}
