"""
CARE-Agent — Aggregate Human Ratings into a Per-Model Leaderboard
=================================================================
Takes the three doctors' returned JSON ratings (produced by the
build_rating_sheets HTML form), un-blinds them with the saved per-doctor
blinding maps, joins with the original LLM comparison run for latency and
token data, and emits a per-model leaderboard.

For each model it computes:
  • Mean ± SD on each of the 6 rubric dimensions, across (doctor × patient)
  • A composite weighted score:
        4 × mean(mechanism_accuracy + risk_driver_id +
                 treatment_rationale + safety_flag_quality + clinical_readability) / 5
        − 1.0 × mean(hallucination)             ← reverse-scored
    (range −SCORE_MAX … 4 × SCORE_MAX; higher = better)
    With the default 0–2 scale, that's range −2 … 8.
  • Latency (mean seconds for reasoning + safety calls per patient)
  • Tokens (mean total tokens per patient)

Plus inter-rater reliability:
  • ICC(2,k) = two-way random effects, average-measures intraclass correlation
    (one ICC per dimension)
  • Mean Spearman ρ between rater pairs (simple sanity check)

Usage
-----
    python aggregate_human_ratings.py \\
        --llm-run comparison_out/llm_run_<ts> \\
        --ratings-dir comparison_out/llm_run_<ts>/ratings/returned \\
        --blinding-dir comparison_out/llm_run_<ts>/ratings/blinding

The ratings directory should contain the JSON files emailed back by the doctors
(any filenames are fine — the doctor name is inside each file).

Output (in <llm-run>/ratings/aggregate_<ts>/):
    leaderboard.csv          — one row per model
    leaderboard.html         — pretty bilingual leaderboard
    long_form_ratings.csv    — one row per (doctor × patient × model × dim)
    irr.json                 — ICC + Spearman summary
"""

from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from statistics import mean, pstdev

import numpy as np


# Must match the rubric in build_rating_sheets.py
RUBRIC = [
    ("mechanism_accuracy",  "Mechanism accuracy",            "机制解释准确性",  False),
    ("risk_driver_id",      "Risk driver identification",    "风险驱动因素识别", False),
    ("treatment_rationale", "Treatment rationale",           "治疗建议合理性",  False),
    ("safety_flag_quality", "Safety-flag quality",           "安全标记质量",    False),
    ("hallucination",       "Hallucination (reverse)",       "幻觉/错误推断",   True),
    ("clinical_readability","Clinical readability",          "临床可读性",      False),
]

VENDOR_DISPLAY = {
    "anthropic":  "Claude",
    "google":     "Gemini",
    "openai":     "GPT",
    "deepseek":   "DeepSeek",
    "moonshotai": "Kimi",
    "qwen":       "Qwen",
}

# Valid rating range — must match SCORE_MAX in build_rating_sheets.py. Used only
# to warn when a returned JSON contains out-of-range scores (e.g. a stale file
# from an older 0–5 scale mixed into the ratings folder).
SCORE_MAX = 2

# ─── Total Mark algorithm (quality + speed + economy) ────────────────────────
# Three named presets; CLI can override individual weights with --w-quality etc.
PRESETS = {
    "research": (0.70, 0.15, 0.15),   # 默认：科研对照，质量最重
    "bedside":  (0.50, 0.40, 0.10),   # 床旁实时，看重响应速度
    "economy":  (0.50, 0.20, 0.30),   # 大规模部署，看重 token 成本
}
LATENCY_REF_DEFAULT = 120.0   # seconds; latency >= this floors speed_score at 0
TOKENS_REF_DEFAULT  = 10000   # tokens; tokens >= this floors economy_score at 0


def total_mark(composite, latency_s, tokens,
               w_quality, w_speed, w_economy,
               latency_ref=LATENCY_REF_DEFAULT,
               tokens_ref=TOKENS_REF_DEFAULT,
               composite_min=-2.0, composite_max=8.0):
    """Compute a normalised 0–1 Total Mark that combines clinical quality,
    response latency, and token cost. Returns (q, s, e, total) — all in [0, 1].

    Steps:
      quality_score   = clamp((composite − min) / (max − min), 0, 1)        # 0–1
      speed_score     = max(0, 1 − latency_s / latency_ref)                 # 0–1
      economy_score   = max(0, 1 − tokens   / tokens_ref)                   # 0–1
      total_mark      = w_q · quality + w_s · speed + w_e · economy         # 0–1
    """
    if composite is None or latency_s is None or tokens is None:
        return None, None, None, None
    rng = max(0.001, composite_max - composite_min)
    q = max(0.0, min(1.0, (composite - composite_min) / rng))
    s = max(0.0, 1.0 - latency_s / latency_ref)
    e = max(0.0, 1.0 - tokens    / tokens_ref)
    total = w_quality * q + w_speed * s + w_economy * e
    return round(q, 3), round(s, 3), round(e, 3), round(total, 3)


def vendor_of(model: str) -> str:
    return VENDOR_DISPLAY.get(model.split("/")[0], model.split("/")[0])


# ─── ICC(2,k) — two-way random effects, average measures ──────────────────────
def icc_2k(matrix: np.ndarray) -> float | None:
    """matrix[i,j] = rating by rater j on subject i (subjects × raters).
    Returns ICC(2,k); None if too few subjects."""
    M = np.asarray(matrix, dtype=float)
    M = M[~np.isnan(M).any(axis=1)]    # drop rows with any missing
    n, k = M.shape
    if n < 2 or k < 2:
        return None
    grand = M.mean()
    row_means = M.mean(axis=1)
    col_means = M.mean(axis=0)
    ss_total = ((M - grand) ** 2).sum()
    ss_rows  = k * ((row_means - grand) ** 2).sum()
    ss_cols  = n * ((col_means - grand) ** 2).sum()
    ss_err   = ss_total - ss_rows - ss_cols
    ms_rows  = ss_rows / (n - 1)
    ms_cols  = ss_cols / (k - 1)
    ms_err   = ss_err  / ((n - 1) * (k - 1))
    if (ms_rows + (ms_cols - ms_err) / n) == 0:
        return None
    icc = (ms_rows - ms_err) / (ms_rows + (ms_cols - ms_err) / n)
    return float(icc)


def mean_spearman(matrix: np.ndarray) -> float | None:
    """Mean Spearman ρ between all pairs of raters (columns)."""
    from itertools import combinations
    M = np.asarray(matrix, dtype=float)
    M = M[~np.isnan(M).any(axis=1)]
    n, k = M.shape
    if n < 3 or k < 2:
        return None

    def rank(a):
        # Average-rank with proper tie handling (equivalent to
        # scipy.stats.rankdata(method='average')). Ties are pervasive on the
        # 0–2 ordinal scale, so averaging them is essential — otherwise tied
        # values get arbitrary ranks and Spearman ρ is distorted.
        a = np.asarray(a, dtype=float)
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        csum = np.cumsum(counts)
        starts = csum - counts
        avg_ranks = (starts + csum + 1) / 2.0   # 1-based average rank per group
        return avg_ranks[inv]

    rs = []
    for i, j in combinations(range(k), 2):
        a, b = M[:, i], M[:, j]
        ra, rb = rank(a), rank(b)
        # Pearson on ranks = Spearman
        if ra.std() == 0 or rb.std() == 0:
            continue
        rs.append(float(np.corrcoef(ra, rb)[0, 1]))
    return float(np.mean(rs)) if rs else None


# ─── Load ─────────────────────────────────────────────────────────────────────
def load_llm_run(llm_run_dir: str) -> dict:
    src = os.path.join(llm_run_dir, "llm_comparison.json")
    if not os.path.exists(src):
        raise FileNotFoundError(f"llm_comparison.json not found in {llm_run_dir}")
    with open(src, "r", encoding="utf-8") as f:
        return json.load(f)


def load_blinding(blinding_dir: str) -> dict[str, dict]:
    """Return {doctor_name -> blinding map dict}."""
    out = {}
    for path in glob.glob(os.path.join(blinding_dir, "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "doctor" not in data:
            print(f"[aggregate] skipping {path} — no 'doctor' field")
            continue
        out[data["doctor"]] = data
    return out


def load_ratings(ratings_dir: str) -> dict[str, dict]:
    """Return {doctor_name -> rating data dict}. Picks the newest file per doctor."""
    out: dict[str, dict] = {}
    files = sorted(glob.glob(os.path.join(ratings_dir, "*.json")),
                   key=os.path.getmtime)
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[aggregate] WARN: cannot read {path}: {e}"); continue
        doc = data.get("doctor")
        if not doc:
            print(f"[aggregate] WARN: no 'doctor' in {path}, skipped"); continue
        out[doc] = data   # newer overwrites older
    return out


# ─── Un-blind ─────────────────────────────────────────────────────────────────
def unblind(ratings: dict[str, dict], blinding: dict[str, dict]) -> list[dict]:
    """Return a list of dicts:
       {doctor, patient_real, patient_display, tier, model, dim, score, comment}"""
    rows = []
    for doctor, rdata in ratings.items():
        bmap = blinding.get(doctor)
        if not bmap:
            print(f"[aggregate] WARN: no blinding map for doctor '{doctor}' — skipped")
            continue
        per_patient = {b["display_pid"]: b for b in bmap.get("per_patient", [])}
        for display_pid, by_letter in (rdata.get("ratings") or {}).items():
            info = per_patient.get(display_pid)
            if not info:
                print(f"[aggregate] WARN: doctor {doctor} rated {display_pid} but no "
                      f"blinding entry — skipped")
                continue
            letter_to_model = info["letter_to_model"]
            for letter, entry in by_letter.items():
                model = letter_to_model.get(letter)
                if not model:
                    continue
                comment = entry.get("comment") or ""
                for dim_key, _en, _zh, _rev in RUBRIC:
                    if dim_key in entry and entry[dim_key] is not None:
                        sc = int(entry[dim_key])
                        if not (0 <= sc <= SCORE_MAX):
                            print(f"[aggregate] WARN: {doctor} {display_pid} {letter} "
                                  f"{dim_key}={sc} is outside 0–{SCORE_MAX} "
                                  f"(stale/incompatible rating file?)")
                        rows.append({
                            "doctor": doctor,
                            "patient_real": info["real_pid"],
                            "patient_display": display_pid,
                            "tier": info["integrated_tier"],
                            "model": model,
                            "vendor": vendor_of(model),
                            "dim": dim_key,
                            "score": sc,
                            "comment": comment,
                        })
    return rows


# ─── Per-model summary + composite ────────────────────────────────────────────
def summarise(rows: list[dict], llm_run: dict,
              w_quality: float = PRESETS["research"][0],
              w_speed:   float = PRESETS["research"][1],
              w_economy: float = PRESETS["research"][2],
              latency_ref: float = LATENCY_REF_DEFAULT,
              tokens_ref:  float = TOKENS_REF_DEFAULT,
              ) -> tuple[list[dict], dict]:
    """Build per-model summary and per-dim IRR. Adds Total Mark = weighted
    blend of clinical composite, response latency and token consumption."""
    # Latency / tokens from the LLM run
    latency = defaultdict(list); tokens = defaultdict(list)
    for p in llm_run.get("patients", []):
        for m in p.get("models", []):
            r = m.get("reasoning", {}); s = m.get("safety", {})
            latency[m["model"]].append((r.get("latency_s") or 0) + (s.get("latency_s") or 0))
            tokens [m["model"]].append((r.get("tokens")     or 0) + (s.get("tokens")    or 0))

    # Per-model per-dim scores
    by_md = defaultdict(list)        # (model, dim) -> [scores]
    by_m_doctor = defaultdict(lambda: defaultdict(dict))   # model -> doctor -> {(pid,dim) -> score}
    for r in rows:
        by_md[(r["model"], r["dim"])].append(r["score"])
        by_m_doctor[r["model"]][r["doctor"]][(r["patient_display"], r["dim"])] = r["score"]

    summary = []
    for model in sorted({r["model"] for r in rows}):
        dim_means = {}; dim_sds = {}
        for dim_key, _en, _zh, _rev in RUBRIC:
            scores = by_md.get((model, dim_key), [])
            dim_means[dim_key] = round(mean(scores), 3) if scores else None
            dim_sds[dim_key]   = round(pstdev(scores), 3) if len(scores) > 1 else None

        # Composite: mean of 5 positive dims (each 0–SCORE_MAX) × 4  −  hallucination
        pos_keys = [k for k, _e, _z, rev in RUBRIC if not rev]
        pos_vals = [dim_means[k] for k in pos_keys if dim_means[k] is not None]
        composite = None
        if pos_vals:
            pos_mean = mean(pos_vals)            # 0–5
            halluc   = dim_means.get("hallucination") or 0
            composite = round(4 * pos_mean - halluc, 3)   # range: -SCORE_MAX … 4*SCORE_MAX  (i.e. −2 … 8 for 0–2 scale)

        lat_mean = round(mean(latency[model]), 2) if latency[model] else None
        tok_mean = round(mean(tokens [model]), 0) if tokens [model] else None
        # Primary Total Mark (using the user-chosen / CLI-overridden weights)
        q_norm, s_norm, e_norm, tmark = total_mark(
            composite, lat_mean, tok_mean,
            w_quality, w_speed, w_economy,
            latency_ref=latency_ref, tokens_ref=tokens_ref,
        )
        # Also compute Total Mark under every named preset so the leaderboard
        # can show all three rankings side-by-side without re-running the script.
        marks_by_preset = {}
        for pname, (pa, pb, pc) in PRESETS.items():
            _q, _s, _e, _t = total_mark(
                composite, lat_mean, tok_mean,
                pa, pb, pc,
                latency_ref=latency_ref, tokens_ref=tokens_ref,
            )
            marks_by_preset[pname] = {"quality": _q, "speed": _s,
                                       "economy": _e, "total": _t}

        summary.append({
            "model":          model,
            "vendor":         vendor_of(model),
            "n_ratings":      len(by_md.get((model, "mechanism_accuracy"), [])),
            "composite":      composite,
            "quality_score":  q_norm,
            "speed_score":    s_norm,
            "economy_score":  e_norm,
            "total_mark":     tmark,
            "marks_by_preset": marks_by_preset,
            "dim_means":      dim_means,
            "dim_sds":        dim_sds,
            "latency_s_mean": lat_mean,
            "tokens_mean":    tok_mean,
        })

    # Sort: highest Total Mark first (with composite as a secondary key for ties).
    summary.sort(key=lambda r: (r["total_mark"] is None,
                                -(r["total_mark"] or -1e9),
                                -(r["composite"] or -1e9)))

    # IRR per dimension across (doctor × subject), where subject = (model, patient).
    # The subject MUST be keyed by the REAL patient id: display ids (P-01…) are
    # re-shuffled per doctor, so the same display id maps to different real
    # patients across doctors. Keying IRR by display id would correlate ratings
    # of DIFFERENT patients and produce meaningless reliability numbers.
    irr = {}
    doctors = sorted({r["doctor"] for r in rows})
    for dim_key, en, zh, _rev in RUBRIC:
        if len(doctors) < 2:
            irr[dim_key] = {"icc_2k": None, "mean_spearman": None,
                            "label_en": en, "label_zh": zh}
            continue
        subjects = sorted({(r["model"], r["patient_real"])
                           for r in rows if r["dim"] == dim_key})
        if not subjects:
            irr[dim_key] = {"icc_2k": None, "mean_spearman": None,
                            "label_en": en, "label_zh": zh}
            continue
        M = np.full((len(subjects), len(doctors)), np.nan)
        sub_idx = {s: i for i, s in enumerate(subjects)}
        doc_idx = {d: j for j, d in enumerate(doctors)}
        for r in rows:
            if r["dim"] != dim_key:
                continue
            i = sub_idx.get((r["model"], r["patient_real"]))
            j = doc_idx.get(r["doctor"])
            if i is not None and j is not None:
                M[i, j] = r["score"]
        icc = icc_2k(M)
        rho = mean_spearman(M)
        irr[dim_key] = {
            "icc_2k": round(icc, 3) if icc is not None else None,
            "mean_spearman": round(rho, 3) if rho is not None else None,
            "label_en": en, "label_zh": zh,
            "n_subjects": len(subjects), "n_raters": len(doctors),
        }
    return summary, irr


# ─── Writers ──────────────────────────────────────────────────────────────────
def write_long_csv(rows: list[dict], path: str):
    cols = ["doctor", "patient_display", "patient_real", "tier",
            "model", "vendor", "dim", "score", "comment"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])


def write_leaderboard_csv(summary: list[dict], path: str):
    base = ["rank", "model", "vendor", "n_ratings",
            "total_mark", "quality_score", "speed_score", "economy_score",
            "composite", "latency_s_mean", "tokens_mean"]
    dim_cols = []
    for k, _e, _z, _r in RUBRIC:
        dim_cols += [f"{k}_mean", f"{k}_sd"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(base + dim_cols)
        for i, s in enumerate(summary, 1):
            row = [i, s["model"], s["vendor"], s["n_ratings"],
                   s["total_mark"], s["quality_score"], s["speed_score"], s["economy_score"],
                   s["composite"], s["latency_s_mean"], s["tokens_mean"]]
            for k, _e, _z, _r in RUBRIC:
                row += [s["dim_means"].get(k), s["dim_sds"].get(k)]
            w.writerow(row)


def _bar(value: float | None, max_val: float = 2.0, reverse: bool = False) -> str:
    """Render a 0-100% mini bar. When reverse=True (e.g. for hallucination),
    INVERT the width so 'longer bar = better' is consistent across every
    dimension. The raw value displayed alongside is NOT inverted; only the
    visual fill is. A small badge class identifies reverse-scored cells so
    the CSS can colour them differently."""
    if value is None:
        return '<div class="bar"><div class="fill" style="width:0"></div></div>'
    bar_value = (max_val - value) if reverse else value
    pct = max(0, min(100, 100 * bar_value / max_val))
    cls = "bar reverse" if reverse else "bar"
    return f'<div class="{cls}"><div class="fill" style="width:{pct:.1f}%"></div></div>'


def write_leaderboard_html(summary: list[dict], irr: dict, meta: dict, path: str):
    # Vendor colour for the dot
    VC = {"Claude": "#d97706", "Gemini": "#2563eb", "GPT": "#16a34a",
          "DeepSeek": "#7c3aed", "Kimi": "#db2777", "Qwen": "#0891b2"}

    # One <th> per dimension, with both EN and ZH labels inside so the language
    # toggle just flips which span is visible. Crucially this keeps the header
    # column count equal to the body column count (12 = 12); the previous
    # implementation nested <th> inside <th>, which browsers auto-close, leaving
    # the right-most "Clinical readability" column with no body data → blank.
    # Reverse-scored dimensions get a 🔁 marker so the bar direction (longer =
    # better for ALL columns, including reverse) is unambiguous.
    rubric_th_parts = []
    for k, en, zh, reverse in RUBRIC:
        marker = " 🔁" if reverse else ""
        rubric_th_parts.append(
            f'<th title="{html.escape(k)}">'
            f'<span class="lang-en">{html.escape(en)}{marker}</span>'
            f'<span class="lang-zh">{html.escape(zh)}{marker}</span></th>'
        )
    rubric_th = "".join(rubric_th_parts)

    def _fmt_mean(v):
        return "—" if v is None else f"{v:.2f}"
    def _fmt_sd(v):
        return "" if v is None else f"±{v:.2f}"

    rows = []
    for i, s in enumerate(summary, 1):
        dot = VC.get(s["vendor"], "#64748b")
        cell_parts = []
        for k, _e, _z, reverse in RUBRIC:
            mv = s["dim_means"].get(k)
            sv = s["dim_sds"].get(k)
            # The raw value stays honest (0.00 means doctor gave 0); only the
            # bar is inverted for reverse-scored dims so "longer = better".
            badge = ' <span class="rev-badge" title="reverse-scored">🔁</span>' if reverse else ''
            cell_parts.append(
                f'<td><div class="cellv">{_fmt_mean(mv)}{badge}'
                f'<span class="sd">{_fmt_sd(sv)}</span></div>'
                f'{_bar(mv, reverse=reverse)}</td>'
            )
        cells_en = "".join(cell_parts)
        comp = s["composite"]
        comp_s = f"{comp:.2f}" if isinstance(comp, (int, float)) else "—"
        tmark = s["total_mark"]
        tmark_s = f"<b>{tmark:.3f}</b>" if isinstance(tmark, (int, float)) else "<b>—</b>"
        # Sub-scores tooltip: show the three normalised components
        q = s.get("quality_score"); sp = s.get("speed_score"); ec = s.get("economy_score")
        tip = (f"quality {q:.2f}  ·  speed {sp:.2f}  ·  economy {ec:.2f}"
               if None not in (q, sp, ec) else "")
        rows.append(
            f'<tr>'
            f'<td class="rank">#{i}</td>'
            f'<td><span class="vd" style="background:{dot}"></span>'
            f'<span class="vendor">{html.escape(s["vendor"])}</span>'
            f'<div class="model"><code>{html.escape(s["model"])}</code></div></td>'
            f'<td class="num" title="{html.escape(tip)}">{tmark_s}</td>'
            f'<td class="num">{comp_s}</td>'
            f'<td class="num">{s["latency_s_mean"] if s["latency_s_mean"] is not None else "—"}s</td>'
            f'<td class="num">{int(s["tokens_mean"]) if s["tokens_mean"] is not None else "—"}</td>'
            f'<td class="num">{s["n_ratings"]}</td>'
            f'{cells_en}'
            f'</tr>'
        )

    # IRR table
    irr_rows = []
    for k, _e, _z, _r in RUBRIC:
        info = irr.get(k, {})
        irr_rows.append(
            f'<tr><td><span class="lang-en">{html.escape(info.get("label_en", k))}</span>'
            f'<span class="lang-zh">{html.escape(info.get("label_zh", k))}</span></td>'
            f'<td class="num">{info.get("icc_2k") if info.get("icc_2k") is not None else "—"}</td>'
            f'<td class="num">{info.get("mean_spearman") if info.get("mean_spearman") is not None else "—"}</td>'
            f'</tr>'
        )

    # Variables used inside the f-string template below
    weights = meta.get("weights") or {"quality": 0, "speed": 0, "economy": 0}
    PRESET      = html.escape(str(meta.get("preset", "research")))
    ALPHA       = f"{weights.get('quality', 0):.2f}"
    BETA        = f"{weights.get('speed',   0):.2f}"
    GAMMA       = f"{weights.get('economy', 0):.2f}"
    LATENCY_REF = f"{meta.get('latency_ref_s', LATENCY_REF_DEFAULT):.0f}s"
    TOKENS_REF  = f"{int(meta.get('tokens_ref', TOKENS_REF_DEFAULT)):,}"

    # Build the cross-preset comparison table — rank + score for each model
    # under every named preset. Sorted by vendor name so the rows are stable.
    preset_zh = {"research": "研究", "bedside": "床旁", "economy": "经济"}
    PRESET_NAMES = list(PRESETS.keys())   # research, bedside, economy
    # Rank per preset: highest total → rank 1
    ranks = {p: {} for p in PRESET_NAMES}
    for p in PRESET_NAMES:
        ordered = sorted(summary,
                         key=lambda r: -(r["marks_by_preset"].get(p, {}).get("total") or -1e9))
        for i, s in enumerate(ordered, 1):
            ranks[p][s["model"]] = i

    cross_header_cells = "".join(
        f'<th colspan="2" title="{html.escape(p)}">'
        f'<span class="lang-en">{html.escape(p.capitalize())}<br>'
        f'<small style="font-weight:normal;color:var(--muted)">'
        f'α={PRESETS[p][0]:.2f} β={PRESETS[p][1]:.2f} γ={PRESETS[p][2]:.2f}</small></span>'
        f'<span class="lang-zh">{html.escape(preset_zh.get(p, p))}<br>'
        f'<small style="font-weight:normal;color:var(--muted)">'
        f'α={PRESETS[p][0]:.2f} β={PRESETS[p][1]:.2f} γ={PRESETS[p][2]:.2f}</small></span></th>'
        for p in PRESET_NAMES
    )
    cross_subheader = "".join(
        f'<th><span class="lang-en">Rank</span><span class="lang-zh">排名</span></th>'
        f'<th><span class="lang-en">Score</span><span class="lang-zh">分数</span></th>'
        for _ in PRESET_NAMES
    )
    cross_body_rows = []
    # Order by vendor name (stable) so rows are easy to scan top-down
    for s in sorted(summary, key=lambda r: r["vendor"]):
        dot = VC.get(s["vendor"], "#64748b")
        cells = (
            f'<td><span class="vd" style="background:{dot}"></span>'
            f'<span class="vendor">{html.escape(s["vendor"])}</span></td>'
        )
        # Find this model's BEST rank across presets so we can highlight it
        best_rank = min(ranks[p][s["model"]] for p in PRESET_NAMES)
        for p in PRESET_NAMES:
            r = ranks[p][s["model"]]
            t = s["marks_by_preset"].get(p, {}).get("total")
            t_s = f"{t:.3f}" if t is not None else "—"
            # Visual emphasis on rank #1 in any preset
            rank_cls = "rank top" if r == 1 else ("rank top2" if r == best_rank else "rank")
            cells += f'<td class="{rank_cls}">#{r}</td>'
            cells += f'<td class="num">{t_s}</td>'
        cross_body_rows.append(f'<tr>{cells}</tr>')
    cross_body = "".join(cross_body_rows)

    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CARE-Agent · Human-Rater Leaderboard</title>
<style>
  :root {{
    --bg:#fbfaf7; --fg:#1f2330; --muted:#5a6072; --line:#e3e0d8;
    --accent:#7a3e1d; --accent-soft:#f0e6db;
    --font-zh:"PingFang SC","Microsoft YaHei",system-ui,sans-serif;
    --font-en:"Charter","Source Serif Pro",Georgia,serif;
    --font-mono:"JetBrains Mono","SF Mono",ui-monospace,monospace;
  }}
  *{{box-sizing:border-box}}
  html,body{{margin:0;background:var(--bg);color:var(--fg)}}
  body{{font-family:var(--font-en);font-size:14.5px}}
  body.lang-zh{{font-family:var(--font-zh)}}
  .wrap{{max-width:1280px;margin:0 auto;padding:24px 18px 80px}}
  header.bar{{position:sticky;top:0;z-index:10;background:rgba(251,250,247,.96);backdrop-filter:blur(8px);border-bottom:1px solid var(--line);
            display:flex;justify-content:space-between;align-items:center;padding:12px 22px;flex-wrap:wrap;gap:10px}}
  .lang-toggle{{display:inline-flex;border:1px solid var(--line);border-radius:999px;background:#fff;overflow:hidden}}
  .lang-toggle button{{border:0;background:transparent;padding:5px 14px;cursor:pointer;font:inherit;font-size:12.5px;color:var(--muted)}}
  .lang-toggle button.active{{background:var(--accent);color:#fff}}
  h1{{font-size:24px;margin:6px 0 4px}}
  h2{{font-size:17px;margin:24px 0 8px;color:var(--accent)}}
  .chips{{margin:6px 0 12px}}
  .chip{{display:inline-block;background:#fff;border:1px solid var(--line);border-radius:999px;padding:4px 12px;font-size:12px;color:var(--muted);margin-right:6px}}
  table{{border-collapse:collapse;width:100%;font-size:12.5px;background:#fff}}
  th,td{{border:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}}
  th{{background:#f0ece2;font-weight:700;font-size:11.5px}}
  td.num{{text-align:right;font-family:var(--font-mono);white-space:nowrap}}
  td.rank{{font-weight:800;font-family:var(--font-mono);color:var(--accent)}}
  .vd{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;vertical-align:middle}}
  .vendor{{font-weight:700}}
  .model{{font-size:11px;color:var(--muted);margin-top:2px}}
  .cellv{{display:flex;justify-content:space-between;font-family:var(--font-mono);font-size:12.5px}}
  .cellv .sd{{color:var(--muted);font-size:11px;margin-left:4px}}
  .bar{{height:4px;background:#f0ece2;border-radius:3px;margin-top:3px;overflow:hidden}}
  .bar .fill{{height:100%;background:var(--accent);border-radius:3px}}
  /* Reverse-scored dimension: bar is inverted (longer = better) and tinted a
     different colour as a hint that this column's value should be read with
     care (raw value still shows; the bar already shows the "effective good"). */
  .bar.reverse .fill{{background:#8a8a85}}
  .rev-badge{{font-size:10px;margin-left:4px;opacity:.75;cursor:help}}
  /* Cross-preset table rank cells */
  td.rank{{font-family:var(--font-mono);font-weight:700;text-align:center;color:var(--muted)}}
  td.rank.top{{color:var(--accent);background:#fff4e8}}
  td.rank.top2{{background:#e5f3ec;color:#1b6a4a;font-weight:700}}
  footer{{margin-top:30px;padding-top:14px;border-top:1px solid var(--line);color:var(--muted);font-size:11.5px;text-align:center}}
  :not(body).lang-en, :not(body).lang-zh {{ display:none }}
  body.lang-en .lang-en {{ display:block }}
  body.lang-zh .lang-zh {{ display:block }}
  body.lang-en span.lang-en {{ display:inline }}
  body.lang-zh span.lang-zh {{ display:inline }}
  body.lang-en th .lang-en, body.lang-en th span.lang-en {{ display:inline }}
  body.lang-zh th .lang-zh, body.lang-zh th span.lang-zh {{ display:inline }}
</style></head>
<body class="lang-zh">
<header class="bar">
  <div>
    <div style="font-weight:700">
      <span class="lang-en">CARE-Agent · Human-Rater Leaderboard</span>
      <span class="lang-zh">CARE-Agent · 人工评分排行榜</span>
    </div>
    <div style="font-size:12px;color:var(--muted)">{html.escape(meta["timestamp"])}</div>
  </div>
  <div class="lang-toggle">
    <button id="btn-en" type="button">English</button>
    <button id="btn-zh" type="button">中文</button>
  </div>
</header>

<div class="wrap">
  <div class="chips">
    <span class="chip">
      <span class="lang-en">Doctors: {meta["n_doctors"]}</span>
      <span class="lang-zh">医生：{meta["n_doctors"]}</span>
    </span>
    <span class="chip">
      <span class="lang-en">Patients: {meta["n_patients"]}</span>
      <span class="lang-zh">患者：{meta["n_patients"]}</span>
    </span>
    <span class="chip">
      <span class="lang-en">Models: {meta["n_models"]}</span>
      <span class="lang-zh">模型：{meta["n_models"]}</span>
    </span>
    <span class="chip">
      <span class="lang-en">Total ratings: {meta["n_total_ratings"]}</span>
      <span class="lang-zh">评分总数：{meta["n_total_ratings"]}</span>
    </span>
  </div>

  <h2><span class="lang-en">Total Mark leaderboard</span><span class="lang-zh">总分排行榜</span></h2>
  <div style="color:var(--muted);font-size:12.5px;margin-bottom:6px;line-height:1.6">
    <span class="lang-en"><b>Total Mark</b> = α · quality + β · speed + γ · economy, all normalised to [0, 1].
      <code>quality = (composite+2)/10</code>;
      <code>speed = max(0, 1 − latency_s / {LATENCY_REF})</code>;
      <code>economy = max(0, 1 − tokens / {TOKENS_REF})</code>.<br>
      <b>Composite</b> (kept as a column for transparency) = 4 × mean(5 positive dims) − mean(hallucination). Range −2 … 8.<br>
      <b>Preset</b>: <code>{PRESET}</code>  ·  <b>Weights</b>: α (quality) = {ALPHA}, β (speed) = {BETA}, γ (economy) = {GAMMA}</span>
    <span class="lang-zh"><b>总分</b> = α · 质量 + β · 速度 + γ · 经济性，三者均归一化到 [0, 1]。
      <code>质量 = (综合分+2)/10</code>；
      <code>速度 = max(0, 1 − 延迟秒 / {LATENCY_REF})</code>；
      <code>经济 = max(0, 1 − tokens / {TOKENS_REF})</code>。<br>
      <b>综合分</b>（保留以便核查）= 4 × 5 个正向维度均值 − 幻觉均值。范围 −2 … 8。<br>
      <b>预设</b>：<code>{PRESET}</code>  ·  <b>权重</b>：α（质量）= {ALPHA}，β（速度）= {BETA}，γ（经济）= {GAMMA}</span>
  </div>
  <table>
    <thead><tr>
      <th><span class="lang-en">Rank</span><span class="lang-zh">排名</span></th>
      <th><span class="lang-en">Model</span><span class="lang-zh">模型</span></th>
      <th title="weighted blend of quality + speed + economy">
        <span class="lang-en">Total Mark</span><span class="lang-zh">总分</span></th>
      <th><span class="lang-en">Composite</span><span class="lang-zh">综合分</span></th>
      <th><span class="lang-en">Latency</span><span class="lang-zh">延迟</span></th>
      <th><span class="lang-en">Tokens</span><span class="lang-zh">Tokens</span></th>
      <th><span class="lang-en">N</span><span class="lang-zh">样本</span></th>
      {rubric_th}
    </tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>

  <h2><span class="lang-en">Cross-preset ranking</span><span class="lang-zh">三种预设场景下的排名对比</span></h2>
  <div style="color:var(--muted);font-size:12.5px;margin-bottom:6px;line-height:1.6">
    <span class="lang-en">Each preset weights quality/speed/economy differently. <b>Rank #1</b> in any preset is highlighted; <b>green</b> = the preset where this model performs best.</span>
    <span class="lang-zh">三种预设对"质量／速度／成本"的权重不同。任一预设中的 <b>#1</b> 会加深；<b>绿色</b>底色表示这是该模型表现最好的预设。</span>
  </div>
  <table style="max-width:780px">
    <thead>
      <tr><th rowspan="2"><span class="lang-en">Model</span><span class="lang-zh">模型</span></th>{cross_header_cells}</tr>
      <tr>{cross_subheader}</tr>
    </thead>
    <tbody>{cross_body}</tbody>
  </table>

  <h2><span class="lang-en">Inter-rater reliability</span><span class="lang-zh">评估者间一致性</span></h2>
  <div style="color:var(--muted);font-size:12.5px;margin-bottom:6px">
    <span class="lang-en">ICC(2,k) = two-way random, average-measures. &gt;0.75 excellent, 0.6–0.75 good, 0.4–0.6 moderate, &lt;0.4 poor.</span>
    <span class="lang-zh">ICC(2,k) = 两因素随机效应、平均测量。&gt;0.75 极好；0.6–0.75 良好；0.4–0.6 中等；&lt;0.4 较差。</span>
  </div>
  <table style="max-width:680px">
    <thead><tr>
      <th><span class="lang-en">Dimension</span><span class="lang-zh">维度</span></th>
      <th>ICC(2,k)</th>
      <th><span class="lang-en">Mean Spearman ρ</span><span class="lang-zh">平均 Spearman ρ</span></th>
    </tr></thead>
    <tbody>{''.join(irr_rows)}</tbody>
  </table>
  <footer>Generated by aggregate_human_ratings.py · simulation study, not for clinical use.</footer>
</div>
<script>
(function(){{
  var K='careAgentRatingLBLang';
  function apply(l){{
    if(l!=='en') l='zh';
    document.body.classList.toggle('lang-en', l==='en');
    document.body.classList.toggle('lang-zh', l==='zh');
    document.getElementById('btn-en').classList.toggle('active', l==='en');
    document.getElementById('btn-zh').classList.toggle('active', l==='zh');
    document.documentElement.setAttribute('lang', l==='zh'?'zh-CN':'en');
    try{{localStorage.setItem(K, l);}}catch(e){{}}
  }}
  var s=null; try{{s=localStorage.getItem(K);}}catch(e){{}}
  apply(s || 'zh');
  document.getElementById('btn-en').onclick=function(){{apply('en');}};
  document.getElementById('btn-zh').onclick=function(){{apply('zh');}};
}})();
</script>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)


def run(args) -> int:
    llm_run = os.path.abspath(args.llm_run)
    llm = load_llm_run(llm_run)

    bdir = args.blinding_dir or os.path.join(llm_run, "ratings", "blinding")
    blinding = load_blinding(bdir)
    if not blinding:
        print(f"[aggregate] ERROR: no blinding maps found in {bdir}"); return 2

    rdir = args.ratings_dir or os.path.join(llm_run, "ratings", "returned")
    if not os.path.isdir(rdir):
        print(f"[aggregate] ERROR: ratings directory not found: {rdir}")
        print(f"[aggregate]   Drop each doctor's exported JSON in there.")
        return 2
    ratings = load_ratings(rdir)
    if not ratings:
        print(f"[aggregate] ERROR: no ratings JSONs in {rdir}"); return 2

    print(f"[aggregate] LLM run     : {llm_run}")
    print(f"[aggregate] Blinding    : {bdir}  ({len(blinding)} maps)")
    print(f"[aggregate] Ratings dir : {rdir}  ({len(ratings)} doctors: {', '.join(ratings)})")

    rows = unblind(ratings, blinding)
    if not rows:
        print("[aggregate] ERROR: nothing un-blinded successfully — check blinding/ratings filenames")
        return 2

    # Resolve weights: --preset supplies defaults; individual --w-* override.
    α, β, γ = PRESETS.get(args.preset, PRESETS["research"])
    if args.w_quality is not None:  α = args.w_quality
    if args.w_speed   is not None:  β = args.w_speed
    if args.w_economy is not None:  γ = args.w_economy
    # Re-normalise so weights sum to 1 (handles overrides cleanly).
    total_w = α + β + γ
    if total_w <= 0:
        print("[aggregate] ERROR: total weight ≤ 0"); return 2
    α, β, γ = α/total_w, β/total_w, γ/total_w

    summary, irr = summarise(
        rows, llm,
        w_quality=α, w_speed=β, w_economy=γ,
        latency_ref=args.latency_ref, tokens_ref=args.tokens_ref,
    )
    n_patients = len({r["patient_real"] for r in rows})
    n_models   = len({r["model"]        for r in rows})
    meta = {
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_doctors":        len(ratings),
        "n_patients":       n_patients,
        "n_models":         n_models,
        "n_total_ratings":  len(rows) // len(RUBRIC),
        "preset":           args.preset,
        "weights":          {"quality": round(α, 3), "speed": round(β, 3), "economy": round(γ, 3)},
        "latency_ref_s":    args.latency_ref,
        "tokens_ref":       args.tokens_ref,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or os.path.join(llm_run, "ratings", f"aggregate_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    lb_csv  = os.path.join(out_dir, "leaderboard.csv")
    lb_html = os.path.join(out_dir, "leaderboard.html")
    lf_csv  = os.path.join(out_dir, "long_form_ratings.csv")
    irr_jp  = os.path.join(out_dir, "irr.json")

    write_leaderboard_csv (summary, lb_csv)
    write_leaderboard_html(summary, irr, meta, lb_html)
    write_long_csv        (rows, lf_csv)
    with open(irr_jp, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "irr": irr, "summary": summary},
                  f, ensure_ascii=False, indent=2)

    # Console summary
    print(f"\n[aggregate] Wrote:")
    for p in (lb_html, lb_csv, lf_csv, irr_jp):
        print(f"  - {p}")
    print(f"\n[aggregate] Preset: {args.preset}   weights  α(quality)={α:.2f}  β(speed)={β:.2f}  γ(economy)={γ:.2f}")
    print(f"[aggregate] References: latency_ref={args.latency_ref}s  tokens_ref={args.tokens_ref}")
    print(f"[aggregate] Top of leaderboard (sorted by Total Mark):")
    for i, s in enumerate(summary, 1):
        comp = s["composite"]
        tmark = s["total_mark"]
        print(f"  #{i}  {s['vendor']:9s}  {s['model']:40s}  "
              f"total={tmark if tmark is not None else '—':>5}  "
              f"composite={comp if comp is not None else '—':>6}  "
              f"latency={s['latency_s_mean']}s  tokens={s['tokens_mean']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate doctor-rated outputs into a per-model leaderboard.")
    p.add_argument("--llm-run", required=True, dest="llm_run",
                   help="path to the llm_run_<ts>/ folder used to build the rating sheets")
    p.add_argument("--ratings-dir", default=None, dest="ratings_dir",
                   help="folder containing the doctors' exported JSON files "
                        "(default: <llm-run>/ratings/returned)")
    p.add_argument("--blinding-dir", default=None, dest="blinding_dir",
                   help="folder with the per-doctor un-blinding maps "
                        "(default: <llm-run>/ratings/blinding)")
    p.add_argument("--output-dir", default=None, dest="output_dir",
                   help="defaults to <llm-run>/ratings/aggregate_<ts>/")
    # Total Mark weighting — quality + speed + economy
    p.add_argument("--preset", default="research", choices=list(PRESETS.keys()),
                   help="weight preset for Total Mark (default: research)")
    p.add_argument("--w-quality", type=float, default=None, dest="w_quality",
                   help="override α (clinical quality weight)")
    p.add_argument("--w-speed", type=float, default=None, dest="w_speed",
                   help="override β (response latency weight)")
    p.add_argument("--w-economy", type=float, default=None, dest="w_economy",
                   help="override γ (token cost weight)")
    p.add_argument("--latency-ref", type=float, default=LATENCY_REF_DEFAULT, dest="latency_ref",
                   help=f"seconds at which speed_score floors to 0 (default {LATENCY_REF_DEFAULT})")
    p.add_argument("--tokens-ref", type=float, default=TOKENS_REF_DEFAULT, dest="tokens_ref",
                   help=f"tokens at which economy_score floors to 0 (default {TOKENS_REF_DEFAULT})")
    return p


def main():
    args = build_parser().parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n[aggregate] interrupted."); return 130


if __name__ == "__main__":
    sys.exit(main())
