#!/usr/bin/env python3
"""Regenerate Figure 2: integrated risk tier distribution (overall and by CKD status)
with overlaid crude composite endpoint proportions. Reproduces the published legend."""
import os as _os, sys, io, contextlib, json
_HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
CODE = REPO
OUT = _os.path.join(REPO, "figures")
# The 395-patient clinical dataset is NOT part of this public release (see the
# Data Availability statement in the manuscript). Set CARE_AGENT_DATA to a copy
# of the restricted dataset to reproduce the published figure; otherwise the
# bundled 4-record demo file is used and the output will differ.
REAL = _os.environ.get("CARE_AGENT_DATA", _os.path.join(REPO, "data", "basic_data.xlsx"))
SHEET = _os.environ.get("CARE_AGENT_SHEET", 0)
import pandas as pd, numpy as np
_os.chdir(CODE); sys.path.insert(0, CODE)

LABS = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
df = pd.read_excel(REAL, sheet_name=SHEET)
df.columns = [str(c).strip() for c in df.columns]
ev = pd.to_numeric(df["event_all"], errors="coerce").values
ckd = pd.to_numeric(df["ckd"], errors="coerce").values

with contextlib.redirect_stdout(io.StringIO()):
    from agent_a_data_harvester import DataHarvesterAgent
    from agent_b_imaging_analyst import RaImagingAnalystAgent
    from agent_c_risk_synthesizer import RiskSynthesizerAgent
    a = DataHarvesterAgent(data_path=REAL); b = RaImagingAnalystAgent(); c = RiskSynthesizerAgent()
    snaps = a.harvest_all()
    tier = np.array([c.synthesise(s, b.analyse(s), llm_client=None, lang="en").integrated_risk_tier
                     for s in snaps])
assert len(tier) == len(df), "row/snapshot mismatch"

N, n_ckd, n_non = len(df), int((ckd == 1).sum()), int((ckd == 0).sum())
pct_all = [100*float((tier == l).sum())/N for l in LABS]
pct_non = [100*float(((tier == l) & (ckd == 0)).sum())/n_non for l in LABS]
pct_ckd = [100*float(((tier == l) & (ckd == 1)).sum())/n_ckd for l in LABS]
ev_pct  = [100*float(((tier == l) & (ev == 1)).sum())/max(1, int((tier == l).sum())) for l in LABS]
cnt     = [int((tier == l).sum()) for l in LABS]
evn     = [int(((tier == l) & (ev == 1)).sum()) for l in LABS]

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(9.4, 5.4), dpi=300)
x = np.arange(4); w = 0.26
C = {"all": "#3C6E9F", "non": "#A8C8E4", "ckd": "#C0504D", "ev": "#1a1a1a"}
b1 = ax.bar(x - w, pct_all, w, label=f"Overall (N={N})", color=C["all"], edgecolor="white", linewidth=.6)
b2 = ax.bar(x,     pct_non, w, label=f"Non-CKD (n={n_non})", color=C["non"], edgecolor="white", linewidth=.6)
b3 = ax.bar(x + w, pct_ckd, w, label=f"CKD (n={n_ckd})", color=C["ckd"], edgecolor="white", linewidth=.6)
for bars in (b1, b2, b3):
    for r in bars:
        ax.annotate(f"{r.get_height():.1f}", (r.get_x()+r.get_width()/2, r.get_height()),
                    textcoords="offset points", xytext=(0, 2.5), ha="center",
                    fontsize=7.5, color="#666666")
ax.set_ylabel("Patients within group (%)"); ax.set_xticks(x); ax.set_xticklabels(LABS)
ax.set_xlabel("CARE-Agent integrated risk tier"); ax.set_ylim(0, 88)

ax2 = ax.twinx(); ax2.spines["top"].set_visible(False)
ax2.plot(x, ev_pct, marker="o", ms=7.5, color=C["ev"], lw=1.6, ls="--", zorder=6,
         label="Composite endpoint proportion", clip_on=False)
# per-point label offsets chosen to avoid collision with the bar-value labels
offs = [(0, 52), (-56, 10), (-54, 20), (-60, 10)]
for xi, v, e, n, off in zip(x, ev_pct, evn, cnt, offs):
    ax2.annotate(f"{v:.1f}% ({e}/{n})", (xi, v), textcoords="offset points", xytext=off,
                 ha="center", fontsize=9, fontweight="medium", zorder=7,
                 bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=.9))
ax2.set_ylabel("Composite endpoint proportion (%)"); ax2.set_ylim(0, 110)
ax2.set_yticks([0, 20, 40, 60, 80, 100])

h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc="upper center", ncol=2, frameon=False, fontsize=9.5,
          bbox_to_anchor=(0.42, 1.0))
fig.tight_layout()
OUTPNG = _os.path.join(OUT, "Figure2_integrated_risk_tier.png")
fig.savefig(OUTPNG, bbox_inches="tight"); print("saved", OUTPNG)
print(json.dumps({"pct_all": [round(v,1) for v in pct_all], "pct_nonCKD": [round(v,1) for v in pct_non],
                  "pct_CKD": [round(v,1) for v in pct_ckd], "endpoint_pct": [round(v,1) for v in ev_pct],
                  "n": cnt, "events": evn}, indent=1))
