"""
CARE-Agent — Agent B: RA Imaging Analyst
==========================================
Receives a PatientSnapshot from Agent A, validates and classifies
right atrial strain measurements, and outputs a structured RA
imaging report for Agent C.

In production: ingests DICOM, runs speckle-tracking segmentation.
In simulation: validates and classifies existing strain measurements.
"""

from dataclasses import dataclass, field
from typing import Optional
from agent_a_data_harvester import PatientSnapshot


# ─── RA function classification thresholds ────────────────────────────────────
RA_GRADES = {
    "reservoir": [
        (35,  float("inf"), "Normal",              "RA储存功能正常"),
        (25,  35,           "Mildly impaired",      "轻度受损"),
        (20,  25,           "Moderately impaired",  "中度受损"),
        (0,   20,           "Severely impaired",    "重度受损"),
    ],
    "conduit": [
        (18,  float("inf"), "Normal",              "正常"),
        (12,  18,           "Mildly impaired",      "轻度受损"),
        (8,   12,           "Moderately impaired",  "中度受损"),
        (0,   8,            "Severely impaired",    "重度受损"),
    ],
    "booster": [
        (14,  float("inf"), "Normal",              "正常"),
        (10,  14,           "Mildly impaired",      "轻度受损"),
        (6,   10,           "Moderately impaired",  "中度受损"),
        (0,   6,            "Severely impaired",    "重度受损"),
    ],
}

# Composite RA function score weights
WEIGHTS = {"reservoir": 0.50, "conduit": 0.30, "booster": 0.20}

# Published normal reference means (Addetia et al. J Am Soc Echocardiogr 2022;35:375-385)
# Systematic review & meta-analysis of healthy volunteers by 2D speckle-tracking
RA_NORMAL_REFS = {
    "reservoir": {"mean": 42.7, "ci_lo": 39.4, "ci_hi": 45.9},
    "conduit":   {"mean": 23.6, "ci_lo": 20.7, "ci_hi": 26.6},
    "booster":   {"mean": 16.1, "ci_lo": 13.6, "ci_hi": 18.6},
}


# ─── RA Imaging Report data structure ─────────────────────────────────────────
@dataclass
class RaImagingReport:
    patient_id:             str
    agent_b_status:         str = "PENDING"   # PASS / WARN / FAIL

    # Raw values (from Agent A)
    ra_reservoir:           Optional[float] = None
    ra_conduit:             Optional[float] = None
    ra_booster:             Optional[float] = None
    ra_myopathy:            Optional[int]   = None

    # Classification
    reservoir_grade:        str  = "Unknown"
    conduit_grade:          str  = "Unknown"
    booster_grade:          str  = "Unknown"
    overall_ra_function:    str  = "Unknown"   # Normal / Mildly / Moderately / Severely impaired

    # Derived metrics
    ra_composite_score:     Optional[float] = None   # 0–100, higher = better
    reservoir_conduit_ratio:Optional[float] = None   # reservoir / conduit
    booster_fraction:       Optional[float] = None   # booster / reservoir

    # Contextual adjustments
    af_adjustment_applied:  bool  = False
    ckd_flag:               bool  = False
    booster_af_note:        str   = ""

    # Validation
    strain_consistency:     str   = "Unknown"   # Consistent / Inconsistent / Cannot check
    consistency_note:       str   = ""

    # Quality flags
    imaging_quality_flags:  list  = field(default_factory=list)

    # Handoff note for Agent C
    agent_b_notes:          str   = ""


# ─── Agent B class ─────────────────────────────────────────────────────────────
class RaImagingAnalystAgent:
    """
    Agent B — validates RA strain measurements, classifies RA function,
    applies contextual adjustments, and outputs an RaImagingReport.
    """

    def analyse(self, snapshot: PatientSnapshot) -> RaImagingReport:
        print(f"\n[Agent B] ── Analysing RA imaging for: {snapshot.patient_id} ──")

        report = RaImagingReport(patient_id=snapshot.patient_id)

        # Step 1: ingest strain values from Agent A snapshot
        self._ingest(report, snapshot)

        # Step 2: validate measurement plausibility
        self._validate(report)

        # Step 3: classify each RA phase
        self._classify(report)

        # Step 4: compute derived imaging metrics
        self._derive_metrics(report)

        # Step 5: apply contextual adjustments (AF, CKD)
        self._contextual_adjustment(report, snapshot)

        # Step 6: determine overall RA function grade
        self._overall_grade(report)

        # Step 7: write handoff note for Agent C
        self._write_note(report, snapshot)

        print(f"[Agent B] RA function: {report.overall_ra_function} | "
              f"Composite score: {report.ra_composite_score:.1f}/100 | "
              f"Status: {report.agent_b_status}")

        return report

    # ── Step 1: Ingest ─────────────────────────────────────────────────────────
    def _ingest(self, report: RaImagingReport, snap: PatientSnapshot):
        report.ra_reservoir = snap.ra_reservoir
        report.ra_conduit   = snap.ra_conduit
        report.ra_booster   = snap.ra_booster
        report.ra_myopathy  = snap.ra_myopathy
        report.ckd_flag     = (snap.ckd == 1)

    # ── Step 2: Validate ───────────────────────────────────────────────────────
    def _validate(self, report: RaImagingReport):
        flags = []

        # Check all three values are present
        if any(v is None for v in [report.ra_reservoir, report.ra_conduit, report.ra_booster]):
            flags.append("MISSING_STRAIN: One or more RA strain phases unavailable — re-acquisition needed")
            report.agent_b_status = "FAIL"
            report.strain_consistency = "Cannot check"
            report.imaging_quality_flags = flags
            return

        r, c, b = report.ra_reservoir, report.ra_conduit, report.ra_booster

        # Physiological range checks
        if not (5 <= r <= 80):
            flags.append(f"OUT_OF_RANGE: ra_reservoir={r:.1f}% (expected 5–80%)")
        if not (0 <= c <= 60):
            flags.append(f"OUT_OF_RANGE: ra_conduit={c:.1f}% (expected 0–60%)")
        if not (0 <= b <= 40):
            flags.append(f"OUT_OF_RANGE: ra_booster={b:.1f}% (expected 0–40%)")

        # Consistency: reservoir ≈ conduit + booster (±20% relative tolerance)
        expected_sum = c + b
        if r > 0:
            deviation = abs(r - expected_sum) / r
            if deviation > 0.25:
                report.strain_consistency = "Inconsistent"
                report.consistency_note = (
                    f"reservoir ({r:.1f}%) ≠ conduit + booster "
                    f"({c:.1f} + {b:.1f} = {expected_sum:.1f}%). "
                    f"Deviation {deviation*100:.0f}% — check image quality."
                )
                flags.append("STRAIN_INCONSISTENCY: " + report.consistency_note)
            else:
                report.strain_consistency = "Consistent"
                report.consistency_note = (
                    f"reservoir ({r:.1f}%) ≈ conduit + booster "
                    f"({expected_sum:.1f}%). Deviation {deviation*100:.0f}%."
                )

        # Booster > reservoir is physiologically impossible
        if b > r:
            flags.append(f"IMPOSSIBLE: booster ({b:.1f}%) > reservoir ({r:.1f}%)")

        report.imaging_quality_flags = flags
        report.agent_b_status = "WARN" if flags else "PASS"

    # ── Step 3: Classify ───────────────────────────────────────────────────────
    def _classify(self, report: RaImagingReport):
        if report.ra_reservoir is None:
            return
        report.reservoir_grade = self._grade(report.ra_reservoir, "reservoir")
        report.conduit_grade   = self._grade(report.ra_conduit,   "conduit")
        report.booster_grade   = self._grade(report.ra_booster,   "booster")

    @staticmethod
    def _grade(value: float, phase: str) -> str:
        for lo, hi, label, _ in RA_GRADES[phase]:
            if lo <= value < hi:
                return label
        return "Unknown"

    # ── Step 4: Derived metrics ────────────────────────────────────────────────
    def _derive_metrics(self, report: RaImagingReport):
        if report.ra_reservoir is None:
            return
        r, c, b = report.ra_reservoir, report.ra_conduit, report.ra_booster

        # Composite score: weighted sum normalised to 0–100
        # Reference maximums: reservoir=60, conduit=40, booster=25
        score = (
            WEIGHTS["reservoir"] * min(r / 60, 1.0) +
            WEIGHTS["conduit"]   * min(c / 40, 1.0) +
            WEIGHTS["booster"]   * min(b / 25, 1.0)
        ) * 100
        report.ra_composite_score = round(score, 1)

        # Reservoir/conduit ratio (reflects passive vs active function balance)
        if c > 0:
            report.reservoir_conduit_ratio = round(r / c, 2)

        # Booster fraction (active contribution to total RA function)
        if r > 0:
            report.booster_fraction = round(b / r * 100, 1)

    # ── Step 5: Contextual adjustment ─────────────────────────────────────────
    def _contextual_adjustment(self, report: RaImagingReport, snap: PatientSnapshot):
        # AF: booster phase unreliable — atria not contracting rhythmically
        if snap.af == 1 and report.ra_booster is not None:
            report.af_adjustment_applied = True
            report.booster_af_note = (
                f"Patient has AF — booster strain ({report.ra_booster:.1f}%) "
                "reflects passive recoil rather than active atrial contraction. "
                "Reservoir and conduit phases are the reliable indicators."
            )

        # CKD: flag that RA impairment is likely volume-mediated
        if report.ckd_flag and report.ra_reservoir is not None:
            if report.ra_reservoir < 30:
                report.imaging_quality_flags.append(
                    "CKD_CONTEXT: Impaired RA reservoir in CKD patient — "
                    "volume overload and uraemic toxin-mediated oxidative stress likely "
                    "driving RA remodelling. SGLT2i should be considered if eGFR ≥20 "
                    "mL/min/1.73m² (EMPEROR-Reduced/DAPA-HF evidence)."
                )

    # ── Step 6: Overall RA function grade ─────────────────────────────────────
    def _overall_grade(self, report: RaImagingReport):
        if report.ra_myopathy == 1:
            report.overall_ra_function = "Severely impaired (RA myopathy)"
            return

        if report.ra_composite_score is None:
            report.overall_ra_function = "Unknown"
            return

        s = report.ra_composite_score
        if s >= 65:
            report.overall_ra_function = "Normal"
        elif s >= 50:
            report.overall_ra_function = "Mildly impaired"
        elif s >= 35:
            report.overall_ra_function = "Moderately impaired"
        else:
            report.overall_ra_function = "Severely impaired"

    # ── Step 7: Handoff note ───────────────────────────────────────────────────
    def _write_note(self, report: RaImagingReport, snap: PatientSnapshot):
        af_note = " (AF — booster unreliable)" if report.af_adjustment_applied else ""
        ckd_note = " in CKD patient" if report.ckd_flag else ""
        qc_note = f" QC: {len(report.imaging_quality_flags)} flag(s)." if report.imaging_quality_flags else " QC: clean."

        # Deviation from published normal means
        def _dev(val, phase):
            if val is None:
                return ""
            diff = val - RA_NORMAL_REFS[phase]["mean"]
            return f" [ref mean {RA_NORMAL_REFS[phase]['mean']}%; Δ{diff:+.1f}%]"

        report.agent_b_notes = (
            f"RA function: {report.overall_ra_function}{ckd_note}. "
            f"Composite score {report.ra_composite_score:.1f}/100. "
            f"Reservoir {report.ra_reservoir:.1f}%{_dev(report.ra_reservoir, 'reservoir')} "
            f"({report.reservoir_grade}){af_note}, "
            f"conduit {report.ra_conduit:.1f}%{_dev(report.ra_conduit, 'conduit')} "
            f"({report.conduit_grade}), "
            f"booster {report.ra_booster:.1f}%{_dev(report.ra_booster, 'booster')} "
            f"({report.booster_grade})."
            f"{qc_note} "
            f"Consistency: {report.strain_consistency}. "
            f"Handoff to Agent C for risk synthesis."
        )
