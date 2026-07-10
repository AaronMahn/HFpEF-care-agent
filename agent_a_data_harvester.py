"""
CARE-Agent — Agent A: Data Harvester
=====================================
Simulates autonomous EHR data retrieval for HFpEF cardiorenal patients.
In production: connects to EHR API, lab system, DICOM server.
In simulation: reads from the 395-patient Excel cohort.

Output: Structured JSON patient snapshot passed to Agent B and Agent C.
"""

import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─── Data source (simulates EHR connection) ────────────────────────────────────
# Resolved relative to this file so the project works on any machine.
# The xlsx lives in the data/ subfolder; legacy root location is a fallback.
_HERE_A = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = (
    os.path.join(_HERE_A, "data", "basic_data.xlsx")
    if os.path.exists(os.path.join(_HERE_A, "data", "basic_data.xlsx"))
    else os.path.join(_HERE_A, "basic_data.xlsx")
)

# ─── Clinical thresholds for risk flagging ─────────────────────────────────────
THRESHOLDS = {
    'egfr_severe':        30,    # eGFR < 30 = G4-5 CKD (severe)
    'egfr_moderate':      60,    # eGFR < 60 = CKD confirmed
    'ntprobnp_high':      900,   # NT-proBNP > 900 pg/mL = high burden
    'ntprobnp_critical':  2000,  # NT-proBNP > 2000 = critical
    'tyg_high':           9.5,   # TyG > 9.5 = high insulin resistance
    'pasp_elevated':      35,    # PASP > 35 mmHg = pulmonary hypertension
    'rvef_reduced':       45,    # RVEF < 45% = RV dysfunction
    'ra_reservoir_low':   30,    # RA reservoir < 30% = impaired
    'ra_reservoir_vlow':  20,    # RA reservoir < 20% = severely impaired
    'mitral_ee_high':     14,    # E/e' > 14 = elevated filling pressure
}


# ─── Patient snapshot data structure ───────────────────────────────────────────
@dataclass
class PatientSnapshot:
    # Identifiers
    patient_id:         str
    harvest_timestamp:  str

    # Demographics
    age:                Optional[float] = None
    sex:                Optional[str]   = None   # 'Male' / 'Female'
    bmi:                Optional[float] = None
    bsa:                Optional[float] = None

    # Vital signs
    sbp:                Optional[float] = None
    dbp:                Optional[float] = None

    # Comorbidities (binary flags)
    hypertension:       Optional[int]   = None
    diabetes:           Optional[int]   = None
    ckd:                Optional[int]   = None
    cad:                Optional[int]   = None
    af:                 Optional[int]   = None
    smoking:            Optional[int]   = None

    # Medications
    sglt2i:             Optional[int]   = None
    diuretics:          Optional[int]   = None
    beta_blocker:       Optional[int]   = None
    acei_arb:           Optional[int]   = None
    insulin:            Optional[int]   = None

    # NYHA class
    nyha_class:         Optional[int]   = None

    # Renal function
    egfr:               Optional[float] = None
    creatinine:         Optional[float] = None
    ckd_stage:          Optional[str]   = None   # derived

    # Key biomarkers
    ntprobnp:           Optional[float] = None
    log_ntprobnp:       Optional[float] = None
    fbg:                Optional[float] = None
    tg:                 Optional[float] = None
    tyg_index:          Optional[float] = None   # derived
    tyg_tertile:        Optional[str]   = None

    # Left heart echo
    lvef:               Optional[float] = None
    lavi:               Optional[float] = None
    mitral_ee:          Optional[float] = None
    lvmi:               Optional[float] = None

    # Right heart echo
    pasp:               Optional[float] = None
    tapse:              Optional[float] = None
    rvef:               Optional[float] = None
    rvfwls:             Optional[float] = None

    # RA strain (passed to Agent B for validation)
    ra_reservoir:       Optional[float] = None
    ra_conduit:         Optional[float] = None
    ra_booster:         Optional[float] = None
    ra_myopathy:        Optional[int]   = None

    # Outcomes (known at follow-up)
    followup_months:    Optional[float] = None
    event_all:          Optional[int]   = None
    hf_rehos:           Optional[int]   = None
    cv_death:           Optional[int]   = None

    # ── Agent A outputs ──────────────────────────────────────────────────────
    risk_flags:         list  = field(default_factory=list)
    data_quality:       dict  = field(default_factory=dict)
    risk_tier:          str   = 'UNKNOWN'   # LOW / MODERATE / HIGH / CRITICAL
    agent_a_notes:      str   = ''


# ─── Agent A class ─────────────────────────────────────────────────────────────
class DataHarvesterAgent:
    """
    Agent A — autonomously retrieves, validates, derives, and flags
    patient data. Outputs a PatientSnapshot for Agent B and C.
    """

    def __init__(self, data_path: str = DATA_PATH, extra_df: pd.DataFrame = None):
        print("[Agent A] Initialising — connecting to data source...")
        self._raw = pd.read_excel(data_path, sheet_name='395全量HFpEF数据')
        self._raw = self._raw.dropna(subset=['name'])
        base_count = len(self._raw)

        if extra_df is not None:
            # Merge an uploaded cohort extension. Patient names are NOT unique, so naive
            # name matching can silently collapse several distinct existing patients. We:
            #   1) match on an explicit unique-id column if the upload provides one, else 'name';
            #   2) UPDATE an existing record only when the key matches EXACTLY ONE existing row;
            #   3) APPEND (never overwrite) when the key is new OR ambiguous (matches >1 row).
            up = extra_df.dropna(subset=['name']).copy()
            key = next((c for c in ('patient_id', 'id', 'uid')
                        if c in up.columns and c in self._raw.columns), 'name')
            counts = self._raw[key].value_counts()

            def _merge_action(v):
                c = int(counts.get(v, 0))
                return 'new' if c == 0 else ('update' if c == 1 else 'ambiguous')

            up['__act'] = up[key].map(_merge_action)
            updates = up[up['__act'] == 'update'].drop(columns='__act')
            appends = up[up['__act'].isin(('new', 'ambiguous'))].drop(columns='__act')
            n_amb   = int((up['__act'] == 'ambiguous').sum())

            # Updates: replace only the uniquely-matched existing rows.
            if len(updates) > 0:
                self._raw = self._raw[~self._raw[key].isin(updates[key])]
                self._raw = pd.concat([self._raw, updates], ignore_index=True)
            # New (and ambiguous-as-new) patients: append; _pid is regenerated below.
            if len(appends) > 0:
                self._raw = pd.concat([self._raw, appends], ignore_index=True)

            print(
                f"[Agent A] Merged dataset (key='{key}'): {base_count} built-in + "
                f"{len(appends)} new + {len(updates)} updated = {len(self._raw)} total patients."
            )
            if n_amb:
                print(
                    f"[Agent A] WARNING: {n_amb} uploaded row(s) matched a non-unique "
                    f"'{key}' and were added as NEW patients (existing records left untouched). "
                    f"Add a unique 'patient_id' column to your upload to update those instead."
                )
        else:
            print(f"[Agent A] Data source loaded: {len(self._raw)} patients available.")

        # Names are NOT unique (~9 repeat in the cohort); assign a unique patient_id
        # so every row is processed once and downstream keys do not collide.
        self._raw = self._raw.reset_index(drop=True)
        _seen, _pids = {}, []
        for _nm in self._raw["name"].astype(str):
            _seen[_nm] = _seen.get(_nm, 0) + 1
            _pids.append(_nm if _seen[_nm] == 1 else f"{_nm} ({_seen[_nm]})")
        self._raw["_pid"] = _pids
        _dups = sum(1 for v in _seen.values() if v > 1)
        if _dups:
            print(f"[Agent A] Disambiguated {_dups} repeated name(s) into unique patient IDs.")

    # ── Main harvest method ─────────────────────────────────────────────────
    def harvest(self, patient_id: str) -> PatientSnapshot:
        """
        Retrieve, validate, and structure all data for a single patient.
        Returns a PatientSnapshot ready for Agent B and C.
        """
        print(f"\n[Agent A] ── Harvesting data for: {patient_id} ──")

        # Step 1: retrieve raw record
        row = self._retrieve(patient_id)
        if row is None:
            raise ValueError(f"Patient '{patient_id}' not found in data source.")

        # Step 2: build snapshot skeleton
        snap = PatientSnapshot(
            patient_id        = patient_id,
            harvest_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        )

        # Step 3: populate fields
        self._populate_demographics(snap, row)
        self._populate_comorbidities(snap, row)
        self._populate_medications(snap, row)
        self._populate_labs(snap, row)
        self._populate_echo(snap, row)
        self._populate_outcomes(snap, row)

        # Step 4: derive computed variables
        self._derive_variables(snap, row)

        # Step 5: validate data quality
        self._validate_quality(snap)

        # Step 6: flag clinical risks
        self._flag_risks(snap)

        # Step 7: assign overall risk tier
        self._assign_risk_tier(snap)

        print(f"[Agent A] Snapshot complete — Risk tier: {snap.risk_tier} | Flags: {len(snap.risk_flags)}")
        return snap

    def harvest_all(self) -> list:
        """Harvest all 395 patients. Returns list of PatientSnapshots."""
        snapshots = []
        names = self._raw['_pid'].tolist()
        print(f"\n[Agent A] Batch harvest — {len(names)} patients...")
        for i, name in enumerate(names):
            try:
                snap = self.harvest(name)
                snapshots.append(snap)
            except Exception as e:
                print(f"[Agent A] WARNING: Failed for {name}: {e}")
        print(f"\n[Agent A] Batch complete: {len(snapshots)}/{len(names)} successful.")
        return snapshots

    # ── Population helpers ──────────────────────────────────────────────────
    def _retrieve(self, patient_id):
        matches = self._raw[self._raw['_pid'] == patient_id]
        if matches.empty:
            return None
        return matches.iloc[0]

    def _populate_demographics(self, snap, row):
        snap.age = self._get(row, 'age')
        snap.sex = 'Male' if self._get(row, 'sex') == 1 else 'Female'
        snap.bmi = self._get(row, 'bmi')
        snap.bsa = self._get(row, 'bsa')
        snap.sbp = self._get(row, 'sbp')
        snap.dbp = self._get(row, 'dbp')
        snap.nyha_class = self._get(row, 'nyha_class')

    def _populate_comorbidities(self, snap, row):
        snap.hypertension = self._get(row, 'hypertension')
        snap.diabetes      = self._get(row, 'diabetes')
        snap.ckd           = self._get(row, 'ckd')
        snap.cad           = self._get(row, 'cad')
        snap.af            = self._get(row, 'af')
        snap.smoking       = self._get(row, 'smoking')

    def _populate_medications(self, snap, row):
        snap.sglt2i      = self._get(row, 'SGLT-2i')
        snap.diuretics   = self._get(row, 'diuretics')
        snap.beta_blocker= self._get(row, 'bate_blockers')
        snap.acei_arb    = self._get(row, 'acri_arb')
        snap.insulin     = self._get(row, 'insulin')

    def _populate_labs(self, snap, row):
        snap.egfr         = self._get(row, 'egfr')
        snap.creatinine   = self._get(row, 'scr')
        snap.ntprobnp     = self._get(row, 'ntprobnp')
        snap.log_ntprobnp = self._get(row, 'logntprobnp')
        snap.fbg          = self._get(row, 'fbg')
        snap.tg           = self._get(row, 'tg')
        snap.tyg_index    = self._get(row, 'tyg_index')
        snap.tyg_tertile  = str(self._get(row, 'tyg_tertile')) if self._get(row, 'tyg_tertile') else None

    def _populate_echo(self, snap, row):
        snap.lvef       = self._get(row, 'lvef')
        snap.lavi       = self._get(row, 'lavi')
        snap.mitral_ee  = self._get(row, 'mitral_ee')
        snap.lvmi       = self._get(row, 'lvmi')
        snap.pasp       = self._get(row, 'pasp')
        snap.tapse      = self._get(row, 'tapse')
        snap.rvef       = self._get(row, 'rvef')
        snap.rvfwls     = self._get(row, 'rvfwls')
        snap.ra_reservoir= self._get(row, 'ra_reservoir')
        snap.ra_conduit = self._get(row, 'ra_conduit')
        snap.ra_booster = self._get(row, 'ra_booster')
        snap.ra_myopathy= self._get(row, 'ra_myopathy')

    def _populate_outcomes(self, snap, row):
        snap.followup_months = self._get(row, 'followup_time')
        snap.event_all       = self._get(row, 'event_all')
        snap.hf_rehos        = self._get(row, 'hf_rehos')
        snap.cv_death        = self._get(row, 'cv_death')

    # ── Derived variables ───────────────────────────────────────────────────
    def _derive_variables(self, snap, row):
        # CKD staging from eGFR
        if snap.egfr is not None:
            if snap.egfr >= 90:
                snap.ckd_stage = 'G1 (≥90)'
            elif snap.egfr >= 60:
                snap.ckd_stage = 'G2 (60–89)'
            elif snap.egfr >= 45:
                snap.ckd_stage = 'G3a (45–59)'
            elif snap.egfr >= 30:
                snap.ckd_stage = 'G3b (30–44)'
            elif snap.egfr >= 15:
                snap.ckd_stage = 'G4 (15–29)'
            else:
                snap.ckd_stage = 'G5 (<15)'

        # Recompute TyG if raw values available and stored value missing
        if snap.tyg_index is None and snap.tg is not None and snap.fbg is not None:
            tg_mg  = snap.tg  * 88.57   # mmol/L → mg/dL
            fbg_mg = snap.fbg * 18.02   # mmol/L → mg/dL
            snap.tyg_index = round(np.log(tg_mg * fbg_mg / 2), 4)

    # ── Data quality validation ─────────────────────────────────────────────
    def _validate_quality(self, snap):
        missing, implausible = [], []

        critical = ['egfr', 'ntprobnp', 'lvef', 'ra_reservoir', 'tyg_index']
        for f in critical:
            if getattr(snap, f) is None:
                missing.append(f)

        # Physiological plausibility checks
        if snap.lvef is not None and (snap.lvef < 10 or snap.lvef > 85):
            implausible.append(f'lvef={snap.lvef} (out of range 10–85)')
        if snap.egfr is not None and (snap.egfr < 3 or snap.egfr > 150):
            implausible.append(f'egfr={snap.egfr} (out of range 3–150)')
        if snap.ra_reservoir is not None and (snap.ra_reservoir < 0 or snap.ra_reservoir > 80):
            implausible.append(f'ra_reservoir={snap.ra_reservoir} (out of range 0–80)')

        snap.data_quality = {
            'status':      'PASS' if not missing and not implausible else 'WARN',
            'missing_critical': missing,
            'implausible':  implausible,
            'completeness': f"{100*(1 - len(missing)/5):.0f}%"
        }

    # ── Clinical risk flagging ──────────────────────────────────────────────
    def _flag_risks(self, snap):
        flags = []
        T = THRESHOLDS

        if snap.egfr is not None:
            if snap.egfr < T['egfr_severe']:
                flags.append('SEVERE_CKD: eGFR < 30 — high cardiorenal risk')
            elif snap.egfr < T['egfr_moderate']:
                flags.append('CKD_CONFIRMED: eGFR < 60')

        if snap.ntprobnp is not None:
            if snap.ntprobnp > T['ntprobnp_critical']:
                flags.append(f'CRITICAL_NT-proBNP: {snap.ntprobnp:.0f} pg/mL > 2000')
            elif snap.ntprobnp > T['ntprobnp_high']:
                flags.append(f'ELEVATED_NT-proBNP: {snap.ntprobnp:.0f} pg/mL > 900')

        if snap.tyg_index is not None and snap.tyg_index > T['tyg_high']:
            flags.append(f'HIGH_INSULIN_RESISTANCE: TyG = {snap.tyg_index:.2f} > 9.5')

        if snap.ra_reservoir is not None:
            if snap.ra_reservoir < T['ra_reservoir_vlow']:
                flags.append(f'SEVERELY_IMPAIRED_RA: reservoir = {snap.ra_reservoir:.1f}% < 20%')
            elif snap.ra_reservoir < T['ra_reservoir_low']:
                flags.append(f'IMPAIRED_RA_RESERVOIR: {snap.ra_reservoir:.1f}% < 30%')

        if snap.pasp is not None and snap.pasp > T['pasp_elevated']:
            flags.append(f'PULMONARY_HYPERTENSION: PASP = {snap.pasp} mmHg > 35')

        if snap.rvef is not None and snap.rvef < T['rvef_reduced']:
            flags.append(f'RV_DYSFUNCTION: RVEF = {snap.rvef:.1f}% < 45%')

        if snap.mitral_ee is not None and snap.mitral_ee > T['mitral_ee_high']:
            flags.append(f'ELEVATED_FILLING_PRESSURE: E/e\' = {snap.mitral_ee:.1f} > 14')

        if snap.af == 1:
            flags.append('ATRIAL_FIBRILLATION: ra_booster likely suppressed')

        if snap.ra_myopathy == 1:
            flags.append('RA_MYOPATHY_POSITIVE: severe RA structural remodeling')

        snap.risk_flags = flags

    # ── Risk tier assignment ────────────────────────────────────────────────
    def _assign_risk_tier(self, snap):
        n = len(snap.risk_flags)
        critical_flags = [f for f in snap.risk_flags if 'CRITICAL' in f or 'SEVERE' in f or 'MYOPATHY' in f]

        if critical_flags or n >= 4:
            snap.risk_tier = 'CRITICAL'
        elif n >= 3:
            snap.risk_tier = 'HIGH'
        elif n >= 1:
            snap.risk_tier = 'MODERATE'
        else:
            snap.risk_tier = 'LOW'

        # Generate brief natural-language note
        ckd_str  = f"CKD {snap.ckd_stage}" if snap.ckd_stage else "no CKD staging"
        ra_str   = f"RA reservoir {snap.ra_reservoir:.1f}%" if snap.ra_reservoir else "RA data pending"
        nyha_str = f"NYHA {snap.nyha_class}" if snap.nyha_class else ""
        af_str   = " with AF" if snap.af == 1 else ""
        snap.agent_a_notes = (
            f"{snap.age}y {snap.sex}{af_str}, {nyha_str}, {ckd_str} (eGFR {snap.egfr:.1f}), "
            f"NT-proBNP {snap.ntprobnp:.0f} pg/mL, TyG {snap.tyg_index:.2f}, {ra_str}. "
            f"Risk tier: {snap.risk_tier} ({n} flags). "
            f"Handoff to Agent B for RA strain validation."
        )

    # ── Utility ─────────────────────────────────────────────────────────────
    @staticmethod
    def _get(row, col):
        val = row.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return val

    # ── Export ──────────────────────────────────────────────────────────────
    @staticmethod
    def to_json(snap: PatientSnapshot, indent: int = 2) -> str:
        # `default` coerces numpy scalar types (int64/float64) that pandas
        # produces into native Python so json.dumps never chokes on them.
        def _native(o):
            if hasattr(o, "item"):
                return o.item()
            raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
        return json.dumps(asdict(snap), indent=indent, ensure_ascii=False,
                          default=_native)

    @staticmethod
    def to_dataframe(snapshots: list) -> pd.DataFrame:
        return pd.DataFrame([asdict(s) for s in snapshots])


# ══════════════════════════════════════════════════════════════════════════════
# DEMO — run Agent A on 5 representative patients
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    agent_a = DataHarvesterAgent()

    # Pick 5 varied patients
    sample_names = agent_a._raw['_pid'].tolist()[:5]

    snapshots = []
    for name in sample_names:
        snap = agent_a.harvest(name)
        snapshots.append(snap)
        print(f"\n  Agent A Note: {snap.agent_a_notes}")
        print(f"  Flags ({len(snap.risk_flags)}): {snap.risk_flags}")

    # Save sample outputs into examples/ so they don't clutter the project root.
    _here = os.path.dirname(os.path.abspath(__file__))
    _examples = os.path.join(_here, 'examples')
    os.makedirs(_examples, exist_ok=True)

    json_out = os.path.join(_examples, 'sample_patient_snapshot.json')
    with open(json_out, 'w', encoding='utf-8') as f:
        f.write(agent_a.to_json(snapshots[0]))
    print(f"\n[Agent A] Sample JSON snapshot saved → {json_out}")

    # Batch all 395
    all_snaps = agent_a.harvest_all()
    df = agent_a.to_dataframe(all_snaps)

    print("\n[Agent A] Risk tier distribution across cohort:")
    print(df['risk_tier'].value_counts())

    csv_out = os.path.join(_examples, 'agent_a_all_snapshots.csv')
    df.to_csv(csv_out, index=False, encoding='utf-8-sig')
    print(f"[Agent A] Full cohort snapshots saved → {csv_out}")
