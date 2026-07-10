#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rerun_failed.py — Re-run ONLY the failed cells of an existing LLM-comparison run
and merge the recovered results back into the same file (in place).

A "failed cell" is, per (patient, model):
  • reasoning  — not success, OR success but JSON did not parse (unstructured)
  • safety     — not available (no usable safety output)

Each failed sub-call is retried independently with the hardened LLMClient
(429 back-off, empty/truncation retries, JSON mode). A recovered result only
*overwrites* the old one when it is genuinely better, so a merge can never make
the file worse. Then the JSON, CSV and HTML are rewritten in place (original
JSON is backed up to <name>.bak.json first).

Usage (typically via the launcher):
  python rerun_failed.py                         # newest file in comparison_out
  python rerun_failed.py --in path/to/run.json   # a specific file
  python rerun_failed.py --dry-run               # just list what would re-run
  python rerun_failed.py --mock                  # offline smoke test
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_a_data_harvester import DataHarvesterAgent           # noqa: E402
from agent_b_imaging_analyst import RaImagingAnalystAgent        # noqa: E402
from agent_c_risk_synthesizer import RiskSynthesizerAgent        # noqa: E402
import settings_manager as SM                                    # noqa: E402
from benchmark_llms import make_client                           # noqa: E402
from compare_llm_outputs import write_csv, write_html, write_json  # noqa: E402


# ── failure detection ────────────────────────────────────────────────────────
def _blank(x) -> bool:
    """True if a text field is missing or only whitespace."""
    return not str(x or "").strip()


def reason_failed(r: dict) -> bool:
    # Failed if the call errored or JSON did not parse, OR it "succeeded" but the
    # core mechanism field came back empty (a silent / blank success).
    if not (r.get("success") and r.get("structured")):
        return True
    return _blank(r.get("mechanism"))


def safety_failed(s: dict) -> bool:
    # Failed if unavailable, OR available but with no usable content — no flags AND
    # an empty summary (e.g. a model that returns an empty safety object). A genuine
    # "no contradictions" result keeps a non-empty summary and is NOT failed.
    if not s.get("available"):
        return True
    return (not (s.get("flags") or [])) and _blank(s.get("summary"))


# ── re-run a single sub-call (returns a fresh sub-block dict) ─────────────────
def rerun_reason(agent_c, snap, ra, client, lang: str) -> dict:
    report = agent_c.synthesise(snap, ra, llm_client=None, lang=lang)
    t0 = perf_counter()
    agent_c._llm_reasoning(report, snap, ra, client, lang=lang)
    lat = round(perf_counter() - t0, 3)
    return {
        "success": report.llm_available,
        "structured": report.llm_structured,
        "latency_s": lat,
        "tokens": report.llm_tokens_used,
        "error": report.llm_error or "",
        "mechanism": report.llm_mechanism or "",
        "clinical_priority": report.llm_clinical_priority or "",
        "watchpoints": list(report.llm_watchpoints or []),
        "patient_note": report.llm_patient_note or "",
        "model_caveat": report.llm_model_caveat or "",
        "raw": report.llm_reasoning or "",
    }


def rerun_safety(agent_c, snap, ra, client, lang: str) -> dict:
    report = agent_c.synthesise(snap, ra, llm_client=None, lang=lang)
    t0 = perf_counter()
    agent_c._llm_contradiction_check(report, snap, ra, client, lang=lang)
    lat = round(perf_counter() - t0, 3)
    return {
        "available": report.llm_contradiction_available,
        "latency_s": lat,
        "tokens": report.llm_contradiction_tokens,
        "error": report.llm_contradiction_error or "",
        "flags": list(report.llm_contradiction_flags or []),
        "summary": report.llm_contradiction_summary or "",
    }


# ── "is the new result better?" — guards against regressions on merge ─────────
def reason_better(old: dict, new: dict) -> bool:
    # Fully good (parsed + non-empty mechanism) → always take it.
    if new.get("success") and new.get("structured") and not _blank(new.get("mechanism")):
        return True
    # Otherwise only take it if it at least revived a dead (unsuccessful) cell.
    if new.get("success") and not old.get("success"):
        return True
    return False


def safety_better(old: dict, new: dict) -> bool:
    # Only accept a recovered safety result that actually carries content, so an
    # empty-but-"available" reply can never overwrite or masquerade as a fix.
    return bool(new.get("available")
                and ((new.get("flags") or []) or not _blank(new.get("summary"))))


def run(args) -> int:
    # 1) locate the results file
    path = args.in_path
    if not path:
        # Find the newest REAL comparison run. Supports the new nested layout
        # (llm_run_<ts>/llm_comparison.json) and the legacy flat layout
        # (llm_comparison_<ts>.json at the top of comparison_out/).
        # Mocks are excluded by default — pass --include-mock to opt in.
        cands = (
            glob.glob(os.path.join(args.output_dir, "llm_run_*", "llm_comparison.json")) +
            glob.glob(os.path.join(args.output_dir, "llm_run_*", "llm_comparison_*.json")) +
            glob.glob(os.path.join(args.output_dir, "llm_comparison_*.json"))   # legacy flat
        )
        if args.include_mock:
            cands += glob.glob(os.path.join(args.output_dir, "llm_mock_run_*", "llm_comparison.json"))
        # Filter out rerun-created backups
        cands = [c for c in cands if not c.endswith(".bak.json")]
        if not cands:
            print(
                f"[rerun] No real llm_comparison*.json found under {args.output_dir}/.\n"
                f"        (mocks excluded; pass --include-mock or --in <path> to override.)"
            )
            return 2
        # Sort by the timestamp embedded in the folder/file name so that "newest"
        # reflects when the comparison was generated, not when the file was last
        # touched on disk. The path always contains a ..._YYYYMMDD_HHMMSS... segment.
        ts_re = re.compile(r"(\d{8}_\d{6})")
        def _ts(p):
            m = ts_re.search(p)
            return m.group(1) if m else ""   # untimestamped legacy files sort first
        path = max(cands, key=_ts)
    if not os.path.isfile(path):
        print(f"[rerun] File not found: {path}")
        return 2
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    meta = d["meta"]
    patients = d["patients"]
    models = meta["models"]
    lang = args.lang or meta.get("lang", "en")

    # 2) find failed cells
    tasks = []   # (pidx, midx, kind)
    for pidx, p in enumerate(patients):
        for midx, m in enumerate(p["models"]):
            if reason_failed(m["reasoning"]):
                tasks.append((pidx, midx, "reason"))
            if safety_failed(m["safety"]):
                tasks.append((pidx, midx, "safety"))

    print("=" * 64)
    print(f"  Source     : {path}")
    print(f"  Patients   : {len(patients)}  |  Models: {len(models)}  |  Lang: {lang}")
    print(f"  Failed cells to re-run: {len(tasks)}")
    # breakdown by model
    by_model = {}
    for pidx, midx, kind in tasks:
        mm = patients[pidx]["models"][midx]["model"]
        by_model.setdefault(mm, {"reason": 0, "safety": 0})[kind] += 1
    for mm, c in by_model.items():
        print(f"      - {mm}: reasoning×{c['reason']}  safety×{c['safety']}")
    print("=" * 64)

    if not tasks:
        print("[rerun] Nothing failed — nothing to do. ✅")
        return 0
    if args.dry_run:
        print("[rerun] --dry-run: no calls made.")
        return 0

    # 3) credentials + clients
    api_key = (args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
               or SM.get_api_key(args.provider))
    if not args.mock and not api_key:
        print(f"[rerun] ERROR: no API key for '{args.provider}'. Use --api-key, set "
              f"OPENROUTER_API_KEY, or save one in the app.")
        return 2

    agent_a = DataHarvesterAgent()
    agent_b = RaImagingAnalystAgent()
    agent_c = RiskSynthesizerAgent()
    clients = {m: make_client(args.provider, m, api_key, args.mock,
                              args.timeout, args.max_tokens, args.temperature)
               for m in models}

    # 4) pre-harvest each patient that needs work (local, deterministic)
    need_pids = sorted({pidx for pidx, _, _ in tasks})
    ctx = {}     # pidx -> (snap, ra)
    for pidx in need_pids:
        pid = patients[pidx]["patient_id"]
        try:
            snap = agent_a.harvest(pid)
            ctx[pidx] = (snap, agent_b.analyse(snap))
        except Exception as e:
            print(f"[rerun] WARN: cannot harvest {pid}: {e} (skipping its cells)")

    # 5) run failed cells (throttled concurrency — avoids the 429 storm)
    def _do(task):
        pidx, midx, kind = task
        if pidx not in ctx:
            return task, None
        snap, ra = ctx[pidx]
        client = clients[models[midx]]
        try:
            if kind == "reason":
                return task, rerun_reason(agent_c, snap, ra, client, lang)
            return task, rerun_safety(agent_c, snap, ra, client, lang)
        except Exception as e:
            return task, {"_error": f"harness error: {e}"}

    conc = max(1, int(args.concurrency))
    results = {}
    done = 0
    if conc == 1:
        for t in tasks:
            results.__setitem__(*_do(t)); done += 1
    else:
        with ThreadPoolExecutor(max_workers=min(conc, len(tasks))) as ex:
            for fut in as_completed([ex.submit(_do, t) for t in tasks]):
                t, res = fut.result()
                results[t] = res
                done += 1
                pidx, midx, kind = t
                pid = patients[pidx]["patient_id"]; mm = models[midx]
                tag = "n/a" if res is None else (
                    "ERR" if res.get("_error") else
                    ("ok" if (res.get("structured") or res.get("available")) else "still-fail"))
                print(f"  [{done}/{len(tasks)}] {pid} · {mm} · {kind}: {tag}")

    # 6) merge — only when strictly better
    applied = 0
    for t, res in results.items():
        if not res or res.get("_error"):
            continue
        pidx, midx, kind = t
        cell = patients[pidx]["models"][midx]
        if kind == "reason" and reason_better(cell["reasoning"], res):
            cell["reasoning"] = res; applied += 1
        elif kind == "safety" and safety_better(cell["safety"], res):
            cell["safety"] = res; applied += 1

    # 7) recompute failure tally + annotate meta
    still = sum(1 for p in patients for m in p["models"]
                if reason_failed(m["reasoning"])) + \
            sum(1 for p in patients for m in p["models"]
                if safety_failed(m["safety"]))
    meta.setdefault("rerun_history", []).append({
        "at": time.strftime("%Y%m%d_%H%M%S"),
        "attempted": len(tasks), "applied": applied, "remaining": still,
    })

    if args.mock:
        print(f"[rerun] (mock) would apply {applied}/{len(tasks)} fixes; {still} cells still failing.")
    # 8) write back in place (+ backup), regenerate CSV/HTML
    base = os.path.splitext(path)[0]
    bak = base + ".bak.json"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"[rerun] backed up original → {bak}")
    write_json(patients, meta, path)
    write_csv(patients, base + ".csv")
    write_html(patients, meta, base + ".html")

    print(f"\n[rerun] Applied {applied} fix(es). Cells still failing: {still}.")
    print("[rerun] Updated in place:")
    for ext in (".json", ".csv", ".html"):
        print(f"  - {base + ext}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Re-run only the failed cells of a comparison run and merge in place.")
    p.add_argument("--in", dest="in_path", default="",
                   help="results JSON (default: newest REAL run in --output-dir)")
    p.add_argument("--include-mock", action="store_true", dest="include_mock",
                   help="also consider llm_mock_run_*/ folders when auto-discovering")
    p.add_argument("--output-dir", default="comparison_out", dest="output_dir")
    p.add_argument("--provider", default="openrouter")
    p.add_argument("--api-key", default="", dest="api_key")
    p.add_argument("--lang", default="", choices=["", "en", "zh"], help="default: same as the run")
    p.add_argument("--timeout", type=float, default=150.0)
    p.add_argument("--max-tokens", type=int, default=4000, dest="max_tokens")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--concurrency", type=int, default=8,
                   help="parallel re-run workers (kept modest to avoid 429s)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--mock", action="store_true", help="offline smoke test (no network)")
    return p


def main():
    args = build_parser().parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n[rerun] interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
