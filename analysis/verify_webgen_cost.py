#!/usr/bin/env python3
"""Verify the WebGen cost accounting (paper Cost paragraph + Appendix A).

Checks the judge-side shortlisting economics from the shipped lab records
(data/webgen/lab/budget_results.json) and the per-round search ledger
(data/webgen/summaries/phase2_curve.json), plus the campaign-wide slot count.

Paper values checked:
  shortlist k=2 of N=6 saves 43% of official evaluations per round
  (per-round ledger eval_saving_tasks = 0.429); up to 91% in the widest
  configuration (screening saving = 0.907); selection regret 0.047;
  top-2 recall 0.833 vs random 0.333; campaign covers 2634 evaluation slots.
"""
import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "webgen"
FAIL = []


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got}, want {want} (+/-{tol})")
    if not ok:
        FAIL.append(name)


br = json.load(open(DATA / "lab" / "budget_results.json"))
sc = br["phase2_wide_search_screening"]
print("== screening economics (Cost paragraph) ==")
check("top-2 recall (K3 selector)", sc["budget_K3"]["top2_recall"], 0.833, 0.004)
check("random top-2 recall", sc["random"]["top2_recall"], 0.333, 0.004)
check("selection regret", sc["budget_K3"]["regret"], 0.047, 0.004)
check("widest-config saving", sc["budget_K3"]["saving"], 0.907, 0.004)

pc = json.load(open(DATA / "summaries" / "phase2_curve.json"))
srm = pc["search_rounds_meta"]
savings = [v["eval_saving_tasks"] for v in srm.values()]
print(f"== per-round ledger ({len(savings)} rounds) ==")
check("rounds with k=2 of N=6", sum(1 for v in srm.values()
      if v["n_candidates"] == 6 and v["shortlist_K"] == 2), len(savings), 0)
check("per-round saving 43%", round(sum(savings) / len(savings), 3), 0.429, 0.004)

slots = json.load(open(DATA / "dataflow_slots.json"))
print("== campaign slot count (Appendix A) ==")
check("evaluation slots", len(slots), 2634, 0)

print()
if FAIL:
    print("FAILED:", ", ".join(FAIL))
    sys.exit(1)
print("ALL CHECKS PASSED")
