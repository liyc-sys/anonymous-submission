#!/usr/bin/env python3
"""Verify the WebGen official-noise statistics (paper Setup + Appendix C).

Source: slots whose official evaluator produced >= 2 independent repeated
scores (rep_accs) in data/webgen/dataflow_slots.json, earliest successful
attempt per (arm, case, seed).

Paper values checked (Setup; evaluator-noise appendix; noise table caption):
  site-level re-eval: n = 696 sites, mean |rep0-rep1| = 0.082 (median 0.071),
  sigma_judge estimate from re-evals = 0.087 (Appendix C);
  pairwise flip rate pooled = 0.333 over 234 pairs;
  variance decomposition sigma_judge = 0.141, sigma_gen = 0.194
  (documented calibration constants, cross-checked against the shipped
  summary).
"""
import json
import statistics
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "webgen"
FAIL = []


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got}, want {want} (+/-{tol})")
    if not ok:
        FAIL.append(name)


slots = json.load(open(DATA / "dataflow_slots.json"))
reps = {}
for s in slots:
    if s["eval_reps"] < 2 or not s["rep_accs"] or not s["launched"]:
        continue
    if any(a is None for a in s["rep_accs"]):
        continue
    k = (s["arm"], s["case"], s["seed"])
    if k not in reps or s["attempt"] < reps[k][1]:
        reps[k] = (s["rep_accs"], s["attempt"])

diffs_signed = [v[0][0] - v[0][1] for v in reps.values()]
diffs = [abs(d) for d in diffs_signed]
sd = statistics.stdev(diffs_signed)
print(f"sites with 2 official reps: {len(diffs)}")
check("n sites with 2 reps", len(diffs), 696, 0)
check("mean |rep delta|", round(statistics.mean(diffs), 4), 0.082, 0.004)
check("median |rep delta|", round(statistics.median(diffs), 4), 0.071, 0.004)
check("sigma_judge est (sd/sqrt2)", round(sd / 2 ** 0.5, 4), 0.087, 0.004)

AB = [("final_gen0", "final_gen3b"), ("ctrlF", "navF"), ("r7fC", "r7fX"),
      ("r18fC", "r18fX"), ("pb3", "ns3")]
pf = pn = 0
for a, b in AB:
    for (arm, case, seed), (accs, _) in reps.items():
        if arm != a:
            continue
        other = reps.get((b, case, seed))
        if not other:
            continue
        s0 = (accs[0] > other[0][0]) - (accs[0] < other[0][0])
        s1 = (accs[1] > other[0][1]) - (accs[1] < other[0][1])
        pn += 1
        pf += (s0 != s1)
print(f"pairwise flips: {pf}/{pn}")
check("flip pairs n", pn, 234, 0)
check("pooled flip rate", round(pf / pn, 3), 0.333, 0.004)

summ = json.load(open(DATA / "summaries" / "official_judge_noise.json"))
cc = summ["mac_era_calibration_constants"]
print("== documented calibration constants (Setup) ==")
check("sigma_judge (documented)", cc["sigma_judge"], 0.141, 0.0005)
check("sigma_gen (documented)", cc["sigma_gen"], 0.194, 0.0005)

print()
if FAIL:
    print("FAILED:", ", ".join(FAIL))
    sys.exit(1)
print("ALL CHECKS PASSED")
