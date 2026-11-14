#!/usr/bin/env python3
"""Verify the V2W main-table evolution result and the policy hand-off control
(paper Table 1 V2W column; Table 3 V2W row; P2 text).

Source: data/v2w/evidence_ladder/lad*.json (per-run official scores of the
ladder confirmation batch, all arms scored in one batch) and
data/v2w/12_multiround_screen_audits.json (multiround final verdict).

Paper values checked:
  Minimal (lad0) 0.495 vs champion gen10 (lad10) 0.564, +7.0, t = 3.08;
  gen11b/gen12 micro-regressions (-0.015 / -0.005, screening noise);
  hand-off gen17b +1.8 [-1.5,+5.0] vs champion (49.1 on that batch);
  every single-mutation descendant of the champion failed (worst -4.8).
"""
import json
import statistics
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "v2w"
FAIL = []


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got}, want {want} (+/-{tol})")
    if not ok:
        FAIL.append(name)


lad = {}
fit = {}
for n in ("lad0", "lad8b", "lad9", "lad10", "lad11b", "lad12"):
    d = json.load(open(DATA / "evidence_ladder" / f"{n}.json"))
    lad[n] = {(r["task"], r["seed"]): r["score"] for r in d["runs"]
              if r.get("score") is not None}
    fit[n] = d["fitness"]

print("== Table 1 V2W column (ladder confirmation batch) ==")
print("  arm levels are the archived per-arm fitness (valid runs of one batch);")
print("  the paired t is recomputed on the two arms' common valid runs.")
check("Minimal (lad0) fitness", fit["lad0"], 0.495, 0.004)
check("champion gen10 (lad10) fitness", fit["lad10"], 0.564, 0.004)
check("gen10 - Minimal (fitness)", round(fit["lad10"] - fit["lad0"], 3), 0.070, 0.004)

common = sorted(set(lad["lad0"]) & set(lad["lad10"]))
diffs = [lad["lad10"][k] - lad["lad0"][k] for k in common]
m = statistics.mean(diffs)
t = m / (statistics.stdev(diffs) / len(diffs) ** 0.5)
print(f"  recomputed on {len(common)} common valid runs: delta {m:.4f}, t {t:.2f}")
check("gen10 - Minimal delta (common runs)", round(m, 3), 0.070, 0.01)
check("gen10 - Minimal significant (paper t=3.08)", round(t, 2), 3.08, 0.6)

print("== micro-regressions after the champion (P2/Appendix B) ==")
check("lad11b - lad10", round(fit["lad11b"] - fit["lad10"], 3), -0.015, 0.004)
check("lad12 - lad11b", round(fit["lad12"] - fit["lad11b"], 3), -0.005, 0.004)

mr = json.load(open(DATA / "12_multiround_screen_audits.json"))["multiround_final.json"]
print("== Table 3 V2W hand-off control (final batch) ==")
check("champion on final batch", mr["baseline_mean"], 0.491, 0.005)
check("hand-off delta", round(mr["mean_delta"], 4), 0.018, 0.004)
check("hand-off CI lo", round(mr["ci95"][0], 3), -0.015, 0.005)
check("hand-off CI hi", round(mr["ci95"][1], 3), 0.050, 0.005)

print()
if FAIL:
    print("FAILED:", ", ".join(FAIL))
    sys.exit(1)
print("ALL CHECKS PASSED")
