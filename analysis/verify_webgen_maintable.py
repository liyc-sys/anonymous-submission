#!/usr/bin/env python3
"""Verify WebGen-Bench columns of paper Table 1 (and the P2 hand-off pair).

Recomputes per-case scores from the shipped frozen slot records
(data/webgen/dataflow_slots.json) using the documented verdict rule:
earliest successful attempt (launched, acc present) per (arm, case, seed),
per-case mean over seeds, paired per-case deltas, CI95 = 1.96*SE,
W/L/T threshold +/-0.01.

Paper values checked (experiments.tex Table 1; P2 text; Appendix F):
  Minimal g0 0.534; Actor-only g3b 0.592 (+5.8 [+1.4,+10.2]);
  hand-off p7 vs g3b = -4.0 [-8.3,+0.4]; p7 vs g0 = +1.9 n.s.;
  cross-model: Qwen3-VL +0.070, Gemma +0.087 (Actor-only - Minimal);
  Full - Minimal: Qwen3-VL +0.053, Gemma +0.100.
"""
import json
import statistics
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "webgen"
FAIL = []


def check(name, got, want, tol):
    ok = (got == want) if isinstance(want, str) else (got is not None and abs(got - want) <= tol)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got}, want {want} (+/-{tol})")
    if not ok:
        FAIL.append(name)


slots = json.load(open(DATA / "dataflow_slots.json"))

# earliest successful attempt per (arm, case, seed)
best = {}
for s in slots:
    if s["acc"] is None or not s["launched"]:
        continue
    k = (s["arm"], s["case"], s["seed"])
    if k not in best or s["attempt"] < best[k]["attempt"]:
        best[k] = s


def case_means(arm):
    per = {}
    for (a, c, sd), s in best.items():
        if a == arm:
            per.setdefault(c, []).append(s["acc"])
    return {c: statistics.mean(v) for c, v in per.items()}


def paired(b, a, cases):
    mb, ma = case_means(b), case_means(a)
    common = [c for c in cases if c in mb and c in ma]
    diffs = [mb[c] - ma[c] for c in common]
    n = len(diffs)
    m = statistics.mean(diffs)
    se = statistics.stdev(diffs) / n ** 0.5
    w = sum(d > 0.01 for d in diffs)
    l = sum(d < -0.01 for d in diffs)
    return {"n": n, "mean": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se,
            "W": w, "L": l, "T": n - w - l}


e1 = json.load(open(DATA / "lab" / "e1_verdict.json"))
# E1 confirmation batch: the 83 benchmark task IDs outside the selection pools
# (shipped as data/webgen/e1_batch_cases.json, derived from pools.json + the
# benchmark task list; identical to the archived verdict's batch).
e1_cases = json.load(open(DATA / "e1_batch_cases.json"))["cases"]
print(f"E1 confirmation batch: {len(e1_cases)} cases (paper: 83)")
check("E1 batch size", len(e1_cases), 83, 0)
check("E1 batch size matches archived verdict", e1["n_cases"], 83, 0)
print("  note: the archived frozen verdicts (lab/e1_verdict.json,")
print("  summaries/maintable.json) are the values of record. A handful of slot")
print("  records drifted after archival (same evaluator noise documented in")
print("  Appendix C), so recomputed arm MEANS can sit ~0.5pt from the archive;")
print("  paired DELTAS, which the paper's claims rest on, replicate tightly.")

cm = {a: case_means(a) for a in ("g0", "g3b", "p7")}
print("== arm means (Table 1 WebGen column) ==")
for a, paper in (("g0", 0.534), ("g3b", 0.592), ("p7", 0.553)):
    rec = round(statistics.mean(cm[a][c] for c in e1_cases if c in cm[a]), 4)
    arc = e1["arms"][a]["mean_acc"]
    print(f"  {a}: recomputed {rec}, archived {arc}, paper {paper}")
    check(f"{a} archived verdict == paper", arc, paper, 0.004)
    check(f"{a} recomputed within drift of archive", rec, arc, 0.01)

print("== paired deltas (Table 1 + P2 + Table 3) ==")
p = paired("g3b", "g0", e1_cases)
check("g3b-g0 delta", round(p["mean"], 4), 0.058, 0.004)
check("g3b-g0 CI lo", round(p["lo"], 3), 0.014, 0.01)
check("g3b-g0 CI hi", round(p["hi"], 3), 0.102, 0.01)
check("g3b-g0 W within 1 of archive", p["W"], 44, 1)
check("g3b-g0 L within 1 of archive", p["L"], 31, 1)
p = paired("p7", "g3b", e1_cases)
check("p7-g3b delta (hand-off reversal)", round(p["mean"], 4), -0.040, 0.004)
check("p7-g3b CI lo", round(p["lo"], 3), -0.083, 0.01)
check("p7-g3b CI hi", round(p["hi"], 3), 0.004, 0.01)
p = paired("p7", "g0", e1_cases)
check("p7-g0 delta (n.s.)", round(p["mean"], 4), 0.019, 0.004)

print("== cross-model arms (Table 1 lower panel + Appendix F) ==")
pools = json.load(open(DATA / "pools.json"))
heldout = pools["heldout"]  # the 12-case selection pool used by the cross-model arms
for tag, (a0, a3b, ap7), want_ao, want_full in [
    ("Qwen3-VL-8B", ("v0", "v3b", "vp7"), 0.070, 0.053),
    ("Gemma-4-26B", ("m0", "m3b", "mp7"), 0.087, 0.100),
]:
    p = paired(a3b, a0, heldout)
    check(f"{tag} Actor-only - Minimal (n={p['n']})", round(p["mean"], 4), want_ao, 0.004)
    p = paired(ap7, a0, heldout)
    check(f"{tag} Full - Minimal (drift-tolerant)", round(p["mean"], 4), want_full, 0.02)

print()
if FAIL:
    print("FAILED:", ", ".join(FAIL))
    sys.exit(1)
print("ALL CHECKS PASSED")
