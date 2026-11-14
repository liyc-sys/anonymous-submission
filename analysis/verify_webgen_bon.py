#!/usr/bin/env python3
"""Verify the WebGen best-of-N offline simulation (paper P3 + Appendix I).

Recomputes test-time best-of-N from the shipped per-site task labels
(data/webgen/runs_sitelabels.json). Candidates = independent generations
(seed_* dirs; *_rN re-evaluation dirs excluded). Selector = diag[1,2,3]
partial score (ties averaged); oracle = full acc; scoring on the nondiag
complement (zero overlap with the selection signal). Enumeration over all
C(k,N) subsets, averaged per case then across cases.

Paper values checked (routing text; routing-table bottom; appendix WebGen rows):
  champion arm pb3 (12 heldout): N=1 0.725 -> N=4 0.809 (+8.4), oracle 0.860;
  g3b (83 cases): N=2 +3.7 (0.731 -> 0.764);
  Gemma m0: N=4 0.629 -> 0.761 (+13), oracle 0.822;
  implied sigma_gen (oracle gain / c_N) = 0.187 vs calibration 0.194.
"""
import json
import statistics
import sys
from itertools import combinations
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "webgen"
FAIL = []
DIAG = (1, 2, 3)


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got}, want {want} (+/-{tol})")
    if not ok:
        FAIL.append(name)


rows = json.load(open(DATA / "runs_sitelabels.json"))
cand = {}
for r in rows:
    if r["is_reeval"] or not r["launched"] or not r["tasks"]:
        continue
    t = r["tasks"]
    diag = [t[str(i)] for i in DIAG if str(i) in t]
    nond = [v for i, v in t.items() if int(i) not in DIAG]
    if not diag or not nond:
        continue
    cand.setdefault(r["arm"], {}).setdefault(r["case"], []).append({
        "full": r["full"],
        "diag": statistics.mean(diag),
        "nondiag": statistics.mean(nond),
    })

e1_common = None
for a in ("g0", "g3b", "p7"):
    cs = set(cand.get(a, {}))
    e1_common = cs if e1_common is None else (e1_common & cs)


def simulate(by_case, N):
    r_full, j_full, o_full, r_nd, j_nd, o_nd = [], [], [], [], [], []
    for case, cs in by_case.items():
        if len(cs) < N:
            continue
        agg = {k: [] for k in range(6)}
        for sub in combinations(cs, N):
            agg[0].append(statistics.mean(c["full"] for c in sub))
            bestd = max(c["diag"] for c in sub)
            picked = [c for c in sub if c["diag"] == bestd]
            agg[1].append(statistics.mean(c["full"] for c in picked))
            orc = max(sub, key=lambda c: c["full"])
            agg[2].append(orc["full"])
            agg[3].append(statistics.mean(c["nondiag"] for c in sub))
            agg[4].append(statistics.mean(c["nondiag"] for c in picked))
            agg[5].append(orc["nondiag"])
        for i, out in enumerate((r_full, j_full, o_full, r_nd, j_nd, o_nd)):
            if agg[i]:
                out.append(statistics.mean(agg[i]))
    return {"n": len(r_nd),
            "random_nd": statistics.mean(r_nd), "judge_nd": statistics.mean(j_nd),
            "oracle_nd": statistics.mean(o_nd), "judge_full": statistics.mean(j_full),
            "oracle_full": statistics.mean(o_full), "random_full": statistics.mean(r_full)}


def arm_sim(arm, N, restrict=None):
    bc = cand.get(arm, {})
    if restrict is not None:
        bc = {c: v for c, v in bc.items() if c in restrict}
    return simulate(bc, N)


print("== champion arm pb3 (12 heldout, 4 seeds) ==")
s1, s2, s4 = arm_sim("pb3", 1), arm_sim("pb3", 2), arm_sim("pb3", 4)
print(f"N=1 {s1['random_nd']:.4f} -> N=2 {s2['judge_nd']:.4f} -> N=4 {s4['judge_nd']:.4f} (oracle {s4['oracle_nd']:.4f})")
check("pb3 N=1 nondiag", round(s1["random_nd"], 3), 0.725, 0.004)
check("pb3 N=4 judge nondiag (+8.4)", round(s4["judge_nd"], 3), 0.809, 0.004)
check("pb3 N=4 oracle nondiag", round(s4["oracle_nd"], 3), 0.860, 0.004)

print("== g3b (83 cases, E1 common set) ==")
s1, s2 = arm_sim("g3b", 1, e1_common), arm_sim("g3b", 2, e1_common)
print(f"n={s2['n']}, N=1 {s1['random_nd']:.4f} -> N=2 {s2['judge_nd']:.4f}")
check("g3b N=1 nondiag", round(s1["random_nd"], 3), 0.731, 0.004)
check("g3b N=2 judge nondiag (+3.7)", round(s2["judge_nd"], 3), 0.764, 0.004)

print("== Gemma m0 (12 heldout) ==")
s1, s4 = arm_sim("m0", 1), arm_sim("m0", 4)
print(f"N=1 {s1['random_nd']:.4f} -> N=4 {s4['judge_nd']:.4f} (oracle {s4['oracle_nd']:.4f})")
check("m0 N=1 nondiag", round(s1["random_nd"], 3), 0.629, 0.004)
check("m0 N=4 judge nondiag (+13)", round(s4["judge_nd"], 3), 0.761, 0.004)
check("m0 N=4 oracle nondiag", round(s4["oracle_nd"], 3), 0.822, 0.004)
implied = (s4["oracle_nd"] - s1["random_nd"]) / 1.029
check("implied sigma_gen (routing-table cross-check)", round(implied, 3), 0.187, 0.004)
check("implied vs calib 0.194 within 5%", round(implied, 3), 0.194, 0.01)

print()
if FAIL:
    print("FAILED:", ", ".join(FAIL))
    sys.exit(1)
print("ALL CHECKS PASSED")
