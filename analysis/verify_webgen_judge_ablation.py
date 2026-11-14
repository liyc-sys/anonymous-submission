#!/usr/bin/env python3
"""Verify WebGen judge ablation (the paper's judge-evolution section, WebGen lineage; P1 text).

Recomputes the unified-set judge-variant metrics from the shipped frozen
per-site task labels (data/webgen/runs_sitelabels.json): the E1 batch
(g0/g3b/p7 x 83 untouched cases x 2 seeds), sites included when task
labels idx 0..3 are all present (paper: 519 sites, 1112 decidable pairs).

Judge variants (diagnostic-subset scorers over official per-task labels):
  naive01 [0,1] | diag123 [1,2,3] | k3_012 [0,1,2]
  rot_ensemble = mean of top-3 calibrated subsets {(0,2,3),(1,2,3),(0,1,3)}
  diag123_rescue = diag123, plus idx0 re-probe when diag <= 0.1
Pair accuracy: same-case cross-arm cross-seed pairs, official ties dropped,
judge ties count 0.5. HJ/LB: judge>=0.8 & full<=0.4. zero-kill: judge<=0.1 & full>=0.4.

Paper values checked (judge-evolution section; P1 text false-zero 27.2% -> 5.8%):
  naive 0.793/Sp.0.737/HJLB 4 of 145; calibrated 0.850/0.849/0 of 72;
  rescue 0.867/0.862/0 of 72; challenge 0.917/0.917/0 of 51.
"""
import json
import statistics
import sys
from itertools import combinations
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "webgen"
FAIL = []


def check(name, got, want, tol):
    ok = abs(got - want) <= tol if not isinstance(want, str) else got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got}, want {want} (+/-{tol})")
    if not ok:
        FAIL.append(name)


rows = [r for r in json.load(open(DATA / "runs_sitelabels.json"))
        if r["arm"] in ("g0", "g3b", "p7") and not r["is_reeval"]
        and all(str(i) in r["tasks"] for i in range(4))]
print(f"unified set sites: {len(rows)} (paper: 519)")
check("n sites", len(rows), 519, 0)

VARIANTS = {"naive01": [0, 1], "diag123": [1, 2, 3], "k3_012": [0, 1, 2]}
ROT = [(0, 2, 3), (1, 2, 3), (0, 1, 3)]


def jscore(tasks, v):
    if v == "rot_ensemble":
        return statistics.mean(statistics.mean(tasks[str(i)] for i in sub) for sub in ROT)
    if v == "diag123_rescue":
        base = statistics.mean(tasks[str(i)] for i in (1, 2, 3))
        if base <= 0.1:
            return statistics.mean([tasks[str(i)] for i in (1, 2, 3)] + [tasks["0"]])
        return base
    return statistics.mean(tasks[str(i)] for i in VARIANTS[v])


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy)


def metrics(v):
    by_case = {}
    js, fs = [], []
    for r in rows:
        j = jscore(r["tasks"], v)
        by_case.setdefault(r["case"], []).append((j, r["full"]))
        js.append(j)
        fs.append(r["full"])
    agree = tot = 0
    for c, rs in by_case.items():
        for (ja, fa), (jb, fb) in combinations(rs, 2):
            df, dj = fa - fb, ja - jb
            if df != 0:
                tot += 1
                agree += 0.5 if dj == 0 else (1 if (dj > 0) == (df > 0) else 0)
    hi = [(j, f) for j, f in zip(js, fs) if j >= 0.8]
    hjlb = sum(1 for j, f in hi if f <= 0.4)
    zero = [(j, f) for j, f in zip(js, fs) if j <= 0.1]
    zkill = sum(1 for j, f in zero if f >= 0.4)
    return {"agree": agree / tot, "n_pairs": tot, "sp": spearman(js, fs),
            "hjlb": f"{hjlb}/{len(hi)}", "zkill": zkill / len(zero) if zero else None}


want = {
    "naive01": (0.793, 0.737, "4/145", 0.272),
    "diag123": (0.850, 0.849, "0/72", None),
    "diag123_rescue": (0.867, 0.862, "0/72", None),
    "rot_ensemble": (0.917, 0.917, "0/51", 0.058),
}
for v in ("naive01", "diag123", "k3_012", "diag123_rescue", "rot_ensemble"):
    m = metrics(v)
    line = f"{v}: pair_acc {m['agree']:.4f} (pairs {m['n_pairs']}), spearman {m['sp']:.4f}, HJ/LB {m['hjlb']}, zero-kill {m['zkill']:.4f}"
    print(line)
    if v in want:
        wa, ws, wh, wz = want[v]
        check(f"{v} pair acc", round(m["agree"], 3), wa, 0.004)
        check(f"{v} spearman", round(m["sp"], 3), ws, 0.004)
        check(f"{v} HJ/LB", m["hjlb"], wh, 0)
        if wz is not None:
            check(f"{v} zero-kill rate", round(m["zkill"], 3), wz, 0.004)
    if v == "naive01":
        check("decidable pairs", m["n_pairs"], 1112, 0)

print()
if FAIL:
    print("FAILED:", ", ".join(FAIL))
    sys.exit(1)
print("ALL CHECKS PASSED")
