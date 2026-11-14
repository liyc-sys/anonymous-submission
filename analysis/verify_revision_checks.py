#!/usr/bin/env python3
"""Supplementary checks: task-clustered sealed statistic, exact order-statistic
constants and the winner's-curse prediction, HJ/LB rates.  Standard library only."""
import json, math, statistics, itertools, sys
from pathlib import Path
DATA = Path(__file__).resolve().parent.parent / "data" / "v2w"
FAIL = []
def check(name, got, want, tol):
    ok = (got == want) if isinstance(want, str) else abs(got - want) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got}, want {want} (+/-{tol})")
    if not ok: FAIL.append(name)

print("== sealed 8x2 pilot, clustered by task ==")
summ = json.load(open(DATA / "11_phase2_official_summaries.json"))["round2"]
pi4 = summ["sealed_jgen4_k4_summary.json"]["per_instance"]
bf = json.load(open(DATA / "sealed_routing" / "bon_final.json"))
base4 = {p["task"] + "|" + p["seed"]: p["baseline"] for p in bf["sealed_k4"]["pairs"]}
common = sorted(set(pi4) & set(base4)); tasks = sorted({c.split("|")[0] for c in common})
d16 = [pi4[c] - base4[c] for c in common]
t16 = statistics.mean(d16) / (statistics.stdev(d16) / math.sqrt(len(d16)))
td = [statistics.mean(pi4[c] - base4[c] for c in common if c.startswith(t + "|")) for t in tasks]
t8 = statistics.mean(td) / (statistics.stdev(td) / math.sqrt(len(td)))
check("groups n", len(d16), 16, 0); check("group-level t", round(t16, 2), 3.26, 0.05)
check("tasks n", len(td), 8, 0); check("task-level t", round(t8, 2), 5.51, 0.05)
check("tasks won", f"{sum(x > 0 for x in td)}/{len(td)}", "8/8", 0)
m = statistics.mean(td); cnt = sum(1 for sg in itertools.product([1, -1], repeat=len(td))
                                if abs(statistics.mean(a * b for a, b in zip(sg, td))) >= abs(m) - 1e-12)
check("sign-flip permutation p", round(cnt / 2 ** len(td), 4), 0.0078, 0.0005)
s0 = [pi4[t + "|seed_0"] - base4[t + "|seed_0"] for t in tasks]; s1 = [pi4[t + "|seed_1"] - base4[t + "|seed_1"] for t in tasks]
mx, my = statistics.mean(s0), statistics.mean(s1)
r = sum((a - mx) * (b - my) for a, b in zip(s0, s1)) / math.sqrt(sum((a - mx) ** 2 for a in s0) * sum((b - my) ** 2 for b in s1))
check("seed0/seed1 gain correlation", round(r, 2), -0.67, 0.02)

print("== order-statistic constants c_N (numerical) ==")
def phi(z): return math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
def Phi(z): return 0.5 * (1 + math.erf(z / math.sqrt(2)))
def cN(n, lo=-10.0, hi=10.0, steps=200000):
    h = (hi - lo) / steps; s = 0.0
    for i in range(steps + 1):
        z = lo + i * h; w = 1 if i in (0, steps) else (4 if i % 2 else 2)
        s += w * n * z * phi(z) * Phi(z) ** (n - 1)
    return s * h / 3
c2, c4, c48 = cN(2), cN(4), cN(48)
check("c_2", round(c2, 3), 0.564, 0.002); check("c_4", round(c4, 3), 1.029, 0.002); check("c_48", round(c48, 2), 2.23, 0.01)
check("sqrt(2 ln 48)", round(math.sqrt(2 * math.log(48)), 2), 2.78, 0.01)
pred = 0.035 * c48; check("predicted optimism 0.035*c_48", round(pred, 3), 0.078, 0.002)
check("observed/predicted", round(0.098 / pred, 2), 1.25, 0.03)

print("== HJ/LB rates from counts (Table 6) ==")
for name, num, den, want in (("naive subset", 4, 145, 2.8), ("calibrated agg.", 0, 72, 0.0), ("challenge", 0, 51, 0.0)):
    check(f"HJ/LB {name} (%)", round(100 * num / den, 1), want, 0.05)
for name, frac, want in (("I2C raw", 0.094, 9.4), ("I2C evolution", 0.055, 5.5), ("I2C cex", 0.031, 3.1)):
    check(f"HJ/LB {name} (%)", round(100 * frac, 1), want, 0.05)

print()
if FAIL: print("FAILED:", ", ".join(FAIL)); sys.exit(1)
print("ALL CHECKS PASSED")
