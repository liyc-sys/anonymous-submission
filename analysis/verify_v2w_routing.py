#!/usr/bin/env python3
"""Verify V2W instance-level routing (paper Table 4 + Appendix G + sealed headline).

Three independent checks:
1. Ladder-pool replay (Table 4 top + Appendix G): 16 complete groups x 4 seeds,
   from data/v2w/30_bon_night3_replay.json.
2. Dev-pool upgraded judge (Appendix G): jgen4 dev replay N=4, N=6
   (data/v2w/31_bon_phase2_dev_replay.json, data/v2w/34_recon_calib_stats.json).
3. Sealed 8x2 validation (Table 4 bottom + P3): jgen4+vote-tie-break routing vs
   fresh-champion baseline, from data/v2w/11_phase2_official_summaries.json
   (round2 sealed_jgen4_k2/k4 per-instance) and
   data/v2w/sealed_routing/bon_final.json (same-batch gen10 baseline and the
   archived jgen2 routing). Upgrade increment +4.73 (t=4.59, 14/16).

Paper values checked:
  ladder: random 53.4 flat; challenge 55.7/56.6/57.3; oracle 56.8/58.0/58.8;
  implied sigma_gen 0.060/0.055/0.053; capture 72% at N=4;
  dev upgraded judge 59.4 (N=4) / 62.6 (N=6);
  sealed: 55.7 -> 62.4 (+6.7, t=3.26, 12/16); K2 +5.3; archived jgen2 +2.0 (t=0.91).
"""
import json
import statistics
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "v2w"
FAIL = []


def check(name, got, want, tol):
    ok = (got == want) if isinstance(want, str) else abs(got - want) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got}, want {want} (+/-{tol})")
    if not ok:
        FAIL.append(name)


print("== ladder-pool replay (Table 4) ==")
t = json.load(open(DATA / "30_bon_night3_replay.json"))["table"]
for judge, wants in (("jgen2", {2: 0.557, 3: 0.566, 4: 0.573}),
                     ("oracle", {2: 0.568, 3: 0.580, 4: 0.588})):
    for n, w in wants.items():
        check(f"{judge} N={n}", round(t[judge][f"N={n}"], 3), w, 0.004)
for n in (2, 3, 4):
    check(f"random N={n} flat", round(t["random"][f"N={n}"], 4), round(t["random"]["N=1"], 4), 0.0005)
c = {2: 0.564, 3: 0.846, 4: 1.029}
for n, w in ((2, 0.060), (3, 0.055), (4, 0.053)):
    implied = (t["oracle"][f"N={n}"] - t["oracle"]["N=1"]) / c[n]
    check(f"implied sigma_gen N={n}", round(implied, 3), w, 0.004)
capture = (t["jgen2"]["N=4"] - t["random"]["N=4"]) / (t["oracle"]["N=4"] - t["random"]["N=4"])
check("capture ratio at N=4", round(capture, 2), 0.72, 0.02)

print("== dev-pool upgraded judge (Appendix G) ==")
t31 = json.load(open(DATA / "31_bon_phase2_dev_replay.json"))["jgen4_dev48"]["table"]
check("dev jgen4 N=4", round(t31["jgen4_dev48"]["N=4"], 3), 0.594, 0.004)
st34 = json.load(open(DATA / "34_recon_calib_stats.json"))
s = json.dumps(st34)
import re
m = re.search(r'"N=6"\s*:\s*0\.62[0-9]+', s)
found = float(m.group(0).split(":")[1]) if m else None
print(f"  34_recon_calib_stats N=6 entry: {found}")
check("dev upgraded judge N=6", found, 0.626, 0.004)

print("== sealed 8x2 validation (headline) ==")
summ = json.load(open(DATA / "11_phase2_official_summaries.json"))["round2"]
pi4 = summ["sealed_jgen4_k4_summary.json"]["per_instance"]
pi2 = summ["sealed_jgen4_k2_summary.json"]["per_instance"]
bf = json.load(open(DATA / "sealed_routing" / "bon_final.json"))
base4 = {p["task"] + "|" + p["seed"]: p["baseline"] for p in bf["sealed_k4"]["pairs"]}
base2 = {p["task"] + "|" + p["seed"]: p["baseline"] for p in bf["sealed_k2"]["pairs"]}
arch4 = {p["task"] + "|" + p["seed"]: p["candidate"] for p in bf["sealed_k4"]["pairs"]}


def paired(new, old):
    common = sorted(set(new) & set(old))
    diffs = [new[c] - old[c] for c in common]
    m = statistics.mean(diffs)
    tt = m / (statistics.stdev(diffs) / len(diffs) ** 0.5)
    w = sum(1 for d in diffs if d > 0)
    return m, tt, w, len(diffs)


m, tt, w, n = paired(pi4, base4)
check("sealed K4 delta (+6.7)", round(m, 4), 0.067, 0.004)
check("sealed K4 t", round(tt, 2), 3.26, 0.05)
check("sealed K4 wins", f"{w}/{n}", "12/16", 0)
check("sealed K4 champion baseline", round(statistics.mean(base4.values()), 4), 0.557, 0.004)
check("sealed K4 routing mean", round(statistics.mean(pi4.values()), 4), 0.624, 0.004)
m, tt, w, n = paired(pi2, base2)
check("sealed K2 delta", round(m, 4), 0.053, 0.004)
m, tt, w, n = paired(arch4, base4)
check("archived jgen2 routing delta (+2.0)", round(m, 4), 0.020, 0.004)
check("archived jgen2 t (0.91)", round(tt, 2), 0.91, 0.05)
m, tt, w, n = paired(pi4, arch4)
check("routing upgrade increment (+4.7)", round(m, 4), 0.047, 0.004)
check("routing upgrade t (4.59)", round(tt, 2), 4.59, 0.05)
check("routing upgrade wins", f"{w}/{n}", "14/16", 0)

print()
if FAIL:
    print("FAILED:", ", ".join(FAIL))
    sys.exit(1)
print("ALL CHECKS PASSED")
