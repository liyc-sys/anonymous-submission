#!/usr/bin/env python3
"""E1 three-arm 83-case unbiased effect verdict (paper main results table, idempotent read-only).

Reads run_{g0,g3b,p7}__<case>__s<seed>.json from dataflow/results (earliest successful score: s0/s1/__a1),
per-case arm means -> three paired differences (p7-g0 / p7-g3b / g3b-g0): meanΔ, 95%CI(t), median Δ, W/L(±0.01), coverage counts.
Usage: python3 advjudge/e1_verdict.py [--cases-csv FILE] [--out JSON]
"""
import argparse
import json
import math
import re
import statistics
from pathlib import Path

DF = Path("<frozen_corpus>/dataflow/results")
POOLS = json.loads((Path(__file__).resolve().parents[1] / "pools.json").read_text())
ARMS = ["g0", "g3b", "p7"]
PAIRS = [("p7", "g0"), ("p7", "g3b"), ("g3b", "g0")]


def t_crit(df):
    # 95% two-sided t critical value table (common df 1-30 values, approximated as 1.96 beyond)
    tab = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
           8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
           15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
           25: 2.060, 30: 2.042}
    if df <= 0:
        return float("nan")
    if df > 30:
        return 1.96
    for k in sorted(tab, reverse=True):
        if df >= k:
            return tab[k]
    return tab[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "advjudge" / "lab" / "e1_verdict.json"))
    args = ap.parse_args()

    used = set(POOLS["train"]) | set(POOLS["heldout"])
    allids = []
    for i, line in enumerate((Path(__file__).resolve().parents[1] / "data" / "test.jsonl").read_text().splitlines(), 1):
        if line.strip():
            allids.append(json.loads(line).get("id", str(i).zfill(6)))
    cases = [c for c in sorted(allids) if c not in used]

    res = {}
    pat = re.compile(r"run_(g0|g3b|p7)__(\d{6})__s(\d+)(?:__a(\d+))?$")
    for f in DF.glob("run_*.json"):
        m = pat.match(f.stem)
        if not m:
            continue
        try:
            res[(m.group(1), m.group(2), int(m.group(3)), int(m.group(4) or 0))] = json.loads(f.read_text())
        except Exception:
            pass

    def slot(arm, case, seed):
        d0 = res.get((arm, case, seed, 0))
        if d0 is not None and d0.get("acc") is not None and d0.get("launched", True):
            return d0, False
        d1 = res.get((arm, case, seed, 1))
        if d1 is not None and d1.get("acc") is not None and d1.get("launched", True):
            return d1, False
        return d0 or d1, True

    case_means = {}
    crashes = {a: 0 for a in ARMS}
    covered_seeds = 0
    for c in cases:
        case_means[c] = {}
        for a in ARMS:
            vals = []
            for s in (0, 1):
                d, crashed = slot(a, c, s)
                crashes[a] += crashed
                if d and d.get("acc") is not None and d.get("launched", True):
                    vals.append(d["acc"])
                    covered_seeds += 1
            if vals:
                case_means[c][a] = sum(vals) / len(vals)

    out = {"n_cases": len(cases), "covered_seed_slots": covered_seeds,
           "crashes": crashes, "arms": {}, "pairs": {}}
    for a in ARMS:
        v = [case_means[c][a] for c in cases if a in case_means[c]]
        out["arms"][a] = {"n_cases": len(v), "mean_acc": round(sum(v) / len(v), 4) if v else None}

    for b, a in PAIRS:
        diffs = [(case_means[c][b] - case_means[c][a], c)
                 for c in cases if a in case_means[c] and b in case_means[c]]
        dv = [d for d, _ in diffs]
        if not dv:
            continue
        n = len(dv)
        mean = sum(dv) / n
        sd = statistics.stdev(dv) if n > 1 else 0.0
        ci = t_crit(n - 1) * sd / math.sqrt(n)
        med = statistics.median(dv)
        wins = sum(1 for d in dv if d > 0.01)
        losses = sum(1 for d in dv if d < -0.01)
        ties = n - wins - losses
        out["pairs"][f"{b}-{a}"] = {
            "n_cases": n, "mean_delta": round(mean, 4),
            "ci95": round(ci, 4), "sd": round(sd, 4),
            "median": round(med, 4), "W/L/T": f"{wins}/{losses}/{ties}",
            "lo": round(mean - ci, 4), "hi": round(mean + ci, 4),
            "worst": sorted(diffs)[:3], "best": sorted(diffs)[-3:],
        }

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
