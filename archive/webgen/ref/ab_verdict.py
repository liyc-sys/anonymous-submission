#!/usr/bin/env python3
"""A/B paired verdict tool (works for train/heldout, idempotent read-only): compares per-case means of two arms and produces a gate verdict.

Usage (on pod or locally with a results directory):
  python3 advjudge/ab_verdict.py <armA> <armB> <cases-csv> [--results DIR] [--seeds 0,1,2,3]
Example: python3 advjudge/ab_verdict.py p12v2par p12v2X 000004,000015,000032,000036
Conventions: per case, pair the seed-mean acc of the two arms; diff=B-A; gate is meanΔ>=+0.04 and wins>losses.
Also reports the number of crashed slots (launched=false/acc missing).
"""
import argparse
import json
import re
import statistics
from pathlib import Path

DF = Path("<frozen_corpus>/dataflow/results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_a")
    ap.add_argument("arm_b")
    ap.add_argument("cases")
    ap.add_argument("--results", default=str(DF))
    ap.add_argument("--seeds", default="0,1")
    args = ap.parse_args()
    R = Path(args.results)
    cases = args.cases.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    res = {}
    pat = re.compile(r"run_(.+)__(\d+)__s(\d+)(?:__a(\d+))?$")
    for f in R.glob("run_*.json"):
        m = pat.match(f.stem)
        if not m:
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        res[(m.group(1), m.group(2), int(m.group(3)), int(m.group(4) or 0))] = d

    def slot(arm, case, seed):
        d0 = res.get((arm, case, seed, 0))
        if d0 is not None and d0.get("acc") is not None and d0.get("launched", True):
            return d0, False
        d1 = res.get((arm, case, seed, 1))
        if d1 is not None and d1.get("acc") is not None and d1.get("launched", True):
            return d1, False
        return d0 or d1, True

    diffs, crashes = [], {"a": 0, "b": 0}
    for c in cases:
        av, bv = [], []
        for s in seeds:
            da, ca = slot(args.arm_a, c, s)
            db, cb = slot(args.arm_b, c, s)
            crashes["a"] += ca
            crashes["b"] += cb
            if da and da.get("acc") is not None:
                av.append(da["acc"])
            if db and db.get("acc") is not None:
                bv.append(db["acc"])
        if av and bv:
            ma, mb = statistics.mean(av), statistics.mean(bv)
            diffs.append(mb - ma)
            print(f"{c}: {args.arm_a}={ma:.4f} {args.arm_b}={mb:.4f} diff={mb-ma:+.3f}")
        else:
            print(f"{c}: insufficient data (A n={len(av)} B n={len(bv)})")
    if not diffs:
        print("VERDICT: no pairable cases")
        return
    mean = statistics.mean(diffs)
    wins = sum(d > 0.01 for d in diffs)
    losses = sum(d < -0.01 for d in diffs)
    gate = mean >= 0.04 and wins > losses
    print(f"VERDICT meanΔ={mean:+.4f} W/L={wins}/{losses} crashes A/B={crashes['a']}/{crashes['b']} "
          f"gate={'PASS' if gate else 'FAIL'}")


if __name__ == "__main__":
    main()
