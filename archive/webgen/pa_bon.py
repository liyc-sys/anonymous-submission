#!/usr/bin/env python3
"""S4 Test-time Best-of-N offline simulation (zero compute, reads only existing per-task labels).

Data: runs/<arm>/<case>/seed_*/<case>/result.json — each seed is one independent generation of the same case under the same harness.
Arms and N: cases with >=4 successful seeds support N=1/2/4; 2-seed arms (E1: g0/g3b/p7, 83 cases) support N=1/2.
Selectors: random (expectation=candidate mean) / judge=diag[1,2,3] partial max (ties averaged) / oracle=full acc max.
Scoring: (1) full acc (official convention, note 3 tasks overlap with the judge selection signal);
     (2) nondiag acc (excludes tasks idx 1,2,3; clean convention with no selection/scoring overlap).
Enumeration: mean over all C(k,N) subsets per case, then mean over cases.
Output: paper_assets/webgen/bon_simulation.json
"""
import json
import re
import statistics
from itertools import combinations
from pathlib import Path

RUNS = Path("<frozen_corpus>/runs")
OUT = Path("<frozen_corpus>/paper_assets")
OUT.mkdir(parents=True, exist_ok=True)

ARMS = ["g0", "g3b", "p7", "m0", "m3b", "mp7", "v0", "v3b", "vp7", "pb3", "ns3"]
DIAG = (1, 2, 3)


def load_candidates(arm):
    by_case = {}
    root = RUNS / arm
    if not root.is_dir():
        return by_case
    for case_dir in sorted(root.glob("*")):
        if not case_dir.is_dir() or not re.fullmatch(r"\d{6}", case_dir.name):
            continue
        cands = []
        for seed_dir in sorted(case_dir.glob("seed_*")):
            if re.search(r"_r\d+$", seed_dir.name):
                continue  # _rN = second evaluation of the same site (K2 re-eval), not an independent generation, not a candidate
            rj = seed_dir / case_dir.name / "result.json"
            if not rj.exists():
                continue
            try:
                r = json.loads(rj.read_text())
            except Exception:
                continue
            ui = r.get("ui") or []
            tasks = {u["idx"]: u["score"] for u in ui if "idx" in u and "score" in u}
            if not tasks or not r.get("launched", True):
                continue
            diag = [tasks[i] for i in DIAG if i in tasks]
            nond = [v for i, v in tasks.items() if i not in DIAG]
            cands.append({
                "seed": seed_dir.name,
                "full": sum(tasks.values()) / len(tasks),
                "diag": sum(diag) / len(diag) if diag else None,
                "nondiag": sum(nond) / len(nond) if nond else None,
            })
        if cands:
            by_case[case_dir.name] = cands
    return by_case


def simulate(by_case, N):
    per_case = {"random_full": [], "judge_full": [], "oracle_full": [],
                "random_nondiag": [], "judge_nondiag": [], "oracle_full_nondiag": []}
    n_used = 0
    for case, cands in by_case.items():
        cs = [c for c in cands if c["diag"] is not None]
        if len(cs) < N:
            continue
        n_used += 1
        agg = {k: [] for k in per_case}
        for sub in combinations(cs, N):
            agg["random_full"].append(statistics.mean(c["full"] for c in sub))
            best_diag = max(c["diag"] for c in sub)
            picked = [c for c in sub if c["diag"] == best_diag]
            agg["judge_full"].append(statistics.mean(c["full"] for c in picked))
            orc = max(sub, key=lambda c: c["full"])
            agg["oracle_full"].append(orc["full"])
            nd = [c for c in sub if c["nondiag"] is not None]
            if nd:
                agg["random_nondiag"].append(statistics.mean(c["nondiag"] for c in nd))
                pk_nd = [c for c in picked if c["nondiag"] is not None]
                if pk_nd:
                    agg["judge_nondiag"].append(statistics.mean(c["nondiag"] for c in pk_nd))
                if orc["nondiag"] is not None:
                    agg["oracle_full_nondiag"].append(orc["nondiag"])
        for k in per_case:
            if agg[k]:
                per_case[k].append(statistics.mean(agg[k]))
    return {"n_cases": n_used,
            **{k: round(statistics.mean(v), 4) for k, v in per_case.items() if v}}


def main():
    out = {"note": "Test-time Best-of-N offline simulation; judge=diag[1,2,3] partial-acc; "
                   "nondiag convention excludes selection-signal tasks, no overlap; random expectation always equals N=1; "
                   "candidates are independent generations only (seed_N), K2 re-eval directories (seed_N_rM) removed; "
                   "the three E1 arms (g0/g3b/p7) are restricted to their shared case set.",
           "arms": {}}
    e1_arms = {"g0", "g3b", "p7"}
    e1_cases = None
    loaded = {}
    for arm in ARMS:
        loaded[arm] = load_candidates(arm)
        if arm in e1_arms and loaded[arm]:
            cs = set(loaded[arm])
            e1_cases = cs if e1_cases is None else (e1_cases & cs)
    for arm in ARMS:
        by_case = loaded[arm]
        if arm in e1_arms and e1_cases:
            by_case = {c: v for c, v in by_case.items() if c in e1_cases}
        if not by_case:
            continue
        kmax = max(len(v) for v in by_case.values())
        arm_out = {"max_candidates_per_case": kmax,
                   "cases_total": len(by_case)}
        for N in (1, 2, 4):
            if N > kmax:
                continue
            arm_out[f"N={N}"] = simulate(by_case, N)
        out["arms"][arm] = arm_out
        print(arm, json.dumps(arm_out, ensure_ascii=False))
    (OUT / "bon_simulation.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("WROTE", OUT / "bon_simulation.json")


if __name__ == "__main__":
    main()
