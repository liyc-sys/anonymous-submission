#!/usr/bin/env python3
"""E5: generalization validation of the diagnostic subset [1,2,3] on 83 untouched cases (for the paper, offline).

Input: per-task labels from the E1 batch runs/{g0,g3b,p7}/<case>/seed_<n>/<case>/result.json.
Output: advjudge/lab/e5_diag_generalization.json + printed summary.
Conventions aligned with judge_calib.json: spearman / pair_agree (detie removes ties, ε=0.05 variant also reported) / MAE.
"""
import json
import re
import statistics
from itertools import combinations
from pathlib import Path

RUNS = Path("<frozen_corpus>/runs")
OUT = Path("<frozen_corpus>/lab/e5_diag_generalization.json")
DIAG = [1, 2, 3]
ARMS = ["g0", "g3b", "p7"]


def spearman(x, y):
    n = len(x)
    if n < 3:
        return None
    rx = statistics.rankdata(x) if hasattr(statistics, "rankdata") else None
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
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return round(cov / (vx * vy), 4) if vx and vy else None


def main():
    rows = []
    for arm in ARMS:
        for case_dir in sorted((RUNS / arm).glob("*")):
            if not case_dir.is_dir() or not re.fullmatch(r"\d{6}", case_dir.name):
                continue
            for seed_dir in sorted(case_dir.glob("seed_*")):
                rj = seed_dir / case_dir.name / "result.json"
                if not rj.exists():
                    continue
                try:
                    r = json.loads(rj.read_text())
                except Exception:
                    continue
                ui = r.get("ui") or []
                if not ui or max(u.get("idx", -1) for u in ui) < max(DIAG):
                    continue
                full = [u["score"] for u in ui]
                diag = [u["score"] for u in ui if u.get("idx") in DIAG]
                if not diag:
                    continue
                rows.append({"arm": arm, "case": case_dir.name,
                             "seed": seed_dir.name,
                             "full": sum(full) / len(full),
                             "diag": sum(diag) / len(diag),
                             "n_tasks": len(ui)})
    full_v = [r["full"] for r in rows]
    diag_v = [r["diag"] for r in rows]
    mae = round(sum(abs(a - b) for a, b in zip(full_v, diag_v)) / len(rows), 4) if rows else None

    # pairwise within same case: detie pair_agree (+ ε=0.05 variant)
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case"], []).append(r)
    agree = tot = agree_e = tot_e = 0
    for c, rs in by_case.items():
        for a, b in combinations(rs, 2):
            df = a["full"] - b["full"]
            dd = a["diag"] - b["diag"]
            if df != 0:
                tot += 1
                if (dd > 0) == (df > 0) or dd == 0:
                    agree += 1 if dd != 0 else 0.5
            if abs(df) >= 0.05:
                tot_e += 1
                if (dd > 0) == (df > 0) or dd == 0:
                    agree_e += 1 if dd != 0 else 0.5
    out = {
        "n_sites": len(rows), "n_cases": len(by_case),
        "spearman": spearman(diag_v, full_v),
        "mae": mae,
        "pair_agree_detie": round(agree / tot, 4) if tot else None,
        "n_pairs_detie": tot,
        "pair_agree_eps05": round(agree_e / tot_e, 4) if tot_e else None,
        "n_pairs_eps05": tot_e,
        "calibration_reference": {"spearman": 0.915, "pair_agree": 0.809,
                                  "mae": 0.115, "note": "19 arms x 4 train cases"},
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
