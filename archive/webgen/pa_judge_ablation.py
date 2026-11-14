#!/usr/bin/env python3
"""S2 Judge ablation unified-convention table (zero compute, reads only existing per-task labels).

Unified set = per-task labels of the E1 batch (g0/g3b/p7 × 83 untouched cases × 2 seeds), same source and conventions as e5.
Recompute all budget-judge variants on the same site set:
  naive01      = naive subset [0,1] (no calibration, the naive starting point of 'Evolved single-label without Challenge')
  diag123      = calibrated diagnostic subset [1,2,3] (aggregated-label calibration, no Challenge)
  k3_012       = K=3 subset [0,1,2] (historical row)
  rot_ensemble = rotation over top-3 calibrated subsets {(0,2,3),(1,2,3),(0,1,3)} expectation (Challenge/Goodhart defense form)
  diag123_rescue = [1,2,3] + additionally probe idx0 when diag≤0.1 (r1 fix)
Per-variant metrics: spearman / MAE / MAE_affine (in-sample bias correction) / pair_agree_detie (pairwise within same case)
           / HJLB (fraction with High-Judge≥0.8 and Low-Bench full≤0.4) / zero-kill (judge≤0.1 and full≥0.4).
The Raw VLM (jgen0) row cannot be moved onto the E1 set at zero compute (no stored predictions); its stock exam scores (61 heldout pairs) are included with the convention noted.
Also included: judge_calib.json (648-pair calibration), budget_results.json (judge evolution history + shortlist recall),
         prospective_budget.json, judge_fix_sim.json (affine+rescue offline validation), judge_anatomy.json summary.
Output: paper_assets/webgen/judge_ablation.json
"""
import json
import re
import statistics
from itertools import combinations
from pathlib import Path

RUNS = Path("<frozen_corpus>/runs")
LAB = Path("<frozen_corpus>/lab")
OUT = Path("<frozen_corpus>/paper_assets")
OUT.mkdir(parents=True, exist_ok=True)

ARMS = ["g0", "g3b", "p7"]
VARIANTS = {
    "naive01": [0, 1],
    "diag123": [1, 2, 3],
    "k3_012": [0, 1, 2],
}
ROT_SUBSETS = [(0, 2, 3), (1, 2, 3), (0, 1, 3)]


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
    n = len(x)
    if n < 3:
        return None
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return round(cov / (vx * vy), 4) if vx and vy else None


def load_sites():
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
                if not ui or max(u.get("idx", -1) for u in ui) < 3:
                    continue  # same as e5: require coverage through idx3 (unified site set)
                tasks = {u["idx"]: u["score"] for u in ui if "idx" in u and "score" in u}
                if not all(i in tasks for i in (0, 1, 2, 3)):
                    continue
                rows.append({
                    "arm": arm, "case": case_dir.name, "seed": seed_dir.name,
                    "tasks": tasks,
                    "full": sum(tasks.values()) / len(tasks),
                })
    return rows


def judge_score(tasks, variant):
    if variant == "rot_ensemble":
        vals = []
        for sub in ROT_SUBSETS:
            sv = [tasks[i] for i in sub if i in tasks]
            if sv:
                vals.append(sum(sv) / len(sv))
        return sum(vals) / len(vals) if vals else None
    if variant == "diag123_rescue":
        sv = [tasks[i] for i in (1, 2, 3) if i in tasks]
        base = sum(sv) / len(sv) if sv else None
        if base is not None and base <= 0.1 and 0 in tasks:
            sv2 = sv + [tasks[0]]
            return sum(sv2) / len(sv2)
        return base
    sub = VARIANTS[variant]
    sv = [tasks[i] for i in sub if i in tasks]
    return sum(sv) / len(sv) if sv else None


def metrics(rows, variant):
    js = [judge_score(r["tasks"], variant) for r in rows]
    fs = [r["full"] for r in rows]
    keep = [(j, f, r) for j, f, r in zip(js, fs, rows) if j is not None]
    js = [j for j, _, _ in keep]
    fs = [f for _, f, _ in keep]
    mae = sum(abs(a - b) for a, b in zip(js, fs)) / len(js)
    bias = sum(a - b for a, b in zip(js, fs)) / len(js)
    mae_aff = sum(abs((a - bias) - b) for a, b in zip(js, fs)) / len(js)
    # pairwise within same case (across arms and seeds), official de-tie; judge ties count 0.5 (e5 convention)
    by_case = {}
    for (j, f, r) in keep:
        by_case.setdefault(r["case"], []).append((j, f))
    agree = tot = 0
    for c, rs in by_case.items():
        for (ja, fa), (jb, fb) in combinations(rs, 2):
            df, dj = fa - fb, ja - jb
            if df != 0:
                tot += 1
                if dj == 0:
                    agree += 0.5
                elif (dj > 0) == (df > 0):
                    agree += 1
    hi = [(j, f) for j, f in zip(js, fs) if j >= 0.8]
    hjlb = [x for x in hi if x[1] <= 0.4]
    zero = [(j, f) for j, f in zip(js, fs) if j <= 0.1]
    zkill = [x for x in zero if x[1] >= 0.4]
    return {
        "n_sites": len(js),
        "spearman": spearman(js, fs),
        "mae": round(mae, 4),
        "bias": round(bias, 4),
        "mae_affine_insample": round(mae_aff, 4),
        "pair_agree_detie": round(agree / tot, 4) if tot else None,
        "n_pairs_detie": tot,
        "high_judge_low_bench": {"n_high_judge(>=0.8)": len(hi), "n_HJLB(full<=0.4)": len(hjlb),
                                 "ratio": round(len(hjlb) / len(hi), 4) if hi else None},
        "zero_kill": {"n_judge_zero(<=0.1)": len(zero), "n_good_site(full>=0.4)": len(zkill),
                      "ratio": round(len(zkill) / len(zero), 4) if zero else None},
    }


def main():
    rows = load_sites()
    uni = {}
    for v in ["naive01", "diag123", "k3_012", "rot_ensemble", "diag123_rescue"]:
        uni[v] = metrics(rows, v)
        print(v, json.dumps(uni[v], ensure_ascii=False))
    out = {
        "note": "Unified set = per-task labels of the E1 batch (83 untouched cases, g0/g3b/p7 × 2 seeds); "
                "site inclusion condition: tasks idx0-3 all present (same site set for all variants, horizontally comparable). "
                "The Raw VLM row (jgen0) has no stored E1 predictions; under the zero-compute constraint its 61-pair stock-heldout exam scores are included, a different convention that must be stated separately; "
                "a same-set Raw VLM row would require ~500 offline replay VLM calls.",
        "unified_e1_set": uni,
        "raw_vlm_row_stock_heldout": json.loads((LAB / "metrics_jgen0_heldout.json").read_text()),
        "judge_evolution_history_and_recall": json.loads((LAB / "budget_results.json").read_text()),
        "calibration_648pairs": json.loads((LAB / "judge_calib.json").read_text()),
        "prospective_budget": json.loads((LAB / "prospective_budget.json").read_text()),
        "judge_fix_sim": json.loads((LAB / "judge_fix_sim.json").read_text()),
        "e5_reference": json.loads((LAB / "e5_diag_generalization.json").read_text()),
    }
    ja = LAB / "judge_anatomy.json"
    if ja.exists():
        d = json.loads(ja.read_text())
        out["judge_anatomy_keys"] = list(d.keys()) if isinstance(d, dict) else None
        out["judge_anatomy"] = d
    (OUT / "judge_ablation.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("WROTE", OUT / "judge_ablation.json")


if __name__ == "__main__":
    main()
