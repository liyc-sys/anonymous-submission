#!/usr/bin/env python3
"""S3 official judge noise package (zero compute, reads only rep_results from dataflow slot JSONs).

Source: all slots with eval_reps>=2 (the same generated site independently judged twice by the official UI-agent judge).
Metrics: (1) site-level re-evaluation noise: |acc_rep0 − acc_rep1| mean/median, re-eval delta sd, all-zero agreement rate;
     (2) pairwise flip rate: fraction of case-seed level winner flips between the rep0 and rep1 verdicts for known A/B batches;
     (3) include the task-level dual-judge flip ledger from plateau_analysis.json (flip 0.218 / YES flip 0.127 / site-level |Δ| 0.076)
       and the §11 calibration constants (σ_judge=0.141, σ_gen=0.194, 8 sites × 3 eval reps + 6 cases × 4 generations, Mac era doc conventions).
Output: paper_assets/webgen/official_judge_noise.json
"""
import json
import re
import statistics
from pathlib import Path

DF = Path("<frozen_corpus>/dataflow/results")
LAB = Path("<frozen_corpus>/lab")
OUT = Path("<frozen_corpus>/paper_assets")
OUT.mkdir(parents=True, exist_ok=True)

PAT = re.compile(r"run_(.+)__(\d{6})__s(\d+)(?:__a(\d+))?$")

# Known same-batch A/B arm pairs (used for per-case-seed pairwise flip rate)
AB_PAIRS = [
    ("final_gen0", "final_gen3b"),
    ("ctrlF", "navF"),
    ("r7fC", "r7fX"),
    ("r18fC", "r18fX"),
    ("pb3", "ns3"),
]


def main():
    reps = {}  # (arm, case, seed) -> [acc_rep0, acc_rep1] (reps from the earliest successful attempt)
    per_arm_n = {}
    for f in DF.glob("run_*.json"):
        m = PAT.match(f.stem)
        if not m:
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        rr = d.get("rep_results") or []
        if d.get("eval_reps", 1) < 2 or len(rr) < 2 or not d.get("launched", True):
            continue
        key = (m.group(1), m.group(2), int(m.group(3)))
        att = int(m.group(4) or 0)
        # Earliest attempt wins: do not overwrite if attempt0 already present
        if key in reps and reps[key][1] <= att:
            continue
        accs = [r.get("acc") for r in rr[:2]]
        if any(a is None for a in accs):
            continue
        reps[key] = (accs, att)
        per_arm_n[m.group(1)] = per_arm_n.get(m.group(1), 0) + 1

    diffs = [abs(v[0][0] - v[0][1]) for v in reps.values()]
    sd_diff = statistics.stdev([v[0][0] - v[0][1] for v in reps.values()])
    site_noise = {
        "n_sites_with_2reps": len(diffs),
        "arms_covered": sorted(per_arm_n),
        "mean_abs_rep_delta": round(statistics.mean(diffs), 4),
        "median_abs_rep_delta": round(statistics.median(diffs), 4),
        "sd_rep_delta": round(sd_diff, 4),
        "sigma_judge_est_from_reps": round(sd_diff / 2 ** 0.5, 4),
        "frac_identical_reps": round(sum(d == 0 for d in diffs) / len(diffs), 4),
        "frac_abs_delta_ge_0.10": round(sum(d >= 0.10 for d in diffs) / len(diffs), 4),
    }

    flips = {}
    pooled_f = pooled_n = 0
    for a, b in AB_PAIRS:
        f_n = t_n = 0
        for (arm, case, seed), (accs, _) in reps.items():
            if arm != a:
                continue
            other = reps.get((b, case, seed))
            if not other:
                continue
            s0 = (accs[0] > other[0][0]) - (accs[0] < other[0][0])
            s1 = (accs[1] > other[0][1]) - (accs[1] < other[0][1])
            t_n += 1
            if s0 != s1:
                f_n += 1
        if t_n:
            flips[f"{a}_vs_{b}"] = {"n_pairs": t_n, "flip_rate": round(f_n / t_n, 4)}
            pooled_f += f_n
            pooled_n += t_n
    pair_flip = {
        "per_batch": flips,
        "pooled": {"n_pairs": pooled_n, "flip_rate": round(pooled_f / pooled_n, 4) if pooled_n else None},
        "note": "fraction of case-seed level winner signs (three-valued, ties included) that disagree between the rep0 and rep1 verdicts",
    }

    plateau = json.loads((LAB / "plateau_analysis.json").read_text())
    out = {
        "site_level_reeval_noise": site_noise,
        "pairwise_flip_rate": pair_flip,
        "task_level_dual_judge": plateau.get("judge_noise"),
        "generation_variance": plateau.get("gen_var"),
        "mac_era_calibration_constants": {
            "sigma_judge": 0.141, "sigma_gen": 0.194,
            "protocol": "8 sites × 3 evaluation reps + 6 cases × 4 independent generations (progress doc §11, 0716)",
            "share": "generation variance 65% vs judge noise 35%",
        },
    }
    (OUT / "official_judge_noise.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({"site": site_noise, "flip_pooled": pair_flip["pooled"]}, ensure_ascii=False, indent=1))
    print("WROTE", OUT / "official_judge_noise.json")


if __name__ == "__main__":
    main()
