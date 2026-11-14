#!/usr/bin/env python3
"""Judge fix offline simulation (zero slots, using E1 519-site data):
(1) improvement of affine bias correction on the absolute convention; (2) improvement of a one-probe rescue for diag≈0 sites on pair_agree / ranking mis-kills."""
import itertools
import json
import statistics
from collections import defaultdict
from pathlib import Path

RUNS = Path("<frozen_corpus>/runs")
DIAG = (1, 2, 3)
RESCUE_IDX = 0  # probe idx 0 (easiest 0.573, correlation with full 0.567)


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
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else None


def pair_agree(items, key):
    by_case = defaultdict(list)
    for s in items:
        by_case[s["case"]].append(s)
    agree = tot = 0
    for c, rs in by_case.items():
        for a, b in itertools.combinations(rs, 2):
            df = a["full"] - b["full"]
            dd = a[key] - b[key]
            if abs(df) >= 0.05:
                tot += 1
                if dd == 0:
                    agree += 0.5
                elif (dd > 0) == (df > 0):
                    agree += 1
    return round(agree / tot, 4) if tot else None, tot


def main():
    sites = []
    for arm in ["g0", "g3b", "p7"]:
        for rj in sorted(RUNS.glob(f"{arm}/*/seed_*/*/result.json")):
            try:
                r = json.loads(rj.read_text())
            except Exception:
                continue
            ui = r.get("ui") or []
            if not ui:
                continue
            scores = {u["idx"]: u["score"] for u in ui}
            if not all(i in scores for i in DIAG) or RESCUE_IDX not in scores:
                continue
            full = sum(scores.values()) / len(scores)
            diag = sum(scores[i] for i in DIAG) / len(DIAG)
            rescued = (sum(scores[i] for i in DIAG) + scores[RESCUE_IDX]) / (len(DIAG) + 1)
            sites.append({"case": rj.parts[-2], "full": full, "diag": diag,
                          "rescued": rescued, "idx0": scores[RESCUE_IDX]})
    print("sites:", len(sites))

    # (1) affine correction
    bias = statistics.mean(s["diag"] - s["full"] for s in sites)
    mae_raw = statistics.mean(abs(s["diag"] - s["full"]) for s in sites)
    mae_cal = statistics.mean(abs(s["diag"] - bias - s["full"]) for s in sites)
    print(f"(1) affine: bias={bias:+.4f}  MAE raw={mae_raw:.4f} -> calibrated={mae_cal:.4f}")

    # (2) diag≈0 probe rescue
    zero = [s for s in sites if s["diag"] <= 0.1]
    print(f"(2) sites with diag<=0.1: {len(zero)} ({len(zero)/len(sites):.1%}), among them good sites with full>=0.4 (mis-kill targets): "
          f"{sum(1 for s in zero if s['full'] >= 0.4)}")
    # score distribution of these sites after rescue
    saved = [s for s in zero if s["rescued"] > 0.1]
    print(f"   escaped the zero zone after probing idx{RESCUE_IDX}: {len(saved)} ({len(saved)/max(1,len(zero)):.0%})")
    # pair_agree comparison: raw diag vs (rescue score when diag<=0.1, else diag)
    for s in sites:
        s["hybrid"] = s["rescued"] if s["diag"] <= 0.1 else s["diag"]
    pa_raw, n1 = pair_agree(sites, "diag")
    pa_hyb, n2 = pair_agree(sites, "hybrid")
    print(f"   pair_agree: raw {pa_raw} (n={n1}) -> rescue hybrid {pa_hyb} (n={n2})")
    sp_raw = spearman([s["diag"] for s in sites], [s["full"] for s in sites])
    sp_hyb = spearman([s["hybrid"] for s in sites], [s["full"] for s in sites])
    print(f"   spearman: raw {sp_raw:.4f} -> rescue hybrid {sp_hyb:.4f}")
    # ranking mis-kill simulation: within a case, raw judge ranks the full-best site below a non-best; does rescue fix it
    by_case = defaultdict(list)
    for s in sites:
        by_case[s["case"]].append(s)
    err = fix = newerr = 0
    for c, rs in by_case.items():
        if len(rs) < 2:
            continue
        best_full = max(rs, key=lambda s: s["full"])
        raw_top = max(rs, key=lambda s: s["diag"])
        hyb_top = max(rs, key=lambda s: s["hybrid"])
        if raw_top is not best_full:
            err += 1
            if hyb_top is best_full:
                fix += 1
        elif hyb_top is not best_full:
            newerr += 1
    print(f"   top-1 mis-kills (case-best site not ranked first): raw {err} cases, rescue fixed {fix}, rescue new mis-kills {newerr}")
    out = {"affine_bias": round(bias, 4), "mae_raw": round(mae_raw, 4), "mae_calibrated": round(mae_cal, 4),
           "zero_sites": len(zero), "zero_good_sites": sum(1 for s in zero if s["full"] >= 0.4),
           "rescued_out": len(saved), "pair_agree_raw": pa_raw, "pair_agree_hybrid": pa_hyb,
           "spearman_raw": round(sp_raw, 4), "spearman_hybrid": round(sp_hyb, 4),
           "top1_errors_raw": err, "top1_fixed": fix, "top1_new_errors": newerr}
    Path("<frozen_corpus>/lab/judge_fix_sim.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print("saved advjudge/lab/judge_fix_sim.json")


if __name__ == "__main__":
    main()
