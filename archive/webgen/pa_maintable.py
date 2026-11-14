#!/usr/bin/env python3
"""S1 main-table WebGen column numbers package (zero compute, reads only existing results).

Contents: (1) E1 unbiased three arms (directly includes the verified lab/e1_verdict.json)
     (2) §21 cross-model v/m six arms recomputed from dataflow slot JSONs (reproducible artifact)
Conventions: same as ab_verdict.py — earliest successful score (attempt0 launched preferred, else attempt1);
     pairing = per-case difference of seed-mean acc; W/L threshold ±0.01; CI95 uses 1.96*SE (consistent with e1_verdict).
Output: paper_assets/webgen/maintable.json
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


def load_arms(arms):
    res = {}
    for f in DF.glob("run_*.json"):
        m = PAT.match(f.stem)
        if not m or m.group(1) not in arms:
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        res[(m.group(1), m.group(2), int(m.group(3)), int(m.group(4) or 0))] = d
    return res


def slot(res, arm, case, seed):
    d0 = res.get((arm, case, seed, 0))
    d1 = res.get((arm, case, seed, 1))
    for d in (d0, d1):
        if d is not None and d.get("acc") is not None and d.get("launched", True):
            return d, False
    return (d0 or d1), True


def cases_of(res, arm):
    return sorted({k[1] for k in res if k[0] == arm})


def seeds_of(res, arm, case):
    return sorted({k[2] for k in res if k[0] == arm and k[1] == case})


def case_mean(res, arm, case):
    vals = []
    for s in seeds_of(res, arm, case):
        d, _ = slot(res, arm, case, s)
        if d and d.get("acc") is not None:
            vals.append(d["acc"])
    return statistics.mean(vals) if vals else None


def arm_stats(res, arm):
    cvals, app = [], []
    crash_init = retried = residual = 0
    for c in cases_of(res, arm):
        for s in seeds_of(res, arm, c):
            d0 = res.get((arm, c, s, 0))
            d1 = res.get((arm, c, s, 1))
            if d0 is not None and not d0.get("launched", True):
                crash_init += 1
            if d1 is not None:
                retried += 1
            _, cr = slot(res, arm, c, s)
            if cr:
                residual += 1
        cm = case_mean(res, arm, c)
        if cm is not None:
            cvals.append(cm)
    for k, d in res.items():
        if k[0] == arm and d.get("appearance") is not None:
            app.append(d["appearance"])
    return {
        "n_cases": len(cvals),
        "mean_acc": round(statistics.mean(cvals), 4) if cvals else None,
        "median_acc": round(statistics.median(cvals), 4) if cvals else None,
        "mean_appearance": round(statistics.mean(app), 3) if app else None,
        "crash_initial": crash_init,
        "retried_slots": retried,
        "crash_residual": residual,
    }


def paired(res, arm_b, arm_a):
    """B − A, per-case."""
    cases = sorted(set(cases_of(res, arm_a)) & set(cases_of(res, arm_b)))
    diffs, detail = [], {}
    for c in cases:
        ma, mb = case_mean(res, arm_a, c), case_mean(res, arm_b, c)
        if ma is None or mb is None:
            continue
        diffs.append(mb - ma)
        detail[c] = round(mb - ma, 4)
    n = len(diffs)
    if n < 2:
        return {"pair": f"{arm_b}-{arm_a}", "n_cases": n}
    m = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    se = sd / n ** 0.5
    return {
        "pair": f"{arm_b}-{arm_a}",
        "n_cases": n,
        "mean_delta": round(m, 4),
        "median_delta": round(statistics.median(diffs), 4),
        "sd": round(sd, 4),
        "t": round(m / se, 3) if se else None,
        "ci95": [round(m - 1.96 * se, 4), round(m + 1.96 * se, 4)],
        "W/L/T": "%d/%d/%d" % (
            sum(d > 0.01 for d in diffs),
            sum(d < -0.01 for d in diffs),
            sum(abs(d) <= 0.01 for d in diffs),
        ),
        "per_case": detail,
    }


def main():
    out = {
        "note": "Main-table WebGen column numbers package; all recomputed/included offline, no new experiments. "
                "actor=Qwen3.5-9B; v=Qwen3-VL-8B-Thinking; m=Gemma-4-26B-A4B. "
                "E1=83 untouched cases × 2 seeds unbiased convention; cross-model=12 heldout cases (v 2 seeds / m up to 4 seeds).",
        "method_mapping": {
            "Minimal": "gen0 (~160-line minimal scaffold)",
            "Actor-only Evolution": "gen3b (Phase1 pure actor rolling-generation champion)",
            "Full (judge-assisted)": "p7_emitatomicit (Phase2 judge-screened wide-search transferred champion)",
            "Heavy": "bolt.diy reference ledger (run by a colleague 2026-06 on Windows, q9b + official bolt.diy full-set acc≈0.008, "
                     "620/647 NO, main death cause=sites fail to come up; different evaluator, reference only, not same protocol)",
        },
        "nine_b_unbiased_e1": json.loads((LAB / "e1_verdict.json").read_text()),
    }
    for model_tag, (a0, a3b, ap7) in {
        "qwen3vl8b_v": ("v0", "v3b", "vp7"),
        "gemma26b_m": ("m0", "m3b", "mp7"),
    }.items():
        res = load_arms({a0, a3b, ap7})
        out[model_tag] = {
            "arms": {a: arm_stats(res, a) for a in (a0, a3b, ap7)},
            "pairs": {
                "actor_only_vs_minimal": paired(res, a3b, a0),
                "full_vs_actor_only": paired(res, ap7, a3b),
                "full_vs_minimal": paired(res, ap7, a0),
            },
        }
    (OUT / "maintable.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    # summary print
    print("== 9B E1 (83 case unbiased) ==")
    print(json.dumps(out["nine_b_unbiased_e1"]["arms"], ensure_ascii=False))
    for k in ("qwen3vl8b_v", "gemma26b_m"):
        print(f"== {k} ==")
        for a, s in out[k]["arms"].items():
            print(f"  {a}: mean={s['mean_acc']} n_cases={s['n_cases']} crash i/r/res="
                  f"{s['crash_initial']}/{s['retried_slots']}/{s['crash_residual']}")
        for pk, p in out[k]["pairs"].items():
            print(f"  {pk}: Δ={p.get('mean_delta')} t={p.get('t')} W/L/T={p.get('W/L/T')} n={p.get('n_cases')}")
    print("WROTE", OUT / "maintable.json")


if __name__ == "__main__":
    main()
