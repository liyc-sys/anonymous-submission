#!/usr/bin/env python3
"""Budget judge diagnostic task-subset calibration (zero compute): answer, using per-task labels from the p11-p17 A/B batches,

  Q1 How accurately does the current subset [2,1] predict official arm means?
  Q2 Which 2-3 task subset is better (exhaustive search + leave-one-arm-out validation)?
  Q3 Per-subset agreement rate on paired verdicts (candidate vs parent, who wins)?
  Q4 Is stratification by task type (one per type) more stable?

Data: advjudge/lab/ab_tasks/<arm>/<case>/seed_*/<case>/result.json (the ui array contains task/label/score/idx)
Output: advjudge/lab/judge_calib.json + printed report
"""
import json
import re
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "lab"
SRC = ROOT / "ab_tasks"
LAB2SC = {"YES": 1.0, "PARTIAL": 0.5, "NO": 0.0, "NONE": 0.0}
TYPE_RULES = [("search/filter", r"search|filter|find"),
              ("form/submit", r"form|submit|sign ?up|register|book|apply|contact|save|checkout|subscribe"),
              ("auth/login", r"log ?in|sign ?in|auth|password|account"),
              ("crud/persist", r"add|create|post|comment|reply|delete|edit|update|persist|cart"),
              ("nav/reach", r"navigate|navigation|menu|section|page|link|access"),
              ("viz/chart", r"chart|graph|visual|plot|dashboard"),
              ("content/display", r"display|view|show|detail|list|read|info|image|photo|color|style|theme")]


def task_type(t):
    tl = t.lower()
    for name, pat in TYPE_RULES:
        if re.search(pat, tl):
            return name
    return "other"


def main():
    # arm -> case -> {"seeds": {seed: {"acc": x, "tasks": {idx: score}}}}
    data = defaultdict(lambda: defaultdict(lambda: {"seeds": {}}))
    for f in SRC.rglob("result.json"):
        arm = f.relative_to(SRC).parts[0]
        case = f.parent.name
        seed = f.parents[1].name.replace("seed_", "")
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if not d.get("launched", True) or d.get("acc") is None:
            continue
        tasks = {}
        for u in d.get("ui") or []:
            idx = u.get("idx")
            if idx is None:
                continue
            sc = u.get("score")
            if sc is None:
                sc = LAB2SC.get(u.get("label"), 0.0)
            tasks[int(idx)] = (float(sc), task_type(u.get("task", "")), u.get("task", "")[:60])
        data[arm][case]["seeds"][seed] = {"acc": float(d["acc"]), "tasks": tasks}

    arms = sorted(data)
    cases = sorted({c for a in arms for c in data[a]})
    print(f"arms={len(arms)} cases={cases}")

    # Full task set per case (union of seed idxs across arms)
    case_tasks = {c: sorted({i for a in arms for s in data[a].get(c, {"seeds": {}})["seeds"].values()
                             for i in s["tasks"]}) for c in cases}

    def pred(arm, case, subset):
        """Subset partial-acc (task mean, averaged over seeds first)."""
        vals = []
        for s in data[arm].get(case, {"seeds": {}})["seeds"].values():
            tv = [s["tasks"][i][0] for i in subset if i in s["tasks"]]
            if tv:
                vals.append(statistics.mean(tv))
        return statistics.mean(vals) if vals else None

    def off(arm, case):
        vals = [s["acc"] for s in data[arm].get(case, {"seeds": {}})["seeds"].values()]
        return statistics.mean(vals) if vals else None

    def metrics(subset_by_case):
        """subset_by_case: dict case->subset. Returns (level_spearman, pair_agree, n_pairs, mean_abs_err)."""
        pts, pairs = [], []
        for c in cases:
            offs = {a: off(a, c) for a in arms if off(a, c) is not None}
            prs = {a: pred(a, c, subset_by_case[c]) for a in arms}
            for a in arms:
                if offs.get(a) is not None and prs.get(a) is not None:
                    pts.append((prs[a], offs[a]))
            for a, b in combinations(arms, 2):
                if None in (offs.get(a), offs.get(b), prs.get(a), prs.get(b)):
                    continue
                do, dp = offs[b] - offs[a], prs[b] - prs[a]
                pairs.append((do, dp))
        # spearman (rank)
        def ranks(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            r = [0.0] * len(v)
            for k, i in enumerate(order):
                r[i] = float(k)
            return r
        sp = 0.0
        if len(pts) > 3:
            rx, ry = ranks([p[0] for p in pts]), ranks([p[1] for p in pts])
            mx, my = statistics.mean(rx), statistics.mean(ry)
            cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
            vx = sum((a - mx) ** 2 for a in rx)
            vy = sum((b - my) ** 2 for b in ry)
            sp = cov / (vx * vy) ** 0.5 if vx and vy else 0.0
        agree = sum(1 for do, dp in pairs
                    if (do > 0.01 and dp > 0.01) or (do < -0.01 and dp < -0.01)
                    or (abs(do) <= 0.01 and abs(dp) <= 0.01))
        mae = statistics.mean(abs(dp - do) for do, dp in pairs) if pairs else 0
        return round(sp, 3), round(agree / max(1, len(pairs)), 3), len(pairs), round(mae, 3)

    # Q1: current [2,1]
    cur = {c: [i for i in (2, 1) if i in case_tasks[c]] for c in cases}
    q1 = metrics(cur)
    print(f"[2,1] current: level_spearman={q1[0]} pair_agree={q1[1]} mae={q1[3]} (n_pairs={q1[2]})")

    # Q2: exhaustive search over 2-3 task subsets (per-case independent selection — upper-bound reference only; then test unified subsets)
    # Unified subset: the same idx set applied to all cases (actual judge usage)
    uni_candidates = set()
    for c in cases:
        for k in (1, 2, 3):
            for comb in combinations(case_tasks[c], k):
                uni_candidates.add(comb)
    best = []
    for comb in sorted(uni_candidates):
        sc = metrics({c: [i for i in comb if i in case_tasks[c]] for c in cases})
        if sc[2] >= q1[2] * 0.9:
            best.append((comb, sc))
    best.sort(key=lambda x: (-(x[1][0] + x[1][1]), x[1][3]))
    print("\nTop-5 unified subsets (sorted by spearman+pair_agree):")
    for comb, sc in best[:5]:
        print(f"  {comb}: spearman={sc[0]} pair_agree={sc[1]} mae={sc[3]}")

    # Q3: leave-one-arm-out validation of top subsets vs [2,1]
    def loo(comb):
        sps, pas = [], []
        for held in arms:
            sub_arms = [a for a in arms if a != held]
            pts = [(pred(a, c, [i for i in comb if i in case_tasks[c]]), off(a, c))
                   for a in sub_arms for c in cases
                   if pred(a, c, [i for i in comb if i in case_tasks[c]]) is not None and off(a, c) is not None]
            hp = [(pred(held, c, [i for i in comb if i in case_tasks[c]]), off(held, c))
                  for c in cases
                  if pred(held, c, [i for i in comb if i in case_tasks[c]]) is not None and off(held, c) is not None]
            if len(pts) > 3 and len(hp) >= 2:
                def ranks(v):
                    order = sorted(range(len(v)), key=lambda i: v[i])
                    r = [0.0] * len(v)
                    for k, i in enumerate(order):
                        r[i] = float(k)
                    return r
                allp = pts + hp
                rx, ry = ranks([p[0] for p in allp]), ranks([p[1] for p in allp])
                mx, my = statistics.mean(rx), statistics.mean(ry)
                cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
                vx = sum((a - mx) ** 2 for a in rx)
                vy = sum((b - my) ** 2 for b in ry)
                sps.append(cov / (vx * vy) ** 0.5 if vx and vy else 0.0)
                agree = sum(1 for x in hp if abs(x[0] - x[1]) < 0.3)
                pas.append(agree / len(hp))
        return (round(statistics.mean(sps), 3) if sps else None,
                round(statistics.mean(pas), 3) if pas else None)

    print("\nLeave-one-arm-out validation (level spearman / within-arm fit rate):")
    loo_res = {}
    for comb, sc in best[:3]:
        loo_res[comb] = loo(comb)
        print(f"  {comb}: {loo_res[comb]}")
    loo_cur = loo((1, 2))
    print(f"  [2,1]: {loo_cur}")

    # Q4: stratification by task type (one representative idx per type per case)
    strat = {}
    for c in cases:
        by_type = defaultdict(list)
        for a in arms:
            for s in data[a].get(c, {"seeds": {}})["seeds"].values():
                for i, (sc, tp, _) in s["tasks"].items():
                    if i not in [x for xs in by_type.values() for x in xs]:
                        by_type[tp].append(i)
        strat[c] = sorted({xs[0] for _, xs in sorted(by_type.items()) if xs})
    q4 = metrics(strat)
    print(f"\nTask-type stratified ({[strat[c] for c in cases]}): spearman={q4[0]} pair_agree={q4[1]} mae={q4[3]}")

    out = {"current_[2,1]": {"spearman": q1[0], "pair_agree": q1[1], "mae": q1[3], "loo": loo_cur},
           "top_subsets": [{"subset": list(c), "spearman": s[0], "pair_agree": s[1], "mae": s[3],
                            "loo": loo_res.get(c)} for c, s in best[:5]],
           "stratified": {"subsets": strat, "spearman": q4[0], "pair_agree": q4[1], "mae": q4[3]},
           "n_arms": len(arms), "n_cases": len(cases), "case_tasks": case_tasks}
    (ROOT / "judge_calib.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\nwritten -> advjudge/lab/judge_calib.json")


if __name__ == "__main__":
    main()
