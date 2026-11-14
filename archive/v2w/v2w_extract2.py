#!/usr/bin/env python3
"""V2W extraction patch 2: reconstruct calibrations for verdict-only dirs
(jgen4_dev6, jgen5b2_dev, jgen4_mm3, union_dev) by joining verdicts with
per-arm official K1 summaries via selection_report group order.
Also sanity-check GAN recompute vs headline. Read-only on sources."""
import json, os, glob, itertools, statistics

REPO = "<archive>"
OUT = os.path.join(REPO, "analysis", "paper_extract_0728")

def jload(p):
    with open(p) as f: return json.load(f)

def jlines(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line: out.append(json.loads(line))
    return out

def mean(xs): return sum(xs)/len(xs) if xs else None

summ = jload(os.path.join(OUT, "11_phase2_official_summaries.json"))

def arm_scores(tag, arm_substr):
    """per-instance K1 scores for an arm from summaries."""
    for rel, v in summ[tag].items():
        if arm_substr in rel and v.get("K") == 1:
            return v["per_instance"]
    return None

def reconstruct(dirpath, tag, verdict_glob="verdicts_*.jsonl"):
    sel = glob.glob(os.path.join(dirpath, "selection_report_*.json"))
    vfs = glob.glob(os.path.join(dirpath, verdict_glob))
    if not sel or not vfs: return None, f"missing files in {dirpath}"
    sel = jload(sel[0])
    groups = sel["groups"]
    verd = {v["id"]: v for v in jlines(vfs[0])}
    # pair id pattern: <prefix>_g<idx>_<i><j>
    rows = []
    for gi, g in enumerate(groups):
        arms = g["arms"]
        task, seed = g["task"], g["seed"]
        scores = {}
        for a in arms:
            pi = arm_scores(tag, a + "/") or arm_scores(tag, a)
            if pi is None: return None, f"no K1 summary for arm {a}"
            key = f"{task}|{seed}"
            if key not in pi: return None, f"no score {a} {key}"
            scores[a] = pi[key]
        for i, j in itertools.combinations(range(len(arms)), 2):
            vid = None
            for v in verd:
                if f"_g{gi:03d}_{i}{j}" in v: vid = v; break
            if vid is None: continue
            w = verd[vid]["winner"]
            A, B = arms[i], arms[j]
            rows.append({"id": vid, "task": task, "seed": seed, "A": A, "B": B,
                         "scoreA": scores[A], "scoreB": scores[B],
                         "abs_delta": abs(scores[A]-scores[B]),
                         "truth": "A" if scores[A] >= scores[B] else "B",
                         "judge": w,
                         "correct": (w == ("A" if scores[A] >= scores[B] else "B"))})
    return rows, None

def calib_stats(rows):
    st = {"n": len(rows), "correct": sum(r["correct"] for r in rows)}
    st["acc"] = round(st["correct"]/st["n"], 4) if rows else None
    st["by_thresh"] = {}
    fool = {"0.05": 0, "0.10": 0}
    for th in (0.02, 0.05, 0.10):
        sub = [r for r in rows if r["abs_delta"] > th]
        if sub:
            st["by_thresh"][f">{th}"] = [round(sum(r["correct"] for r in sub)/len(sub), 3), len(sub)]
    for r in rows:
        picked, other = r["judge"], ("B" if r["judge"] == "A" else "A")
        if picked not in ("A", "B"): continue
        d = r[f"score{picked}"] - r[f"score{other}"]
        if d < -0.05: fool["0.05"] += 1
        if d < -0.10: fool["0.10"] += 1
    st["fooled_pairs"] = fool
    return st

def replay(rows, judge_name):
    G, V = {}, {}
    for r in rows:
        g = f"{r['task']}|{r['seed']}"
        G.setdefault(g, {})[f"{g}::{r['A']}"] = r["scoreA"]
        G[g][f"{g}::{r['B']}"] = r["scoreB"]
        V[(f"{g}::{r['A']}", f"{g}::{r['B']}")] = r["judge"]
    res = {}
    npool = max(len(d) for d in G.values())
    def pairwin(a, b):
        v = V.get((a, b))
        if v == "A": return a
        if v == "B": return b
        v = V.get((b, a))
        if v == "A": return b
        if v == "B": return a
        return None
    for pol in (judge_name, "random", "oracle"):
        for N in range(1, npool+1):
            vals_g = []
            for g, sc in G.items():
                pages = sorted(sc)
                if len(pages) < N: continue
                vals = []
                for sub in itertools.combinations(pages, N):
                    if pol == "random" or N == 1:
                        vals.append(mean([sc[p] for p in sub]))
                    elif pol == "oracle":
                        vals.append(max(sc[p] for p in sub))
                    else:
                        pts = {p: 0.0 for p in sub}
                        for a, b in itertools.combinations(sub, 2):
                            w = pairwin(a, b)
                            if w: pts[w] += 1
                        top = max(pts.values())
                        tied = [p for p in sub if pts[p] == top]
                        vals.append(mean([sc[p] for p in tied]))
                vals_g.append(mean(vals))
            if vals_g:
                res.setdefault(pol, {})[f"N={N}"] = round(mean(vals_g), 4)
    return {"n_groups": len(G), "pool_size": npool, "table": res}

out_calib, out_bon = {}, {}
jobs = [
    ("jgen4_dev6", f"{REPO}/analysis/n0725_results/jgen4_dev6", "n0725"),
    ("jgen5b2_dev", f"{REPO}/analysis/n0725_results/jgen5b2_dev", "n0725"),
    ("union_dev", f"{REPO}/analysis/n0725_results/union_dev", "n0725"),
    ("jgen4_mm3", f"{REPO}/analysis/n0726_results/phase2_n0726/jgen4_mm3", "n0726"),
]
errs = {}
for name, d, tag in jobs:
    rows, err = reconstruct(d, tag)
    if err: errs[name] = err; continue
    out_calib[name] = calib_stats(rows)
    out_bon[name] = replay(rows, name)
    with open(os.path.join(OUT, f"33_recon_calib_{name}.json"), "w") as f:
        json.dump(rows, f, indent=1)

with open(os.path.join(OUT, "34_recon_calib_stats.json"), "w") as f:
    json.dump({"calib": out_calib, "bon": out_bon, "errors": errs}, f, indent=1)
print(json.dumps({"calib": out_calib, "errors": errs}, indent=1)[:1500])
print("BON keys:", {k: v["table"].keys() if isinstance(v, dict) else v for k, v in out_bon.items()})
