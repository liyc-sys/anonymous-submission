#!/usr/bin/env python3
"""V2W extraction patch 3: mm3 BoN replay with incomplete-group drop; GAN sanity check."""
import json, os, glob, itertools

REPO = "<archive>"
OUT = os.path.join(REPO, "analysis", "paper_extract_0728")

def jload(p):
    with open(p) as f: return json.load(f)
def jlines(p):
    return [json.loads(l) for l in open(p) if l.strip()]
def mean(xs): return sum(xs)/len(xs) if xs else None

summ = jload(os.path.join(OUT, "11_phase2_official_summaries.json"))

def arm_scores(tag, arm):
    merged = {}
    for rel, v in summ[tag].items():
        if arm in rel and v.get("K") == 1:
            merged.update(v["per_instance"])
    return merged

d = f"{REPO}/analysis/n0726_results/phase2_n0726/jgen4_mm3"
sel = jload(glob.glob(os.path.join(d, "selection_report_*.json"))[0])
verd = {v["id"]: v for v in jlines(glob.glob(os.path.join(d, "verdicts_*.jsonl"))[0])}
rows, dropped = [], []
for gi, g in enumerate(sel["groups"]):
    arms, task, seed = g["arms"], g["task"], g["seed"]
    key = f"{task}|{seed}"
    scores = {}
    ok = True
    for a in arms:
        pi = arm_scores("n0726", a)
        if key not in pi: ok = False; break
        scores[a] = pi[key]
    if not ok:
        dropped.append(key); continue
    for i, j in itertools.combinations(range(len(arms)), 2):
        vid = next((v for v in verd if f"_g{gi:03d}_{i}{j}" in v), None)
        if vid is None: continue
        w = verd[vid]["winner"]
        A, B = arms[i], arms[j]
        rows.append({"id": vid, "task": task, "seed": seed, "A": A, "B": B,
                     "scoreA": scores[A], "scoreB": scores[B],
                     "abs_delta": abs(scores[A]-scores[B]),
                     "truth": "A" if scores[A] >= scores[B] else "B", "judge": w,
                     "correct": (w == ("A" if scores[A] >= scores[B] else "B"))})

def calib_stats(rows):
    st = {"n": len(rows), "correct": sum(r["correct"] for r in rows)}
    st["acc"] = round(st["correct"]/st["n"], 4) if rows else None
    st["by_thresh"] = {}
    fool = {"0.05": 0, "0.10": 0}
    for th in (0.02, 0.05, 0.10):
        sub = [r for r in rows if r["abs_delta"] > th]
        if sub: st["by_thresh"][f">{th}"] = [round(sum(r["correct"] for r in sub)/len(sub), 3), len(sub)]
    for r in rows:
        p_, o_ = r["judge"], ("B" if r["judge"] == "A" else "A")
        if p_ not in ("A", "B"): continue
        dd = r[f"score{p_}"] - r[f"score{o_}"]
        if dd < -0.05: fool["0.05"] += 1
        if dd < -0.10: fool["0.10"] += 1
    st["fooled_pairs"] = fool
    return st

def replay(rows, judge_name):
    G, V = {}, {}
    for r in rows:
        g = f"{r['task']}|{r['seed']}"
        G.setdefault(g, {})[f"{g}::{r['A']}"] = r["scoreA"]
        G[g][f"{g}::{r['B']}"] = r["scoreB"]
        V[(f"{g}::{r['A']}", f"{g}::{r['B']}")] = r["judge"]
    npool = max(len(x) for x in G.values())
    def pairwin(a, b):
        v = V.get((a, b))
        if v == "A": return a
        if v == "B": return b
        v = V.get((b, a))
        if v == "A": return b
        if v == "B": return a
        return None
    res = {}
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
            if vals_g: res.setdefault(pol, {})[f"N={N}"] = round(mean(vals_g), 4)
    return {"n_groups": len(G), "pool_size": npool, "table": res}

mm3 = jload(os.path.join(OUT, "32_bon_mm3_replay.json"))
mm3["jgen4_on_mm3_calibration"] = calib_stats(rows)
mm3["bon_replay"] = replay(rows, "jgen4_mm3")
mm3["dropped_groups_missing_arm_score"] = dropped
mm3["note"] = "gen19bmm3 K1 archived summary has 7/8 valid (eventbrite|seed_1 invalid; the resume mentioned in the archived brief never landed in the summary). Replay uses the 7 complete groups."
with open(os.path.join(OUT, "33_recon_calib_jgen4_mm3.json"), "w") as f:
    json.dump(rows, f, indent=1)
with open(os.path.join(OUT, "32_bon_mm3_replay.json"), "w") as f:
    json.dump(mm3, f, indent=1)
print(json.dumps({"calib": mm3["jgen4_on_mm3_calibration"], "bon": mm3["bon_replay"], "dropped": dropped}, indent=1))

# GAN sanity: recompute vs headline
gan = jload(os.path.join(OUT, "21_gan_rounds_case.json"))
for r, e in gan.items():
    h = e.get("headline", {})
    for k, v in e.items():
        if k.startswith("v_"):
            print(r, k, "recomputed winrate", v["judge_winrate_cand"],
                  "headline", h.get("judge_winrate"), "| fool>0.05", v["fooling_pairs_gt0.05"],
                  "headline fooling", h.get("fooling_pairs"))
