#!/usr/bin/env python3
"""V2W paper extraction (2026-07-28) — zero-compute, pure archival extraction.

Reads frozen experiment archives, recomputes offline-replayable metrics,
writes everything to analysis/paper_extract_0728/ in the phase2 work repo.
NO model calls, NO generation, NO official evaluation. Read-only on sources.
"""
import json, os, re, shutil, statistics, itertools, glob, sys

EVO = "<frozen_corpus>"
REPO = "<archive>"
OUT = os.path.join(REPO, "analysis", "paper_extract_0728")
os.makedirs(OUT, exist_ok=True)

def jload(p):
    with open(p) as f: return json.load(f)

def jlines(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try: out.append(json.loads(line))
                except json.JSONDecodeError: pass
    return out

def jdump(obj, name):
    p = os.path.join(OUT, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f: json.dump(obj, f, indent=1, ensure_ascii=False)
    print("WROTE", name)

def mean(xs): return sum(xs)/len(xs) if xs else None

MISSING = []
def need(p):
    if not os.path.exists(p):
        MISSING.append(p); return False
    return True

# ---------------------------------------------------------------- 0. sources
SRC = os.path.join(OUT, "sources")
os.makedirs(SRC, exist_ok=True)
copy_list = [
    (f"{EVO}/history.md", "history.md"),
    (f"{EVO}/reports", "reports"),
]
for src, dst in copy_list:
    d = os.path.join(SRC, dst)
    if os.path.isdir(src):
        shutil.copytree(src, d, dirs_exist_ok=True)
    elif need(src):
        shutil.copy2(src, d)
print("sources copied")

# ------------------------------------------------- 1. Phase I evidence table
# per-generation fitness batches with per-run scores + agent call accounting
phase1 = {}
for p in sorted(glob.glob(f"{EVO}/evidence/*.json")):
    name = os.path.basename(p)[:-5]
    try: d = jload(p)
    except Exception as e:
        MISSING.append(f"{p} (parse: {e})"); continue
    runs = d.get("runs", [])
    scores = [r["score"] for r in runs if isinstance(r.get("score"), (int, float))]
    calls = [r.get("agent", {}).get("calls") for r in runs]
    calls = [c for c in calls if isinstance(c, (int, float))]
    secs = [r.get("agent", {}).get("seconds") for r in runs]
    secs = [s for s in secs if isinstance(s, (int, float))]
    phase1[name] = {
        "fitness_recorded": d.get("fitness"),
        "n_runs": len(runs),
        "mean_score": round(mean(scores), 4) if scores else None,
        "median_score": round(statistics.median(scores), 4) if scores else None,
        "sd": round(statistics.pstdev(scores), 4) if len(scores) > 1 else None,
        "crash_lt_0.05": sum(1 for s in scores if s < 0.05),
        "actor_llm_calls_mean": round(mean(calls), 2) if calls else None,
        "actor_llm_calls_total": sum(calls) if calls else None,
        "seconds_mean": round(mean(secs), 1) if secs else None,
        "mtime": os.path.getmtime(p),
        "per_run": [{"task": r.get("task"), "seed": r.get("seed"),
                     "score": r.get("score"), "viewports": r.get("viewports"),
                     "calls": r.get("agent", {}).get("calls")} for r in runs],
    }
jdump(phase1, "10_phase1_evidence_batches.json")

# ---------------------------------------------- 2. judge lineage accuracies
JL = f"{EVO}/judge_lab"
answers = {}
for split in ("train", "heldout"):
    p = f"{JL}/answers_{split}.json"
    if need(p): answers[split] = jload(p)

lineage = {}
for vfile in sorted(glob.glob(f"{JL}/v_jgen*_train.jsonl") + glob.glob(f"{JL}/v_jgen*_heldout.jsonl")):
    base = os.path.basename(vfile)[2:-6]          # jgenX_split
    m = re.match(r"(jgen[0-9b]+(?:seed)?)_(train|heldout)", base)
    if not m: continue
    judge, split = m.groups()
    if split not in answers: continue
    ans = answers[split]
    verd = {v["id"]: v.get("winner") for v in jlines(vfile)}
    per_bin = {}
    n_ok = n_tot = n_tie = 0
    for pid, a in ans.items():
        if pid not in verd: continue
        w = verd[pid]
        n_tot += 1
        b = a.get("bin", "?")
        per_bin.setdefault(b, [0, 0])
        per_bin[b][1] += 1
        if w not in ("A", "B"):
            n_tie += 1; continue
        if w == a["winner"]:
            n_ok += 1; per_bin[b][0] += 1
    if n_tot == 0: continue
    lineage.setdefault(judge, {})[split] = {
        "n": n_tot, "correct": n_ok, "acc": round(n_ok/n_tot, 4),
        "abstain_or_tie": n_tie,
        "acc_decided": round(n_ok/(n_tot-n_tie), 4) if n_tot > n_tie else None,
        "per_bin_acc": {b: [round(c/t, 3), t] for b, (c, t) in sorted(per_bin.items())},
    }
jdump(lineage, "20_judge_lineage_acc.json")

# ------------------------------------------------------- 3. GAN rounds (case)
gan = {}
for r in ("ganR1", "ganR2", "ganR3"):
    entry = {}
    p = f"{JL}/gan_{r}_result.json"
    if need(p): entry["headline"] = jload(p)
    kp, vps = f"{JL}/key_{r}.json", [f"{JL}/v_{r}.jsonl"]
    if r == "ganR2": vps.append(f"{JL}/v_ganR2_jgen2.jsonl")
    if need(kp):
        key = jload(kp)
        for vp in vps:
            if not need(vp): continue
            tag = os.path.basename(vp)[:-6]
            verd = {v["id"]: v.get("winner") for v in jlines(vp)}
            n = win = fool10 = fool5 = 0
            rows = []
            for pid, k in key.items():
                if pid not in verd: continue
                w = verd[pid]
                cand_side = k["cand"]
                picked_cand = (w == cand_side)
                delta = k["cand_score"] - k["ref_score"]
                n += 1; win += picked_cand
                if picked_cand and delta < -0.10: fool10 += 1
                if picked_cand and delta < -0.05: fool5 += 1
                rows.append({"pid": pid, "task_seed": k.get("k"),
                             "judge_picked_cand": picked_cand,
                             "official_delta": round(delta, 4)})
            entry[tag] = {"n": n, "judge_winrate_cand": round(win/n, 4) if n else None,
                          "official_delta_mean": round(mean([r_["official_delta"] for r_ in rows]), 4),
                          "fooling_pairs_gt0.05": fool5, "fooling_pairs_gt0.10": fool10,
                          "pairs": rows}
    gan[r] = entry
jdump(gan, "21_gan_rounds_case.json")
case_dir = os.path.join(OUT, "60_case_gen13b")
os.makedirs(case_dir, exist_ok=True)
for r in ("ganR1", "ganR2", "ganR3"):
    for pref in ("gan_%s_result.json", "key_%s.json", "exam_%s.json"):
        p = f"{JL}/{pref % r}"
        if os.path.exists(p): shutil.copy2(p, case_dir)
    for vp in (f"v_{r}.jsonl", "v_ganR2_jgen2.jsonl"):
        p = f"{JL}/{vp}"
        if os.path.exists(p): shutil.copy2(p, case_dir)

# --------------------------------------- 4. night3 BoN replay (24 groups x4)
# pages officially scored in evidence/lad*.json ; candidates = 4 seeds per arm/task
lad_scores = {}
for arm in ("lad0", "lad8b", "lad9", "lad10", "lad11b", "lad12"):
    p = f"{EVO}/evidence/{arm}.json"
    if not need(p): continue
    for r in jload(p).get("runs", []):
        if isinstance(r.get("score"), (int, float)):
            lad_scores[f"{arm}__{r['task']}__{r['seed']}"] = r["score"]

def replay_bon(groups, verdicts_by_judge):
    """groups: {gkey: {page_id: score}}; verdicts: judge -> {(a,b): 'A'|'B'|other}"""
    res = {}
    gkeys = sorted(groups)
    def pairwin(judge, a, b):
        v = verdicts_by_judge[judge].get((a, b))
        if v == "A": return a
        if v == "B": return b
        v = verdicts_by_judge[judge].get((b, a))
        if v == "A": return b
        if v == "B": return a
        return None
    for judge in list(verdicts_by_judge) + ["random", "oracle"]:
        for N in (1, 2, 3, 4):
            picks = []
            for g in gkeys:
                pages = sorted(groups[g])
                if len(pages) < N: continue
                subs = list(itertools.combinations(pages, N))
                vals = []
                for sub in subs:
                    sc = {p_: groups[g][p_] for p_ in sub}
                    if judge == "random" or N == 1:
                        vals.append(mean(list(sc.values())))
                    elif judge == "oracle":
                        vals.append(max(sc.values()))
                    else:
                        pts = {p_: 0.0 for p_ in sub}
                        undecided = False
                        for a, b in itertools.combinations(sub, 2):
                            w = pairwin(judge, a, b)
                            if w is None: undecided = True
                            else: pts[w] += 1
                        if undecided and N > 1:
                            vals.append(mean(list(sc.values()))); continue
                        top = max(pts.values())
                        tied = [p_ for p_ in sub if pts[p_] == top]
                        vals.append(mean([sc[p_] for p_ in tied]))
                picks.append(mean(vals))
            if picks:
                res.setdefault(judge, {})[f"N={N}"] = round(mean(picks), 4)
    return {"n_groups": len(gkeys), "table": res}

bon_groups = {}
for pid, s in lad_scores.items():
    arm, task, seed = pid.split("__")
    bon_groups.setdefault(f"{arm}|{task}", {})[pid] = s
bon_groups = {g: d for g, d in bon_groups.items() if len(d) == 4}
bon_verd = {}
for j in ("jgen0", "jgen1", "jgen2"):
    p = f"{JL}/v_bon_{j}.jsonl"
    ep = f"{JL}/exam_bon_{j}.json"
    if not (need(p) and need(ep)): continue
    exam = {e["id"]: e for e in jload(ep)}
    vd = {}
    for v in jlines(p):
        e = exam.get(v["id"])
        if e: vd[(e["A"], e["B"])] = v.get("winner")
    bon_verd[j] = vd
night3_bon = replay_bon(bon_groups, bon_verd)
jdump(night3_bon, "30_bon_night3_replay.json")

# ----------------------------- 5. Phase II dev-pool BoN + judge calib replay
def load_calib(path):
    rows = jload(path)
    groups, verd = {}, {}
    stats_ = {"n": 0, "correct": 0, "by_thresh": {}}
    fool = {"0.05": 0, "0.10": 0}
    for r in rows:
        g = f"{r['task']}|{r['seed']}"
        groups.setdefault(g, {})[r["A"]] = r["scoreA"]
        groups[g][r["B"]] = r["scoreB"]
        verd[(g, r["A"], r["B"])] = r["judge"]
        stats_["n"] += 1
        stats_["correct"] += bool(r.get("correct"))
        for th in (0.02, 0.05, 0.10):
            if r["abs_delta"] > th:
                k = f">{th}"
                stats_["by_thresh"].setdefault(k, [0, 0])
                stats_["by_thresh"][k][1] += 1
                stats_["by_thresh"][k][0] += bool(r.get("correct"))
        # judge-picked side officially worse => fooled
        picked = r["judge"]; other = "B" if picked == "A" else "A"
        d = r[f"score{picked}"] - r[f"score{other}"]
        if d < -0.05: fool["0.05"] += 1
        if d < -0.10: fool["0.10"] += 1
    stats_["acc"] = round(stats_["correct"]/stats_["n"], 4) if stats_["n"] else None
    stats_["by_thresh"] = {k: [round(c/t, 3), t] for k, (c, t) in stats_["by_thresh"].items()}
    stats_["fooled_pairs"] = fool
    return groups, verd, stats_

calib_sets = {}
for label, path in {
    "jgen4_dev48": f"{REPO}/analysis/n0725_results/jgen4_dev/verdicts_jgen4_sealed.calibration.json",
    "jgen5b1_dev48": f"{REPO}/analysis/n0725_results/jgen5b1_dev/verdicts_jgen5b1_sealed.calibration.json",
    "jgen5b2_dev48": f"{REPO}/analysis/n0725_results/jgen5b2_dev/verdicts_jgen5b2_sealed.calibration.json",
    "jgen4_dev6_120": f"{REPO}/analysis/n0725_results/jgen4_dev6/verdicts_jgen4_sealed.calibration.json",
}.items():
    hits = glob.glob(path) or glob.glob(os.path.dirname(path) + "/*.calibration.json")
    if hits: calib_sets[label] = hits[0]
    else: MISSING.append(path)
for d in sorted(glob.glob(f"{REPO}/analysis/n0726_results/phase2_n0725/jgen6*_dev")):
    hits = glob.glob(d + "/*.calibration.json")
    if hits: calib_sets[os.path.basename(d)] = hits[0]

calib_out, dev_bon = {}, {}
for label, path in calib_sets.items():
    groups, verd, st = load_calib(path)
    calib_out[label] = st
    vd = {(a, b): w for (g, a, b), w in verd.items()}
    # regroup with group-qualified page ids to avoid arm-name collisions
    G = {}
    for (g, a, b), w in verd.items():
        G.setdefault(g, {})
    for g in G:
        for (gg, a, b), w in verd.items():
            if gg == g:
                pass
    # simpler: rebuild groups directly
    G = {}
    V = {}
    rows = jload(path)
    for r in rows:
        g = f"{r['task']}|{r['seed']}"
        G.setdefault(g, {})[f"{g}::{r['A']}"] = r["scoreA"]
        G[g][f"{g}::{r['B']}"] = r["scoreB"]
        V[(f"{g}::{r['A']}", f"{g}::{r['B']}")] = r["judge"]
    dev_bon[label] = replay_bon(G, {label: V})
jdump(calib_out, "22_phase2_judge_calibration.json")
jdump(dev_bon, "31_bon_phase2_dev_replay.json")

# ------------------------------------------------ 6. n0725/n0726 summaries
def collect_summaries(root):
    out = {}
    for p in sorted(glob.glob(f"{root}/**/*summary*.json", recursive=True)):
        try: d = jload(p)
        except Exception: continue
        if isinstance(d, dict) and "mean" in d:
            out[os.path.relpath(p, root)] = {
                "label": d.get("label"), "K": d.get("K"), "jobs": d.get("jobs"),
                "valid": d.get("valid"), "mean": d.get("mean"),
                "per_instance": {f"{r['task']}|{r['seed']}": r["score"]
                                 for r in d.get("results", []) if r.get("valid")}}
    return out

summ = {}
for tag, root in {
    "n0725": f"{REPO}/analysis/n0725_results",
    "n0726": f"{REPO}/analysis/n0726_results",
    "round1": f"{REPO}/analysis/round1_results",
    "round2": f"{REPO}/analysis/round2_results",
    "round3": f"{REPO}/analysis/round3_results",
}.items():
    if os.path.isdir(root): summ[tag] = collect_summaries(root)
jdump(summ, "11_phase2_official_summaries.json")

# mm3 pool replay: arms gen10/19b/19c/19d mm3 K1 + jgen4_mm3 verdicts
mm3_dir = f"{REPO}/analysis/n0726_results"
mm3_arms = {}
for arm in ("n0726_gen10mm3", "n0726_gen19bmm3", "n0726_gen19cmm3", "n0726_gen19dmm3", "n0726_gen0mm3"):
    hits = glob.glob(f"{mm3_dir}/{arm}/*k1_summary.json")
    if hits:
        d = jload(hits[0])
        mm3_arms[arm] = {f"{r['task']}|{r['seed']}": r["score"]
                         for r in d.get("results", []) if r.get("valid")}
mm3_calib = glob.glob(f"{mm3_dir}/phase2_n0726/jgen4_mm3/*.calibration.json") \
          + glob.glob(f"{mm3_dir}/phase2_n0726/jgen4_mm3/*calibration*.json")
mm3_out = {"arm_means": {a: round(mean(list(v.values())), 4) for a, v in mm3_arms.items() if v}}
if mm3_calib:
    G, V = {}, {}
    for r in jload(mm3_calib[0]):
        g = f"{r['task']}|{r['seed']}"
        G.setdefault(g, {})[f"{g}::{r['A']}"] = r["scoreA"]
        G[g][f"{g}::{r['B']}"] = r["scoreB"]
        V[(f"{g}::{r['A']}", f"{g}::{r['B']}")] = r["judge"]
    mm3_out["bon_replay"] = replay_bon(G, {"jgen4_mm3": V})
    _, _, st = load_calib(mm3_calib[0])
    mm3_out["jgen4_on_mm3_calibration"] = st
else:
    ls = glob.glob(f"{mm3_dir}/phase2_n0726/jgen4_mm3/*")
    mm3_out["jgen4_mm3_dir_listing"] = [os.path.basename(x) for x in ls]
jdump(mm3_out, "32_bon_mm3_replay.json")

# --------------------------------------------------------- 7. noise metrics
noise = {"documented": {
    "sigma_rep_viewport": 0.0906, "sigma_K1": 0.052, "sigma_K2": 0.037,
    "sigma_K4": 0.026, "same_harness_regen_sigma": 0.06,
    "source": "archived campaign analysis notes"}}
# same-composite repeat K1 (union vs jgen5b1 had identical picks)
u = summ.get("n0725", {})
def find_pi(sub):
    for k, v in u.items():
        if sub in k and v.get("K") == 1:
            return v["per_instance"]
    return None
a, b = find_pi("comp4_union"), find_pi("comp4_jgen5b1")
if a and b:
    common = sorted(set(a) & set(b))
    diffs = [a[k] - b[k] for k in common]
    noise["composite_repeat_K1"] = {
        "n": len(common), "mean_abs_diff": round(mean([abs(d) for d in diffs]), 4),
        "sd_diff": round(statistics.pstdev(diffs), 4) if len(diffs) > 1 else None,
        "max_abs": round(max(abs(d) for d in diffs), 4),
        "note": "identical composite pages evaluated under two labels (union==b1 picks)"}
# same-arm two-draw (generation+eval variance)
for arm in ("gen10", "gen19b"):
    a, b = find_pi(f"n0725_{arm}/") or find_pi(f"n0725_{arm}\\"), find_pi(f"{arm}_r2")
    if a and b:
        common = sorted(set(a) & set(b))
        diffs = [a[k] - b[k] for k in common]
        if diffs:
            noise[f"same_arm_redraw_{arm}"] = {
                "n": len(common),
                "sd_diff": round(statistics.pstdev(diffs), 4) if len(diffs) > 1 else None,
                "mean_diff": round(mean(diffs), 4)}
jdump(noise, "40_noise_metrics.json")

# ------------------------------------------- 8. official call accounting
acct = []
K_BY_BATCH = {  # documented K per batch family (from history.md / closeouts)
    "gen0": 2, "gen1": 2, "gen2": 2, "gen3": 2, "gen5a": 2, "gen5b": 2,
    "gen6": 2, "gen7": 2, "ctrlA": 2, "ctrlB": 2, "ctrlC": 2, "ctrlD": 2,
    "ctrlE": 2, "gen8": 2, "gen8b": 2, "gen9": 2, "gen9b": 2, "gen10": 2,
    "gen10b": 2, "gen11": 2, "gen11b": 2, "gen12": 2, "gen12b": 2,
    "gen12conf": 4, "g0conf2": 4, "g5am3": 4, "gen14a": 2,
    "lad0": 4, "lad8b": 4, "lad9": 4, "lad10": 4, "lad11b": 4, "lad12": 4,
    "gen16d": 4, "gen17b": 4,
}
for name, d in phase1.items():
    k = K_BY_BATCH.get(name)
    acct.append({"batch": f"evidence/{name}", "n_instances": d["n_runs"],
                 "K_documented": k,
                 "official_units": d["n_runs"]*k if k else None,
                 "mean": d["mean_score"], "mtime": d["mtime"]})
for tag, root in summ.items():
    for rel, v in root.items():
        if v.get("K") and v.get("jobs"):
            acct.append({"batch": f"{tag}/{rel}", "n_instances": v["jobs"],
                         "K_documented": v["K"],
                         "official_units": v["jobs"]*v["K"],
                         "mean": v["mean"], "label": v.get("label")})
jdump(acct, "50_official_call_accounting.json")

# judge (minimax) call accounting from verdict files (3 calls/pair jgen2/4)
jcalls = {}
for p in glob.glob(f"{JL}/v_*.jsonl") + glob.glob(f"{REPO}/analysis/**/verdicts*.jsonl", recursive=True):
    jcalls[os.path.relpath(p, "<frozen_corpus>")] = len(jlines(p))
jdump(jcalls, "51_judge_verdict_counts.json")

# ------------------------------------------------------------- 9. mr audits
mr = {}
for p in glob.glob(f"{EVO}/reports/mr_r*_audit.json") + [f"{EVO}/reports/multiround_final.json"]:
    if os.path.exists(p): mr[os.path.basename(p)] = jload(p)
jdump(mr, "12_multiround_screen_audits.json")

jdump(MISSING, "99_missing.json")
print("DONE. missing:", len(MISSING))
