#!/usr/bin/env python3
"""V2W Table-4 judge ablation add-on computation: Spearman rank correlation + HJ/LB.

Zero-compute, purely offline calculation. Inputs:
  - Shipped data directory data/v2w/ (parent directory of this script)
      10_phase1_evidence_batches.json   lad* ladder batch per-run official K4 scores
      12_multiround_screen_audits.json  multiround screening R2-R4 audits (judge=jgen2)
      sources/reports/e3_e4_audit.json  gen14 batch audit (judge=jgen2, see comment below)
  - raw/ in this directory (read-only copied back from the frozen corpus <frozen_corpus>/judge_lab/ on 2026-07-29)
      exam_bon_jgen{0,1,2}.json         Night-3 BoN exam: 144 pairs = 24 groups × C(4,2) round-robin
      v_bon_jgen{0,1,2}.jsonl           verdicts of the three judge generations on the same 144 pairs
      exam_heldout.json / answers_heldout.json / v_jgen{0,1,2}_heldout.jsonl  120-pair heldout exam

Judge codename mapping (verified against history.md / multiround_status.json / file timestamps):
  V2W-J0 = jgen0 raw VLM; V2W-J1 = jgen1 viewport majority vote; V2W-J2 = jgen2 +challenge.
  lad* = ladder confirmation batches (official K4 scores of actor genomes gen0/8b/9/10/11b/12), not judges.
  multiround_status.json explicitly records "judge": "jgen2"; the "t" in the e3_e4 audit is the verdict timestamp
  2026-07-16 18:32, between jgen2 exam creation (07-15 22:59) and jgen3 exam creation (07-16 19:33) → t=jgen2.
"""
import json, os, itertools, statistics

PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")  # shipped data/v2w package
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")

def jload(p):
    with open(p) as f: return json.load(f)

def jlines(p):
    return [json.loads(l) for l in open(p) if l.strip()]

def avg_ranks(xs):
    """Average ranks (ties take the mean); returns a rank list in the same order as xs, ranks starting at 1."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks

def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx ** 0.5 * vy ** 0.5)

def spearman(xs, ys):
    """Spearman ρ = Pearson over average ranks (the standard tie-corrected approach)."""
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return pearson(avg_ranks(xs), avg_ranks(ys))

results = {}

# ============================================================ A. BoN pool Spearman
# Page official K4 scores: lad* ladder batches per run (per_run of 10_phase1_evidence_batches.json)
ev = jload(os.path.join(PKG, "10_phase1_evidence_batches.json"))
page_score = {}
for arm in ("lad0", "lad8b", "lad9", "lad10", "lad11b", "lad12"):
    for r in ev[arm]["per_run"]:
        if isinstance(r.get("score"), (int, float)):
            page_score[f"{arm}__{r['task']}__{r['seed']}"] = r["score"]

groups = {}
for pid, s in page_score.items():
    arm, task, seed = pid.split("__")
    groups.setdefault(f"{arm}|{task}", {})[pid] = s
complete = {g: d for g, d in groups.items() if len(d) == 4}
incomplete = {g: sorted(d) for g, d in groups.items() if len(d) != 4}
print(f"[A] official K4 pages {len(page_score)}; groups {len(groups)}, complete groups (4 seeds) {len(complete)}")
print(f"    incomplete groups: {sorted(incomplete)}")

bon_spearman = {}
for judge in ("jgen0", "jgen1", "jgen2"):
    exam = {e["id"]: e for e in jload(os.path.join(RAW, f"exam_bon_{judge}.json"))}
    # Pair parsing identical to the frozen extraction script v2w_extract.py
    vd = {}
    for v in jlines(os.path.join(RAW, f"v_bon_{judge}.jsonl")):
        e = exam.get(v["id"])
        if e:
            vd[(e["A"], e["B"])] = v.get("winner")
    def pairwin(a, b):
        w = vd.get((a, b))
        if w == "A": return a
        if w == "B": return b
        w = vd.get((b, a))
        if w == "A": return b
        if w == "B": return a
        return None
    # Within-group Copeland points (win=1, undecided=0.5)
    n_undecided = 0
    per_group = []
    for g in sorted(complete):
        pages = sorted(complete[g])
        pts = {p: 0.0 for p in pages}
        for a, b in itertools.combinations(pages, 2):
            w = pairwin(a, b)
            if w is None:
                n_undecided += 1
                pts[a] += 0.5; pts[b] += 0.5
            else:
                pts[w] += 1.0
        sc = [complete[g][p] for p in pages]
        pt = [pts[p] for p in pages]
        rho = spearman(pt, sc)
        per_group.append({"group": g, "pages": pages,
                          "judge_points": pt, "official_k4": sc, "spearman": rho})
    rhos = [r["spearman"] for r in per_group if r["spearman"] is not None]
    bon_spearman[judge] = {
        "n_groups": len(rhos),
        "mean_spearman": round(statistics.mean(rhos), 4),
        "sd": round(statistics.pstdev(rhos), 4),
        "median": round(statistics.median(rhos), 4),
        "min": round(min(rhos), 4), "max": round(max(rhos), 4),
        "n_undecided_pairs_total": n_undecided,
        "per_group": per_group,
    }
    print(f"    {judge}: mean Spearman={bon_spearman[judge]['mean_spearman']} "
          f"(sd {bon_spearman[judge]['sd']}, median {bon_spearman[judge]['median']}, "
          f"n={len(rhos)} groups × 4 pages, undecided pairs {n_undecided})")
results["A_bon_pool_spearman"] = {
    "definition": ("Night-3 BoN pool: 16 complete groups (lad ladder 6 arms × 4 tasks, 4 seed pages within each group)."
                   "Each judge generation runs the within-group 6-pair round-robin to Copeland points (win=1, undecided=0.5) as the judge ranking; "
                   "official axis=same-page K4 official mean. Spearman=Pearson over the 4 pages' average ranks, computed per group then averaged over 16 groups."),
    "judges": bon_spearman,
}

# ================================================== B. Screening campaign candidate-level Spearman (jgen2)
mr = jload(os.path.join(PKG, "12_multiround_screen_audits.json"))
cand = []
for rk in ("mr_r2_audit.json", "mr_r3_audit.json", "mr_r4_audit.json"):
    for c in mr[rk]["candidates"]:
        cand.append({"candidate": c["candidate"], "round": rk[:5],
                     "judge_winrate": c["judge_winrate"], "mean_delta": c["mean_delta"],
                     "official_mean": c["official_mean"], "ci95": c["ci95"]})
jw = [c["judge_winrate"] for c in cand]
md = [c["mean_delta"] for c in cand]
sp10 = spearman(jw, md)
print(f"[B] multiround R2-R4 finalists (jgen2, n={len(cand)}): Spearman(judge_winrate, official Δ)={round(sp10,4)}")

e3 = jload(os.path.join(PKG, "sources", "reports", "e3_e4_audit.json"))
cand15 = cand + [{"candidate": c["candidate"], "round": "e3_e4",
                  "judge_winrate": c["judge_winrate"], "mean_delta": c["mean_delta"],
                  "official_mean": c["official_mean"], "ci95": c["ci95"]}
                 for c in e3["candidates"]]
sp15 = spearman([c["judge_winrate"] for c in cand15], [c["mean_delta"] for c in cand15])
print(f"    +gen14 batch (e3_e4, also jgen2, n={len(cand15)}): Spearman={round(sp15,4)}")
results["B_screening_candidate_spearman_jgen2"] = {
    "definition": ("Judge axis=candidate's judge win rate against champion (gen10) (12-16 page-level duels per candidate); "
                   "official axis=candidate official mean − gen10 same-batch official mean (mean_delta). Cross-round pooled Spearman."),
    "multiround_R2R4_n10": {"n": len(cand), "spearman": round(sp10, 4), "candidates": cand},
    "plus_gen14_batch_n15": {"n": len(cand15), "spearman": round(sp15, 4), "candidates": cand15},
}

# ================================================== C. HJ/LB
# C1. Search-context shortlist version (jgen2, multiround finalists)
hj = {"R2": [], "R3": [], "R4": []}
for c in cand:
    hj[c["round"].replace("mr_r", "R").replace("_", "")].append(c)
per_round = {}
for r, cs in hj.items():
    bad = [c["candidate"] for c in cs if c["mean_delta"] <= 0]
    per_round[r] = {"n_finalist": len(cs), "hj_lb_count": len(bad), "hj_lb_names": bad}
n_bad = sum(v["hj_lb_count"] for v in per_round.values())
n_fin = sum(v["n_finalist"] for v in per_round.values())
n_unconfirmed = sum(1 for c in cand if c["ci95"][0] <= 0)
print(f"[C1] multiround shortlist HJ/LB (jgen2): Δ≤0 -> {n_bad}/{n_fin} = {n_bad/n_fin:.3f}; "
      f"CI95-excludes-0 confirmed improvements 0/{n_fin} (CI lower bound ≤0: {n_unconfirmed}/{n_fin})")
results["C1_hjlb_shortlist_jgen2"] = {
    "definition": ("Multiround screening R2-R4: each round jgen2 screens top-K by judge win rate from 5-6 candidates (K=3, expanded to 4 at the R2 boundary tie) "
                   "sent to official evaluation, 10 finalists in total. HJ/LB count=finalists with official mean_delta≤0 (tied with or worse than champion gen10, i.e. official re-check does not support the improvement). "
                   "Also reports the strict convention: count supported only when CI95 lower bound>0."),
    "per_round": per_round,
    "hj_lb_delta_le0": {"count": n_bad, "n": n_fin, "ratio": round(n_bad / n_fin, 4)},
    "hj_lb_unconfirmed_ci": {"count": n_unconfirmed, "n": n_fin, "ratio": round(n_unconfirmed / n_fin, 4)},
    "note": "None of the 10 finalists has a CI95 excluding 0; gen17b final verdict K4 Δ=+0.0175 CI[-0.0149,0.0500] confirmed=false.",
}

# C2. Exam pair-level fooled rate (three judge generations on the same 120-pair heldout exam)
answers = jload(os.path.join(RAW, "answers_heldout.json"))
exam_h = {e["id"]: e for e in jload(os.path.join(RAW, "exam_heldout.json"))}
exam_fooled = {}
for judge in ("jgen0", "jgen1", "jgen2"):
    verd = {v["id"]: v.get("winner") for v in jlines(os.path.join(RAW, f"v_{judge}_heldout.jsonl"))}
    n = fooled = abst = 0
    by_margin = {"0.05": [0, 0], "0.10": [0, 0]}  # thr -> [fooled, total]
    for pid, a in answers.items():
        if pid not in verd:
            continue
        n += 1
        w = verd[pid]
        gap = a.get("gap")
        if w not in ("A", "B"):
            abst += 1
            is_fooled = False
        else:
            is_fooled = (w != a["winner"])
        fooled += is_fooled
        for thr in ("0.05", "0.10"):
            if isinstance(gap, (int, float)) and gap > float(thr):
                by_margin[thr][1] += 1
                by_margin[thr][0] += is_fooled
    exam_fooled[judge] = {
        "n": n, "abstain": abst, "fooled": fooled,
        "fooled_rate": round(fooled / n, 4),
        "fooled_rate_decided": round(fooled / (n - abst), 4) if n > abst else None,
        "by_margin": {f">{k}": {"fooled": f, "n": t, "rate": round(f / t, 4)}
                      for k, (f, t) in by_margin.items()},
    }
    print(f"[C2] {judge} heldout 120 pairs: fooled {fooled}/{n} = {exam_fooled[judge]['fooled_rate']}"
          f" (>0.05: {by_margin['0.05'][0]}/{by_margin['0.05'][1]} = {exam_fooled[judge]['by_margin']['>0.05']['rate']}; "
          f">0.10: {by_margin['0.10'][0]}/{by_margin['0.10'][1]} = {exam_fooled[judge]['by_margin']['>0.10']['rate']})")
results["C2_hjlb_exam_fooled_heldout120"] = {
    "definition": ("The same 120-pair heldout exam (stratified by official score gap, |Δ|≥0.03): the judge picks A/B as winner; "
                   "fooled means the official answer is the opposite (a pair-level event of judge-high score vs official-low score). "
                   "Fooled rate=fooled pairs/120; also reports fooled rates on subsets with official gap >0.05 / >0.10 (official clearly favors one side)."),
    "judges": exam_fooled,
}

with open(OUT, "w") as f:
    json.dump(results, f, indent=1, ensure_ascii=False)
print("WROTE", OUT)
