#!/usr/bin/env python3
"""S5 Phase II curve — GRADE arm official budget ledger reconstruction (zero compute, reads only existing data).

Produces three blocks:
 A. Per-arm official task execution ledger: scan all dataflow slot JSONs (including retry attempts); launched slots count
    Σ n_ui × eval_reps; auto-classify screening (diagnostic subset, n_ui<=3 and mostly appearance-free) vs official (full task set).
    Mac-era batches compute Σ len(ui) from result.json in the volume-repo runs/ (K=1).
 B. Event ledger: map arms to events in campaign chronology (P1 official-only rolling generations / P2 judge-screened wide search /
    post-champion mechanism hunt / measurement batches), with official task counts and cumulative values per event — the x-axis raw material for the "cumulative official calls vs current champion score" curve.
 C. Champion timeline y-axis: selection-set scores of gen0→gen3b(§16)→p7(r7full) + the E1 83-case unbiased anchor (dual conventions).
    Per-round N candidates / K shortlist / evaluation saving rate are included from ws_p*_result.json + evo_state.json.
Output: paper_assets/webgen/phase2_curve.json (+ .csv summary)
"""
import json
import re
import statistics
from pathlib import Path

DF = Path("<frozen_corpus>/dataflow/results")
LAB = Path("<frozen_corpus>/lab")
EV = Path("<frozen_corpus>/evolve/evidence")
VRUNS = Path("<frozen_corpus>/runs")
OUT = Path("<frozen_corpus>/paper_assets")
OUT.mkdir(parents=True, exist_ok=True)

PAT = re.compile(r"run_(.+)__(\d{6})__s(\d+)(?:__a(\d+))?$")

ERA_RULES = [
    (r"^(r3s?_|r4s?_|r5s?_|final_gen)", "P1_official_only_rolling"),
    (r"^(nav|form|search|color|ctrl)$", "P2_r2_wide_search"),
    (r"^c3", "P2_r2_K3_confirm"),
    (r"^(navF|ctrlF)$", "P2_p2final_full_audit"),
    (r"^navT", "P2_train_evidence"),
    (r"^r1[02]ev$|^r[3-9]ev$", "P2_daemon_evidence"),
    (r"^p2r1", "P2_r1_wide_search"),
    (r"^(p3_|p3par)", "P2_ws_p3"),
    (r"^(p([4-9]|10)_|p([4-9]|10)par)", "P2_daemon_p4_p10"),
    (r"^(r[4-9]s[CPX]|r10s[CX])", "P2_daemon_slices"),
    (r"^r7f[CX]$", "P2_p7_full_transfer"),
    (r"^(p1[1-9]|p2[0-4])", "P3_postchampion_mechanism_hunt"),
    (r"^(r1[378]s[CX]|r18f[CX])", "P3_postchampion_mechanism_hunt"),
    (r"^(fa|dp|wf|fc|ns\d?|pb\d?|par|parB|p7probe|p21dbg)$", "P3_postchampion_mechanism_hunt"),
    (r"^(g0|g3b|p7)$", "M_e1_unbiased_audit"),
    (r"^[vm](0|3b|p7)$", "M_crossmodel_fit"),
    (r"^(podsmoke|smoke)", "M_smoke"),
]


def era_of(arm):
    for pat, era in ERA_RULES:
        if re.match(pat, arm):
            return era
    return "UNMAPPED"


def main():
    # A. dataflow per-arm accounting
    arms = {}
    for f in DF.glob("run_*.json"):
        m = PAT.match(f.stem)
        if not m:
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        a = arms.setdefault(m.group(1), {
            "slots": 0, "launched_slots": 0, "official_task_execs": 0,
            "n_ui_vals": [], "no_appearance": 0})
        a["slots"] += 1
        if d.get("launched", True) and d.get("n_ui"):
            a["launched_slots"] += 1
            a["official_task_execs"] += d["n_ui"] * max(d.get("eval_reps", 1), 1)
            a["n_ui_vals"].append(d["n_ui"])
            if d.get("appearance") is None:
                a["no_appearance"] += 1
    for name, a in arms.items():
        med = statistics.median(a["n_ui_vals"]) if a["n_ui_vals"] else 0
        a["median_n_ui"] = med
        a["kind"] = "screening_diag" if med and med <= 3 else "official_full"
        a["era"] = era_of(name)
        del a["n_ui_vals"]

    # Mac-era volume runs (K=1)
    mac = {}
    for top in sorted(VRUNS.glob("*")):
        if not top.is_dir():
            continue
        tasks = files = 0
        for rj in top.rglob("result.json"):
            try:
                r = json.loads(rj.read_text())
            except Exception:
                continue
            files += 1
            tasks += len(r.get("ui") or [])
        if files:
            mac[top.name] = {"runs": files, "official_task_execs": tasks}

    # B. era totals
    era_tot = {}
    for name, a in arms.items():
        e = era_tot.setdefault(a["era"], {"arms": [], "slots": 0,
                                          "official_full_tasks": 0, "screening_diag_tasks": 0})
        e["arms"].append(name)
        e["slots"] += a["slots"]
        key = "screening_diag_tasks" if a["kind"] == "screening_diag" else "official_full_tasks"
        e[key] += a["official_task_execs"]
    for e in era_tot.values():
        e["arms"] = sorted(e["arms"])

    # C. per-round search metadata + champion timeline
    rounds = {}
    for f in sorted(LAB.glob("ws_p*_result.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        rounds[d.get("round", f.stem)] = {
            "parent": d.get("parent"),
            "n_candidates": len(d.get("directions") or []),
            "n_screened": len(d.get("budget_rank") or []),
            "shortlist_K": len(d.get("top") or []),
            "eval_saving_tasks": d.get("eval_saving_tasks"),
        }
    evo = json.loads((LAB / "evo_state.json").read_text())

    champion_timeline = []
    fin = json.loads((EV / "final_m4k2_retry_compare.json").read_text()) if (EV / "final_m4k2_retry_compare.json").exists() else None
    champion_timeline.append({
        "event": "P1 final verdict gen3b transfer (§16, 12 heldout × m4 × K2)",
        "champion": "gen3b", "selection_set_scores": {"gen0": 0.5410, "gen3b": 0.5921},
        "delta": "+0.0511 t=1.485 W7/L5", "raw": "evolve/evidence/final_m4k2_retry_compare.json"})
    p2f = LAB / "p2final_verdict.json"
    if p2f.exists():
        champion_timeline.append({
            "event": "P2 nav full-set final verdict failed the gate (champion still gen3b)",
            "champion": "gen3b",
            "raw_content": json.loads(p2f.read_text())})
    r7f = LAB / "r7_full_verdict.json"
    if r7f.exists():
        champion_timeline.append({
            "event": "P2 daemon p7_emitatomicit full-set three-gate transfer (§17 cont. 6)",
            "champion": "p7_emitatomicit", "delta": "+0.0584 t=1.67 W8/L4",
            "raw_content": json.loads(r7f.read_text())})
    e1 = json.loads((LAB / "e1_verdict.json").read_text())
    champion_timeline.append({
        "event": "E1 unbiased re-evaluation anchor (83 untouched cases, §18) — second y convention of the curve",
        "arms": e1["arms"],
        "pairs": {k: {kk: v[kk] for kk in ("mean_delta", "W/L/T")} for k, v in e1["pairs"].items()}})

    out = {
        "note": "Phase II curve raw material: x=official UI task executions (launched slots Σ n_ui×eval_reps; "
                "screening=official executions of the diagnostic subset, listed separately), y=champion timeline (selection-set convention + E1 unbiased anchor, dual conventions). "
                "Mac-era batches (P1 r1/r2b, P2 r1) computed from volume-repo runs/, K=1. "
                "UNMAPPED arms are listed as-is without forced bucketing.",
        "per_arm": {k: arms[k] for k in sorted(arms)},
        "mac_era_volume_runs": mac,
        "era_totals": era_tot,
        "search_rounds_meta": rounds,
        "daemon_state_history": evo,
        "champion_timeline": champion_timeline,
    }
    (OUT / "phase2_curve.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

    lines = ["era,official_full_tasks,screening_diag_tasks,slots,n_arms"]
    for e in sorted(era_tot):
        t = era_tot[e]
        lines.append(f"{e},{t['official_full_tasks']},{t['screening_diag_tasks']},{t['slots']},{len(t['arms'])}")
    (OUT / "phase2_curve_era_summary.csv").write_text("\n".join(lines))
    print("\n".join(lines))
    unmapped = era_tot.get("UNMAPPED", {}).get("arms", [])
    print("UNMAPPED arms:", unmapped)
    print("WROTE", OUT / "phase2_curve.json")


if __name__ == "__main__":
    main()
