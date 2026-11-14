#!/usr/bin/env python3
"""Round-2: re-route the EXISTING sealed 5-arm pool with jgen4 + margin tie-break.

Per group (16 groups = 8 sealed tasks x 2 seeds, 5 arms):
  1. round-robin all C(5,2)=10 pairs with the given judge (rendering/caching and
     judge-driving imported from rejudge_pairs.py — no fork);
  2. Copeland points (win 1, tie 0.5 each);
  3. margin tie-break: if the top point total is shared by >1 arm, re-collect ONLY
     the mutual pairs among tied arms (already judged — verdicts reused, no new
     API calls) and rank tied arms by viewport-vote share (an arm's share = its
     counted viewport votes / total counted votes in those mutual pairs); if still
     tied, lowest arm index wins.

Outputs in --out-dir:
  verdicts_jgen4_sealed.jsonl      all judged pairs, with per-viewport vote detail
  selection_report_jgen4_sealed.json  per group: points, tied set, winner, tie-break path
  flips.json                       winners vs the archived jgen2 selection

Dry run: pass --pairs <id,id> to judge only those pairs (selection/flips skipped
unless every group is fully judged).
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rejudge_pairs  # noqa: E402 — reuse render_pages + run_judge (no fork)

DEFAULT_DATASETS = "./datasets/webpage"


def norm_seed(seed):
    s = str(seed)
    return s if s.startswith("seed_") else f"seed_{s}"


def build_pairs(manifest):
    """All C(5,2) pairs per group: [{exam, group_index, left, right}]."""
    pairs = []
    for gi, group in enumerate(manifest["groups"]):
        cands = group["candidates"]
        for left, right in itertools.combinations(range(len(cands)), 2):
            pid = f"r2sealed_g{gi:03d}_{left}{right}"
            pairs.append({
                "exam": {"id": pid, "task": group["task"],
                         "A": cands[left]["id"], "B": cands[right]["id"]},
                "group_index": gi, "left": left, "right": right,
            })
    return pairs


def copeland_points(verdicts, pairs, n_arms):
    points = [0.0] * n_arms
    for p in pairs:
        w = verdicts[p["exam"]["id"]]["winner"]
        if w == "A":
            points[p["left"]] += 1.0
        elif w == "B":
            points[p["right"]] += 1.0
        else:
            points[p["left"]] += 0.5
            points[p["right"]] += 0.5
    return points


def margin_tiebreak(verdicts, pairs, tied):
    """Rank tied arms by viewport-vote share over their mutual pairs.

    Reuses existing verdicts only — no new API calls. Share = arm's counted
    votes / total counted votes (A+B; abstentions excluded) in mutual pairs.
    Returns (ordered_best_first, detail).
    """
    mutual = [p for p in pairs if p["left"] in tied and p["right"] in tied]
    got = {a: 0 for a in tied}
    total = 0
    votes_available = False
    for p in mutual:
        rec = verdicts[p["exam"]["id"]]
        v = rec.get("votes")
        if not isinstance(v, dict):
            continue
        va, vb = int(v.get("A", 0)), int(v.get("B", 0))
        got[p["left"]] += va
        got[p["right"]] += vb
        total += va + vb
        votes_available = True
    share = {a: (got[a] / total if total else 0.0) for a in tied}
    ordered = sorted(tied, key=lambda a: (-share[a], a))
    detail = {
        "mutual_pair_ids": [p["exam"]["id"] for p in mutual],
        "votes_available": votes_available,
        "votes": {str(a): got[a] for a in tied},
        "total_counted_votes": total,
        "share": {str(a): round(share[a], 6) for a in tied},
    }
    return ordered, detail


def select_groups(manifest, verdicts):
    """Copeland + margin tie-break per fully-judged group.

    Completeness is checked against ALL C(5,2) pairs of each group; groups with
    any unjudged pair get status "incomplete" (partial/dry runs).
    """
    arms = manifest["arms"]
    by_group = {}
    for p in build_pairs(manifest):
        by_group.setdefault(p["group_index"], []).append(p)

    groups_out = []
    for gi, group in enumerate(manifest["groups"]):
        gpairs = by_group.get(gi, [])
        if any(p["exam"]["id"] not in verdicts for p in gpairs):
            groups_out.append({"task": group["task"], "seed": group["seed"],
                               "status": "incomplete"})
            continue
        points = copeland_points(verdicts, gpairs, len(arms))
        best = max(points)
        tied = [i for i, v in enumerate(points) if v == best]
        if len(tied) == 1:
            winner, path, margin = tied[0], "copeland", None
        else:
            ordered, margin = margin_tiebreak(verdicts, gpairs, tied)
            if margin["share"][str(ordered[0])] > margin["share"][str(ordered[1])]:
                winner, path = ordered[0], "margin_vote_share"
            else:
                winner, path = tied[0], "first_index_after_margin_tie"
        groups_out.append({
            "task": group["task"],
            "seed": group["seed"],
            "status": "selected",
            "arms": list(arms),
            "points": points,
            "tied_arm_indices": tied,
            "arm_index": winner,
            "selected_arm": arms[winner],
            "tiebreak_path": path,
            "margin": margin,
        })
    return groups_out


def archived_winners(path):
    """Extract {group_key: arm_name} from the archived jgen2 selection report.

    groups[].arm_index indexes into config.arms; seeds are normalized to seed_N.
    """
    rep = json.loads(Path(path).read_text())
    arms = rep["config"]["arms"]
    winners = {}
    for g in rep["groups"]:
        key = f"{g['task']}_{norm_seed(g['seed'])}"
        winners[key] = arms[g["arm_index"]]
    return winners


def compute_flips(manifest, groups_out, archived_path):
    archived = archived_winners(archived_path)
    new_winners, flips = {}, []
    for g in groups_out:
        if g.get("status") != "selected":
            continue
        key = f"{g['task']}_{norm_seed(g['seed'])}"
        new_winners[key] = g["selected_arm"]
        old = archived.get(key)
        if old is not None and old != g["selected_arm"]:
            flips.append({"group": key, "archived_arm": old,
                          "new_arm": g["selected_arm"],
                          "tiebreak_path": g["tiebreak_path"]})
    return {
        "archived_report": str(archived_path),
        "n_groups": len(new_winners),
        "n_flips": len(flips),
        "flips": flips,
        "archived_winners": archived,
        "new_winners": new_winners,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--renders", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--datasets", default=DEFAULT_DATASETS)
    ap.add_argument("--archived", help="archived jgen2 selection_report.json (for flips.json)")
    ap.add_argument("--pairs", help="comma-separated pair ids to judge (dry run)")
    ap.add_argument("--verdicts-name", default="verdicts_jgen4_sealed.jsonl")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--port", type=int, default=26420)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    renders_dir = Path(args.renders)

    all_pairs = build_pairs(manifest)
    if args.pairs:
        keep = set(args.pairs.split(","))
        pairs = [p for p in all_pairs if p["exam"]["id"] in keep]
        unknown = keep - {p["exam"]["id"] for p in pairs}
        if unknown:
            raise SystemExit(f"unknown pair ids: {sorted(unknown)}")
    else:
        pairs = all_pairs

    # render every referenced candidate (cache-complete ones are skipped)
    pages = {}
    for p in pairs:
        cands = manifest["groups"][p["group_index"]]["candidates"]
        for idx in (p["left"], p["right"]):
            pages[cands[idx]["id"]] = cands[idx]["dir"]
    rejudge_pairs.render_pages(pages, renders_dir, args.port)
    incomplete = [pid for pid in pages if not all(
        (renders_dir / pid / f"{vp}.png").exists() for vp in rejudge_pairs.VPS)]
    if incomplete:
        raise SystemExit(f"incomplete renders: {incomplete}")

    exam = [p["exam"] for p in pairs]
    verdicts_path = out_dir / args.verdicts_name
    verdicts = rejudge_pairs.run_judge(exam, args.judge, renders_dir, args.datasets,
                                       verdicts_path, out_dir / "work", args.concurrency)
    missing = [p["exam"]["id"] for p in pairs if p["exam"]["id"] not in verdicts]
    if missing:
        raise SystemExit(f"missing verdicts: {missing}")
    print(json.dumps({"judged_pairs": len(pairs), "verdicts": str(verdicts_path)}))

    groups_out = select_groups(manifest, verdicts)
    report = {
        "manifest": str(Path(args.manifest).resolve()),
        "judge": str(Path(args.judge).resolve()),
        "selection_policy": ("round-robin Copeland points; ties: viewport-vote share "
                             "over tied arms' mutual pairs; then lowest arm index"),
        "groups": groups_out,
    }
    report_path = out_dir / "selection_report_jgen4_sealed.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    n_sel = sum(1 for g in groups_out if g.get("status") == "selected")
    print(json.dumps({"groups_selected": n_sel, "report": str(report_path)}))

    if args.archived and n_sel == len(manifest["groups"]):
        flips = compute_flips(manifest, groups_out, args.archived)
        flips_path = out_dir / "flips.json"
        flips_path.write_text(json.dumps(flips, indent=2) + "\n")
        print(json.dumps({"n_flips": flips["n_flips"], "flips": str(flips_path)}))
    elif args.archived:
        print("partial run: flips.json skipped (not all groups selected)")


if __name__ == "__main__":
    main()
