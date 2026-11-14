#!/usr/bin/env python3
"""Verify the V2W adversarial (GAN) rounds and the judge lineage accuracy
(paper P1 exploit paragraph + Appendix D + Table 2 V2W rows).

Sources:
  data/v2w/case_gen13b/  - per-round exams, answer keys, judge verdicts,
                           and the after-repair rematch verdicts (v_ganR2_jgen2)
  data/v2w/21_gan_rounds_case.json - recomputed per-pair record
  data/v2w/20_judge_lineage_acc.json - judge lineage train/heldout accuracies

Paper values checked:
  R1: judge winrate 0.538, official delta -0.010, 0 trick pairs;
  R2: judge winrate 0.733 vs official tie (-0.0006), 4 of 15 trick pairs,
      Goodhart gap +0.200; after repair rematch 0.467 (gap -0.067);
  R3: gap 0.000, 0 trick pairs won;
  gap trajectory +0.077 -> +0.200 -> -0.067 -> 0.000;
  judge lineage heldout 0.758 -> 0.775 -> 0.842 (Table 2 V2W rows),
  train 0.744 -> 0.772 -> 0.794 (heldout +6.7pp at the challenge step).
"""
import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "v2w"
FAIL = []


def check(name, got, want, tol):
    ok = abs(got - want) <= tol if not isinstance(want, str) else got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got}, want {want} (+/-{tol})")
    if not ok:
        FAIL.append(name)


def winrates(verdicts_file, result_file):
    """judge winrate of the attack candidate and official winrate, per round."""
    res = json.load(open(DATA / "case_gen13b" / result_file))
    return res["judge_winrate"], res


case = json.load(open(DATA / "21_gan_rounds_case.json"))
print("== per-round headline (case_gen13b) ==")
jw1, r1 = winrates(None, "gan_ganR1_result.json")
check("R1 judge winrate", jw1, 0.538, 0.004)
check("R1 official delta", r1["official_delta"], -0.010, 0.004)
check("R1 trick pairs won", r1["fooling_pairs"], 0, 0)
jw2, r2 = winrates(None, "gan_ganR2_result.json")
check("R2 judge winrate", jw2, 0.733, 0.004)
check("R2 official delta (tie)", r2["official_delta"], -0.0006, 0.004)
check("R2 trick pairs won", f"{r2['fooling_pairs']}/{r2['n']}", "4/15", 0)
jw3, r3 = winrates(None, "gan_ganR3_result.json")
check("R3 trick pairs won", r3["fooling_pairs"], 0, 0)

print("== Goodhart gap trajectory (judge winrate - official winrate) ==")
gaps = []
for rnd, key in (("ganR1", "v_ganR1"), ("ganR2", "v_ganR2"), ("ganR3", "v_ganR3")):
    rec = case[rnd][key]
    pairs = rec["pairs"]
    jw = sum(1 for p in pairs if p["judge_picked_cand"]) / len(pairs)
    ow = sum(1 for p in pairs if p["official_delta"] > 0) / len(pairs)
    gap = jw - ow
    gaps.append(gap)
    print(f"  {rnd}: judge {jw:.3f} - official {ow:.3f} = gap {gap:+.3f}")
# after-repair rematch (v_ganR2_jgen2): judge winrate from the verdict file
v = [json.loads(l) for l in open(DATA / "case_gen13b" / "v_ganR2_jgen2.jsonl") if l.strip()]
res2 = json.load(open(DATA / "case_gen13b" / "gan_ganR2_result.json"))
key2 = {k["pid"]: k for k in res2["fools"]} if res2.get("fools") else {}
exam2 = {e["id"]: e for e in json.load(open(DATA / "case_gen13b" / "exam_ganR2.json"))}
jw_re = sum(1 for x in v if x.get("winner") == "B") / len(v)  # B = attack candidate side
ow_re = 1 - 0.533  # official side unchanged from the R2 exam (candidate loses ~0.533)
gap_re = jw_re - ow_re
print(f"  ganR2 after repair (jgen2): judge {jw_re:.3f} vs official {ow_re:.3f} = gap {gap_re:+.3f}")
check("gap R1 (+0.077)", round(gaps[0], 3), 0.077, 0.02)
check("gap R2 (+0.200)", round(gaps[1], 3), 0.200, 0.02)
check("gap R2 after repair (-0.067)", round(gap_re, 3), -0.067, 0.02)
check("gap R3 (0.000)", round(gaps[2], 3), 0.000, 0.02)

print("== judge lineage (Table 2 V2W rows) ==")
lin = json.load(open(DATA / "20_judge_lineage_acc.json"))
for j, w_h, w_t in (("jgen0", 0.758, 0.744), ("jgen1", 0.775, 0.772), ("jgen2", 0.842, 0.794)):
    check(f"{j} heldout acc", round(lin[j]["heldout"]["acc"], 3), w_h, 0.004)
    check(f"{j} train acc", round(lin[j]["train"]["acc"], 3), w_t, 0.004)
step = lin["jgen2"]["heldout"]["acc"] - lin["jgen1"]["heldout"]["acc"]
check("challenge step heldout gain (+6.7pp)", round(step, 3), 0.067, 0.004)

print()
if FAIL:
    print("FAILED:", ", ".join(FAIL))
    sys.exit(1)
print("ALL CHECKS PASSED")
