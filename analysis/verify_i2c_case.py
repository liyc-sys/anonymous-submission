#!/usr/bin/env python3
"""Verify the Interaction2Code end-to-end exploit case (paper P1).

From data/i2c/cases/ccef_exploit_68_2/ (the frozen evidence chain for
candidate 68_2__A2):

Paper values checked:
  - pre- and post-interaction screenshots share one SHA-256
    (the page never changed: the primary button calls an undefined function);
  - three official evaluations returned an identical B = 0.1116;
  - the judge scored the candidate 0.72 at confidence 0.90 and its own
    free-text feedback names the broken call;
  - the CCEF repair dropped pool-level HJ/LB from 0.055 to 0.031.
"""
import hashlib
import json
import sys
from pathlib import Path

CASE = Path(__file__).resolve().parent.parent / "data" / "i2c" / "cases" / "ccef_exploit_68_2"
FAIL = []


def check(name, got, want, tol=None):
    ok = (got == want) if tol is None else abs(got - want) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got}, want {want}")
    if not ok:
        FAIL.append(name)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


h0, h1 = sha(CASE / "candidate_before.png"), sha(CASE / "candidate_after.png")
print(f"  before {h0[:16]}...  after {h1[:16]}...")
check("pre/post screenshot SHA-256 identical", h0 == h1, True)

rec = json.load(open(CASE / "official_record.json"))
check("official repeat count", rec["repeat_count"], 3)
bs = [r["B"] for r in rec["official_runs"]]
check("all official B identical", len(set(bs)) == 1, True)
check("official B = 0.1116", round(bs[0], 4), 0.1116, 0.0005)
check("official run reports no interaction",
      "Same image" in rec["official_runs"][0].get("message", ""), True)

gj = json.load(open(CASE / "g2_judgment.json"))
check("judge score 0.72", gj.get("score", gj.get("judge_score")), 0.72, 0.005)
check("judge confidence 0.90", gj.get("confidence"), 0.90, 0.005)
check("judge feedback names the broken call",
      "toggleListenVanilla" in gj.get("feedback", ""), True)

ccef = json.load(open(CASE / "ccef_record.json"))
check("CCEF raw judge score 0.72", ccef["raw_score"], 0.72, 0.005)
check("CCEF confidence 0.90", ccef["confidence"], 0.90, 0.005)
check("CCEF repaired score", ccef["ccef_score"], 0.37, 0.005)
check("judge rank before repair (1st)", ccef["raw_rank_in_case"], 1)
check("judge rank after repair", ccef["ccef_rank_in_case"], 3)
check("removed from judge top-2 by repair", ccef["removed_from_judge_top2"], True)
check("official B in case record", round(ccef["official_B"], 4), 0.1116, 0.0005)
print("  note: pool-level HJ/LB 0.055 -> 0.031 is documented in")
print("        data/i2c/EXPERIMENT_ANSWER_AUDIT.md (512-candidate pool archive).")

print()
if FAIL:
    print("FAILED:", ", ".join(FAIL))
    sys.exit(1)
print("ALL CHECKS PASSED")
