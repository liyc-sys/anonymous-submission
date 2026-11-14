# Frozen Interaction2Code Evidence Pack

This directory was assembled on 2026-07-28 after the experiments were frozen; it
contains only evidence snapshots of completed experiments. No experiments were
re-run, no harness modified, no models called, and no metrics recomputed.

## Contents

- `cases/CASE_INDEX.md`: index of case purposes and original sources.
- `cases/ccef_exploit_68_2/`: the final paper deliverable case, containing the
  complete evidence chain — the G2 high-score misjudgment, three official
  confirmations, the Failure Signature, the CCEF correction, and corroborating
  evidence of similar errors corrected the same way.

The copied cases are snapshots for reading convenience. The judge-lineage numbers
(pairwise accuracy, R_2, HJ/LB by generation) tabulated in the supplement's
`docs/judge-grid.md` trace to these frozen case records; the shipped
`analysis/verify_i2c_case.py` re-checks the case's key values (scores, ranks,
SHA-256, signature fields) offline.
