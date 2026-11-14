# The GRADE Loop as Pseudocode

Supports Sec. 2 (method) of the paper. One round of the search loop, plus the
deployment-time routing rule.

**Input:** champion actor harness $h_A^{t}$, judge harness $h_J^{t}$, failure
signatures $\mathcal{F}_t$, disjoint task sets $\mathcal{T}_{\mathrm{sel}}$
(selection) and $\mathcal{T}_{\mathrm{conf}}$ (confirmation; the latter split
into a gated layer $\mathcal{T}_{\mathrm{conf}}^{\mathrm{g}}$ and a
never-gated layer $\mathcal{T}_{\mathrm{conf}}^{\mathrm{ng}}$ that no decision
touches), shortlist size $k$.

**Deployment routing:** $\widehat y(x) = \arg\max_{y \in \mathrm{pool}(x)}
Q_J(x, y)$, where the pool holds $N$ rollouts of one accepted harness or one
rollout from each of $N$ confirmed siblings.

```
 1  C_t ← { Validate(Propose(h_A^t, f, Ω_A)) : f ∈ F_t }
 2  roll out C_t on T_sel; judge scores all artifacts
      S_t ← top-k by Q_J(h) plus judge-uncertain candidates
      [instance level: allocation]
 3  evaluate S_t officially m times; drop unstable Q_B            [G1]
 4  A_t ← {(x, y) : Q_J high, Q_B stably low}, enlarged in challenge
      mode every r rounds
      h_J^{t+1} ← Validate(Propose(h_J^t, f, Ω_J)) over judge-side
      signatures from A_t; refit Calibrate                        [G2]
 5  h* ← argmax_{h ∈ S_t} Q_B^sel(h)
      evaluate h* and h_A^t on T_conf^g only
      h_A^{t+1} ← h* if Q_B^g(h*) > Q_B^g(h_A^t), else h_A^t
      [G3; policy level: benchmark only; each acceptance is a
       two-candidate argmax bounded by c_2·σ_c]
 6  F_{t+1} ← signatures from the official traces of h_A^{t+1} on
      T_sel only (confirmation traces never enter the edit loop)
```

**Notation.** $\mathrm{Propose}(h, f, \Omega)$ maps a failure signature $f$
to a candidate edit of harness $h$ within the editable surface $\Omega$
($\Omega_A$: actor side; $\Omega_J$: judge side); $\mathrm{Validate}$ type- and
budget-checks the edit. $Q_J$ is the judge's score; $\bar Q_B$ denotes the
stable official aggregate (mean over $m$ evaluations, dropped when unstable).
G1–G3 are the
three gates: G1 discards officially unstable candidates, G2 admits only
benchmark-verified counterexamples as judge-supervision, G3 lets the benchmark
alone promote the champion on the gated confirmation layer.

The gates make the division of authority operational: the judge allocates
official budget (step 2) and repairs itself on benchmark-verified
counterexamples (step 4), while champion promotion (step 5) is withheld from
it — the control that hands step 5 to the judge instead is the paper's P2
(policy hand-off) arm.
