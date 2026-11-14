# Proofs and Model Details

Supports Secs. 3.2–3.3 of the paper. Statements are restated so this document
is self-contained; notation follows the paper ($Q_B$: benchmark utility,
$\widetilde B_r$: the $r$-th official evaluation, $h$: a harness,
$\sigma_{\mathrm{gen}}$: generation noise, $\sigma_J$: judge noise,
$\sigma_B$: evaluator noise, $b_J(h)$: harness-indexed judge miscalibration).

## Proposition 1 (reported vs. true gain)

**Statement.** Let pool qualities be $q_i \sim \mathcal N(\mu,\sigma_s^2)$ and
scores $y_i = q_i + e_i$, with independent $e_i \sim \mathcal N(0,\sigma_e^2)$
and i.i.d. candidates. For $i^\star = \arg\max_i y_i$ among $N$ candidates,

$$
\mathbb E[y_{i^\star}] - \mu = c_N\sqrt{\sigma_s^2 + \sigma_e^2}
\quad\text{(reported gain)}, \qquad
\mathbb E[q_{i^\star}] - \mu = \rho^2\,\mathbb E[y_{i^\star}] - \rho^2\mu
\quad\text{(true gain)},
$$

where $\rho = \sigma_s / \sqrt{\sigma_s^2+\sigma_e^2}$ and
$c_N = \mathbb E[\max_{i\le N} Z_i]$ for standard normals.

**Proof.** The scores $y_i \sim \mathcal N(\mu, \sigma_s^2+\sigma_e^2)$ are
i.i.d., so $\mathbb E[y_{i^\star}] - \mu = c_N\sqrt{\sigma_s^2+\sigma_e^2}$ by
definition of $c_N$. Since $(q_i, y_i)$ is bivariate normal with correlation
$\rho = \sigma_s/\sqrt{\sigma_s^2+\sigma_e^2}$, the regression
$\mathbb E[q \mid y] = \mu + \rho^2 (y-\mu)$ holds, and because $i^\star$ is
$y$-measurable, $\mathbb E[q_{i^\star}] - \mu = \rho^2 (\mathbb E[y_{i^\star}]
- \mu)$. The optimism is their difference,
$(1-\rho^2)\,c_N\sqrt{\sigma_s^2+\sigma_e^2}
= c_N\,\sigma_e^2 / \sqrt{\sigma_s^2+\sigma_e^2}$, which tends to
$c_N \sigma_e$ as $\sigma_s/\sigma_e \to 0$. $\square$

**Constants.** $c_2 = 1/\sqrt\pi \approx 0.564$, $c_3 = 3/(2\sqrt\pi) \approx
0.846$, $c_4 \approx 1.029$, $c_{48} \approx 2.233$ (numerically,
$c_N = N \int z\,\phi(z)\,\Phi(z)^{N-1}\,dz$). The familiar bound
$c_N \le \sqrt{2\ln N}$ overstates $c_N$ by 25% at $N=48$ and by 62% at $N=4$,
which is why the paper quotes $c_N$ itself: with $\sigma_{\mathrm{est}}=0.035$
and $N \approx 48$ the predicted optimism is $0.078$, not the $0.097$ the
bound would give. The exact values are recomputed by
`analysis/verify_revision_checks.py`.

**Per-gate bound (used by G3).** The two-candidate case supplies it: a single
accept/reject comparison on the gated layer is an argmax over two noisy scores
and carries at most $c_2\,\sigma_c = \sigma_c/\sqrt\pi \approx 0.56\,\sigma_c$
of optimism, with $\sigma_c$ the gated layer's standard error.

## Pairwise accuracy relations (Corollary A support)

For a pair of true gap $\Delta$, the score gap is
$\mathcal N(\Delta, 2\sigma_J^2)$, so the ranking is correct with probability
$\Phi\bigl(\Delta / (\sigma_J \sqrt 2)\bigr)$; setting this to $0.75$ gives
$\Delta_{75}$, the indifference-zone threshold quoted in the paper. For two
pool members drawn at random, $(q_1 - q_2,\; y_1 - y_2)$ is centered
bivariate normal with correlation $\rho$, whence
$P(\text{same sign}) = \tfrac12 + \arcsin(\rho)/\pi$ (the capture-ratio
identity; it is Proposition 1's two statements divided). Substituting
$\sigma_s = \sigma_{\mathrm{gen}}$ and $\sigma_e = \sigma_J$ specializes
Proposition 1 to selection within one fixed $(h, x)$, where the task effect
$\tau(x)$ and the miscalibration $b_J(h)$ are common to all candidates and
cancel — the regime Corollary A assumes.

## Proposition 2(i): the variance floor

**Statement.** For any affine scorer $g = a + b\,Q_B + \varepsilon$
(interpreted on the benchmark-utility scale), the estimate
$\widehat Q = \frac{1}{ns} \sum_{ij} g(y_{ij})$ over $n$ tasks $\times$ $s$
rollouts obeys $\operatorname{Var}[\widehat Q] \ge \sigma_{\mathrm{gen}}^2 / (ns)$,
with equality only for a perfect scorer.

**Proof.** Condition on the harness $h$ and the task set. By the law of total
variance and the independence of rollouts,
$\operatorname{Var}[\widehat Q] \ge
\operatorname{Var}\bigl[\tfrac{1}{ns} \sum_{ij} Q_B(x_i, y_{ij})\bigr]
= \sigma_{\mathrm{gen}}^2/(ns)$, the inequality strict unless $g$ recovers
$Q_B$ exactly. $\square$

Replacing $g$ by the official evaluator adds its own noise, giving
$(\sigma_{\mathrm{gen}}^2 + \sigma_B^2)/(ns)$; the ratio of standard errors
bounds the benefit of any judge at
$\sqrt{1 + \sigma_B^2/\sigma_{\mathrm{gen}}^2}$ — about $1.24$ at the paper's
measured ratio, a ceiling of 35% sample savings per fixed rollout set.

**Scope remarks.** The floor is a statement about rollouts, not about judges:
it falls as $1/\sqrt{ns}$ whoever scores the rollouts, and a judge changes
only the factor above it. Obstruction (ii) of the paper — differential
miscalibration selected by argmax — is empty for a judge with
$b_J(h) \equiv 0$; the proposition therefore describes two regimes
(variance-dominated at screening budgets, bias-dominated once rollouts are
scaled) rather than an impossibility. Shrinkage estimators can beat the floor
in mean-squared error, but buy it by compressing the between-harness
contrasts that promotion must resolve; monotone recalibration is affine to
first order, so the class of scorers covered includes raw, recalibrated and
perfect scorers alike. The paper's Secs. 3.3 and 3.5 supply the measured
$\Delta_{75}$ and exploit tails that place the deployed judges inside these
regimes.
