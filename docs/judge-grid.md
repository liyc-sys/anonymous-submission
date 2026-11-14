# Judge Grid: Three Independent Lineages, Full Results

Supports Secs. 3.3 and 3.5 of the paper ("full results in the supplement").
Artifact-level judge accuracy against stable official aggregates, for every
variant of the three judge lineages. Recomputed from the shipped per-instance
data by `analysis/verify_webgen_judge_ablation.py` (WebGen),
`analysis/verify_v2w_gan_rounds.py` (Vision2Web) and the frozen case
records in `data/i2c/cases/` (Interaction2Code).

## Main grid

| Bench | Judge variant | Pairwise acc. | $R_k$ / Spearman | HJ/LB |
|---|---|---:|---:|---:|
| WebGen | naive subset | 0.793 | 0.737 | 2.8% |
| WebGen | + calibrated agg. | 0.850 | 0.849 | 0.0% |
| WebGen | + fallback probe | 0.867 | 0.862 | 0.0% |
| WebGen | **+ challenge** | **0.917** | **0.917** | **0.0%** |
| V2W | raw VLM | 0.758 | — | — |
| V2W | + viewport vote | 0.775 | — | — |
| V2W | **+ challenge** | **0.842** | — | — |
| I2C | raw VLM | 0.658 | 0.367 | 9.4% |
| I2C | + evolution | 0.749 | 0.531 | 5.5% |
| I2C | **+ cex. calibration** | **0.759** | 0.570 | 3.1% |

Column 4 is Spearman correlation for WebGen and shortlist recall $R_2$ for
I2C. HJ/LB is the rate of shortlisted candidates that are high-judge but
benchmark-unsupported (WebGen counts: 4/145, 0/72, 0/72, 0/51). Pair sets:
WebGen 519 sites / 1112 decidable pairs; V2W 120 held-out pairs; I2C 512
candidates.

## Binned accuracy (WebGen, same 1112 pairs)

| $\lvert\Delta Q_B\rvert$ bin | [0.05,0.1) | [0.1,0.2) | [0.2,0.3) | [0.3,0.5) | [0.5,1.0] |
|---|---:|---:|---:|---:|---:|
| Pairs | 169 | 317 | 211 | 287 | 128 |
| naive subset | 0.615 | 0.711 | 0.794 | 0.899 | 0.992 |
| calibrated agg. | 0.633 | 0.808 | 0.851 | 0.956 | 1.000 |
| + challenge | 0.743 | 0.890 | 0.948 | 0.990 | 1.000 |

The binned companion of the paper's gap-curve figure: accuracy rises with the
true gap for every variant, and evolution lifts the whole curve.

## Binned accuracy (V2W archived exam, 120 pairs, 30 per bin)

| Gap bin | 0.03–0.06 | 0.06–0.1 | 0.1–0.2 | >0.2 |
|---|---:|---:|---:|---:|
| Pairs | 30 | 30 | 30 | 30 |
| raw VLM | 0.667 | 0.667 | 0.733 | 0.967 |
| + viewport vote | 0.600 | 0.733 | 0.800 | 0.967 |
| + challenge | 0.833 | 0.800 | 0.767 | 0.967 |

The same gradient in coarser bins; these support the rough
$\sigma_J \approx 0.08$ quoted in the paper.

## Thurstone fits

One-parameter fits $\Phi\bigl(\Delta / (\sigma_J \sqrt 2)\bigr)$ over the
WebGen bins return $\sigma_J = 0.209$ (naive), $0.144$ (calibrated) and
$0.088$ (challenge) — $\Delta_{75}$ of $0.20$, $0.14$ and $0.08$,
respectively. The calibrated judge's $0.144$ independently matches the
$\sigma_{\mathrm{judge}} = 0.141$ of the paper's Setup variance decomposition
(a within-instrument proxy of that evaluator should). The same-arm
cross-seed refit (604 pairs — the regime Corollary A actually assumes, free
of the harness-indexed bias cross-arm pairs carry) returns
$\sigma_J = 0.081$. Fits minimize $n$-weighted squared error over $\sigma_J$;
exam pairs are not a random sample of deployment pairs, which is the caveat
the paper carries.

## Computation protocol

The gap-curve figure and the bins above were recomputed offline from frozen
per-site task labels. WebGen-Bench: the unified 519-site set (three arms
$\times$ 83 cases $\times$ 2 seeds, sites covering diagnostic indices 0–3),
same-case cross-arm cross-seed pairs, official ties dropped, judge ties
counted $0.5$, yielding 1112 pairs binned by $\lvert\Delta\,\text{full-acc}
\rvert$; judge variants are the diagnostic-subset scorers of the main grid
above. V2W: the archived 120-pair held-out exam of the judge lab. I2C: the
frozen evidence pack (`data/i2c/`), G1/G2/G3 judge generations over 512
candidates.
