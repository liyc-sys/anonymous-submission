# Decoding Specification

The decoding settings frozen for every lineage; per-lineage judge settings
differ, actor settings do not.

| Component | Setting | Value |
|---|---|---|
| Actor, all arms and benchmarks | temperature | $T = 0.4$ |
| Judge, WebGen lineage | temperature / top-$p$ | $T = 0.2$, top-$p$ = 0.9 |
| Judge, WebGen lineage | confidence threshold | $\gamma = 0.6$ |
| Judge, VLM lineages (V2W, I2C) | temperature | $T = 0$ |

Notes.

- The WebGen judge's $\gamma$ is its confidence threshold in the official
  ordering after repetition: pairwise adjudication below $\gamma$ abstains to
  retention or official adjudication rather than a forced call (the paper's
  allocation rule; abstention is analogous in spirit to confidence-based
  deferral, not an implementation of it).
- Actor decoding is identical across arms within a benchmark, so cross-arm
  contrasts do not mix decoding settings; rollouts differ by seed only.
- The shipped harness CLI defaults are placeholders for reproduction against
  any OpenAI-compatible endpoint; the values in the table above are the ones
  used for every frozen measurement in `data/`. Judges additionally obey the
  evolution contract stated in their docstrings: one frozen judge model per
  benchmark, no access to evaluation artifacts, bounded call budgets.
- Renderers and evaluators: Playwright with headless Chromium for rendering
  and page inspection; the benchmarks' released evaluators (Interaction2Code:
  Firefox/Selenium replay); vLLM for serving the frozen models.
