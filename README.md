# A Step Ahead of Drought — Forecasting Global Water Storage (Zindi / ITU)

Predict next-month Total Water Storage (TWS) at a global 1° grid, one month ahead,
from NASA GRACE-derived TWS, SPEI (1/3/6/12-month) and soil moisture. Used as a
practice project for **data visualization, GIS, regression, and feature engineering**.

## Task

- One row per `(time, lat, lon)`. `target` = `TWS_t` at the next calendar month.
- Leaderboard metric: **RMSE** (Phase 1, 50% of final score). Phase 2 scores the
  report on AI Trustworthiness (30%, per `docs/Trustworthiness_Evaluation.pdf`) and
  Innovation/practicality (20%).
- Train: 2,154,021 rows (2002-05 to 2015-08). Test: 280,961 rows, 18 months, of which
  **~66.5% of `TWS_t` is masked** (see `TWS_t_masked` in Test.csv).
- Leaderboard leader's score (reported 2026-09-03): **0.559 RMSE**.
- **Best real submission so far: 0.753811 RMSE** (2026-09-06, native missing-value
  handling in `HistGradientBoostingRegressor` instead of median-imputing — current
  default pipeline, `python -m src.train`). First real submission scored 0.7822
  (2026-09-03) — see Progress log and
  `notebooks/05_leaderboard_gap_investigation.ipynb` for why that came in much
  worse than the internal validation number at the time (0.5994). Current
  internal proxy metric (honest, mask-augmented): **mean 0.7101 RMSE over 5
  masking realisations**, via `evaluate.mask_augmented_horizon_matched_split` —
  much closer to the real score than earlier proxies, but still not exact; see
  Progress log's 2026-09-04 entries for the full proxy history.

## Competition constraints (binding — read before adding any feature)

These come from two data-leak incidents and their follow-up clarifications
(`docs/chats/`, full history there). They are stricter than they look:

1. **No information from month t+1 or later, from any source.** Includes external
   data (ERA5, Copernicus GDO) and the test file's *own* other rows.
2. **Masked test `TWS_t` rows may only be filled using that cell's strictly earlier
   observed months** (last observed ≤ t). Averaging over all observed test months
   (including later ones) is a confirmed leak — see
   `docs/chats/Clarification does the masked-row fill count as t+1 information.txt`.
   Implemented as `src/features.py::backward_fill_tws`/`fill_masked_tws`, wired into
   `src/train.py`'s test-set prediction step.
3. **`lat`/`lon` must never be a model input** (raw, encoded, or embedded). They
   **are** permitted as a lookup key: own-cell history, neighbourhood aggregation, or
   sampling an external gridded covariate at that location.
4. **Neighbouring cells' TWS at months ≤ t is explicitly permitted** — neighbourhood
   means, local deviations, and grid/CNN-style models over the observed field are all
   allowed. This is the natural home for the GIS/spatial component of this project.
5. **No GRACE/TWS-derived external product, ever** (e.g. GTWS-MLrec, GRAiCE, GDO's
   own GRACE TWS Anomaly layer) — regardless of date. This is a hard disqualifier,
   separate from the "date ≤ t" rule that governs everything else.
6. External data (ERA5, Copernicus GDO/EDO, climate indices) is allowed later
   (per user decision, scoped to "provided columns only" for now) but must be
   reported to Zindi and listed on the challenge's "additional data" page, and every
   derived feature needs a per-row source-date ≤ t audit before use. Two more
   constraints from `docs/rules.txt`, easy to miss: the source **must be available
   within one month of acquisition** at real inference time (rules out ERA5 final
   reanalysis for recent months; ERA5T/ERA5-Land near-real-time products are fine),
   and **AutoML tools are banned** (FLAML, TPOT, auto-sklearn, etc.) — plain random
   search over `HistGradientBoostingRegressor`'s own hyperparameters is not AutoML
   and remains fine.

## Project layout

See `structuring-ml-projects` skill for the rationale. `src/` currently reproduces
the official starter notebook's benchmark, unmodified — that's the baseline
everything else in this project must beat (see `RESOURCES.md`).

```
data/{raw,processed}/   raw/ is gitignored (large files) - Train.csv, Test.csv,
                        SampleSubmission.csv already in data/raw/
notebooks/              00_official_starter_reference.ipynb kept as-is; 01_+ = ours
src/                    config, data, features, model, evaluate, train
tests/                  mirrors src/
docs/                   overview.txt, rules.txt, data_dictionary.txt,
                        Trustworthiness_Evaluation.pdf, chats/ (forum clarifications)
models/, outputs/       gitignored except .gitkeep
```

Run the baseline: `python -m src.train` — loads data, chronological fit/val split,
fits the HistGBR baseline, prints RMSE/MAE/R² vs. the persistence baseline, writes
`outputs/submission.csv`.

Run tests: `pytest`

## Progress

- 2026-09-03 — Project skeleton created. Ported the official starter notebook's
  persistence baseline and HistGBR pipeline into `src/` unmodified (this *is* the
  benchmark, not an experiment — nothing to compare it against yet). No original
  feature engineering or modeling has been attempted.
- 2026-09-03 — `python -m src.train` and `notebooks/02_baseline_submission.ipynb`
  both reproduce the official starter notebook's saved benchmark exactly: RMSE
  0.6073 vs. persistence 0.6621 (MAE 0.4287 vs 0.4552, R² 0.460 vs 0.358) on the
  chronological validation split (2012-04 to 2015-08). First `outputs/submission.csv`
  generated and ready to upload. `notebooks/01_eda.ipynb` ran the planned EDA:
  raw single-feature correlation with `target` ranks `TWS_t` (0.80) far above SPEI/
  soil moisture, with `SPEI_12_t`/`SPEI_06_t` (~0.37-0.38) beating `SPEI_01_t`
  (0.20); and confirmed the test mask is bimodal, not uniform — 6 of 18 months are
  fully observed, the other 12 are >99.5% masked, so they likely dominate the
  leaderboard error and are worth scoring separately.
- 2026-09-03 — `notebooks/03_masked_tws_fill.ipynb`: simulated Test.csv's masking
  pattern on the validation split (since real test targets are hidden) and
  confirmed masked rows carry ~55% more RMSE than observed rows under the
  placeholder fill. Validated the compliant backward-anchor fill against it:
  11.2% RMSE improvement overall, 13.6% on masked rows, 0% change on observed
  rows (win on every metric component, no regression anywhere) — graduated to
  `src/features.py::backward_fill_tws`/`fill_masked_tws`, unit-tested, and wired
  into `src/train.py`. All 186,913 masked test rows got a valid anchor (0 fell
  back to median-impute). `outputs/submission.csv` regenerated with the fix.
- 2026-09-03 — `notebooks/04_spatial_neighbour_features.ipynb`: built a 500km
  great-circle neighbour graph over the 15,715 grid cells (87 neighbours/cell on
  average, only 2 fully isolated) and added `tws_neighbour_mean`/
  `tws_local_deviation` (same-month spatial aggregation, permitted per the 19 Aug
  Zindi ruling). Controlled comparison (same model/split, only these 2 columns
  differ): RMSE 0.6073 → 0.5994 (1.3%), MAE and R² also improved, no regression —
  graduated to `src/features.py::build_spatial_adjacency`/
  `add_neighbourhood_features`, unit-tested, wired into `src/train.py`.
  Internal validation RMSE at the time: 0.5994 (vs. persistence 0.6621) — **this
  number turned out to be badly optimistic, see the next entry.**
  `outputs/submission.csv` regenerated with both fixes combined and uploaded to
  Zindi.
- 2026-09-03 — **Real leaderboard score: 0.7822 RMSE**, far worse than the 0.5994
  internal estimate. Root-caused with `systematic-debugging`
  (`notebooks/05_leaderboard_gap_investigation.ipynb`): ruled out a submission-
  format bug (IDs match exactly) and anchor staleness (avg 2.6, max 6 months on
  the real Test.csv — not the culprit); found that `time_train_val_split` pools
  forecast horizons far more evenly than Test.csv's real deployment mix (78% of
  test rows are 10-40 months from the training cutoff, where validation RMSE
  nearly doubles in variance terms, 0.494 → 0.732). Implemented
  `evaluate.compute_test_horizons`/`horizon_matched_split` (unit-tested,
  graduated) to fix the validation protocol. Re-scored: **honest internal RMSE is
  0.6532** — much closer to the real score, and now this project's primary proxy
  metric. Re-checked the neighbourhood feature under it: still wins (0.6584 →
  0.6532, 0.8%, down from the 1.3% the old pooled metric showed, but still real).
  Residual gap (0.6532 vs 0.7822) is plausibly genuine 2016-2018 distribution
  shift that Train.csv (ending 2015-08) can't be validated against directly.
  `src/train.py` now reports both the old (reference-only) and honest metrics.
- 2026-09-04 — `notebooks/06_long_horizon_features.ipynb`: built and validated two
  techniques aimed at the long-horizon regime (seasonal climatology mean + z-score
  deviation per cell/calendar-month, source: a DrivenData streamflow-forecasting
  winner write-up; a long-window per-cell trend slope, leakage-safe design pattern
  sourced from featuretools' `RollingTrend` gap/window docs, implemented in plain
  pandas - featuretools itself evaluated and not adopted, see RESOURCES.md).
  **Mixed result, not a clean win**: every climatology variant improves RMSE and
  R2 (climatology+trend combined: RMSE 0.6532 -> 0.6463, 1.06%, this project's
  second-largest single-change win) and specifically helps the long-horizon
  bucket (0.7048 -> 0.6953), but regresses MAE ~1.4-1.7% consistently across
  every variant - a real trade-off, not noise. Trend alone is a clean loss; it
  only helps in combination with climatology. Given the already-known,
  unexplained gap between `horizon_matched_split` and the real leaderboard
  (0.6532 vs 0.7822), decided to defer the graduation call to real Zindi scores
  rather than the internal proxy alone: generated 3 candidate submission files
  (`outputs/submission_climatology.csv`, `submission_trend.csv`,
  `submission_climatology_trend.csv`) to upload and compare (5/day, 200 total
  limit - `docs/rules.txt`).
- 2026-09-04 — **All 3 real Zindi scores back**: climatology alone **0.7778**
  (beats the prior best 0.7822 by 0.56%), trend alone 0.7834 (worse), climatology
  + trend combined 0.7845 (worse than baseline AND worse than trend alone). The
  internal proxy had ranked climatology+trend combined as best (1.06%
  improvement) - the real leaderboard inverted that entirely, though it agreed on
  each feature's *individual* sign (climatology helps, trend hurts). **Graduated:
  `src/features.py::add_seasonal_climatology_features`** (mean + z-score
  deviation only, TDD - 5 new unit tests in `tests/test_features.py`), wired into
  `src/train.py`. Trend is a confirmed negative result, not graduated.
  `outputs/submission.csv` regenerated with climatology included (honest-metric
  RMSE now 0.6505, reproducing the notebook's number exactly).
- 2026-09-04 — Spawned an Opus-model review of the above (user request) to plan
  next steps. Its key finding: 66.5% of Test.csv's `TWS_t` is masked-then-
  backward-filled (frozen at its last observed value), while
  `horizon_matched_split` always validates on fully-observed `TWS_t` - a
  train/serve skew that plausibly explains why the internal proxy inverted the
  climatology+trend interaction ranking. Recommended fixing the proxy (P0)
  before further feature/tuning/external-data work. Also caught and fixed:
  a stale 94.5%->77.7% figure in `evaluate.py`'s docstring, a duplicated
  RESOURCES.md heading, and this README's Task summary still citing the
  superseded 0.6532 proxy number.
- 2026-09-04 — **P0 (`notebooks/07_mask_aware_validation.ipynb`)**: built
  `evaluate.measure_masking_pattern`/`simulate_masking`/
  `mask_aware_horizon_matched_split` and `features.build_all_features`
  (extracted from `src/train.py` so validation and production run the
  identical feature pipeline - TDD, new unit tests in both test modules).
  Validates Test.csv's real masking pattern (measured: 66.7% of months
  near-fully masked, 99.8% of rows within them - matches `01_eda.ipynb`'s
  estimate almost exactly) onto the validation fold before computing derived
  features. **Partial win**: closed ~79% of the old proxy's gap to the real
  leaderboard (baseline config: 0.6532 -> 0.7552 vs. real 0.7822 - old gap
  0.129, new gap 0.027) and correctly ranks climatology as best - but still
  ranks climatology+trend combined as 2nd-best when reality says it's the
  worst option, so the specific interaction-inversion mystery is not fully
  closed (see notebook's Gate decision for the full reasoning). **Graduated
  as the new primary proxy anyway** (`src/train.py` now reports it) since the
  absolute-calibration win stands on its own. Policy going forward: gate
  backward-looking/seasonal/spatial features on this proxy; anything reading
  or extrapolating the recent `TWS_t` trajectory still needs a real Zindi
  submission before graduating (per the Opus review's Tier A/Tier B split).
- 2026-09-04 — **P1 (`notebooks/08_anchor_age_feature.ipynb`)**: added
  `features.compute_anchor_age` (`months_since_anchor`) and
  `evaluate.augment_with_simulated_masking`/`mask_augmented_horizon_matched_split`
  (TDD, new unit tests) - the fit half now also gets simulated masking, not just
  validation, since `months_since_anchor` would otherwise be a constant 0 in
  training (Train.csv is always fully observed) and unlearnable. Checked over 5
  independent masking realisations, not one: fit augmentation alone beats the P0
  reference (0.7469) in 5/5 seeds (mean 0.7201, ~3.6%); adding
  `months_since_anchor` on top beats that in 5/5 seeds too (mean 0.7126, ~4.6%
  total) - clean wins on every metric component in every seed, unlike the trend
  feature's trade-offs. Judged mechanistically safer than trend (this fixes a
  train/serve mismatch, the same class of fix P0 itself was, rather than
  extrapolating a trajectory) but still treated cautiously per the Tier A/B
  policy: **not wired into `src/train.py`'s default pipeline yet**.
  `outputs/submission_anchor_age.csv` (augmented training + anchor age) and
  `outputs/submission_persistence.csv` (pure persistence sanity check, `Target =
  filled TWS_t`) generated and queued for upload.
- 2026-09-04 — **Persistence sanity check scored: 0.8864 RMSE** - much worse
  than the fitted model (best: 0.7778), confirming the model is genuinely adding
  value out-of-time, not just riding the backward-filled anchor.
- 2026-09-04 — **`submission_anchor_age.csv` scored 0.7579 RMSE** - beats the
  prior best (0.7778, climatology) by ~2.6%, the largest confirmed real-world
  win since masked-fill, and this time with no proxy-inversion surprise (5-seed
  internal validation predicted a clean win, and it held up). **Graduated**:
  `src/train.py`'s final fit now trains on masking-augmented Train.csv with
  `months_since_anchor` included (`evaluate.mask_augmented_horizon_matched_split`
  is the new primary proxy, mean RMSE 0.7126 over 5 seeds, reported by
  `src/train.py`). `outputs/submission.csv` regenerated with this as the new
  default pipeline.
- 2026-09-05 — **P2 investigation (`notebooks/09_own_cell_dynamics.ipynb`)**: own-cell
  lag/rolling/AR(1) features on SPEI/soil moisture (never masked, gateable on the
  proxy alone per the Gate policy). Built the first two candidates from ad-hoc data
  exploration without first finding a source, in violation of `structuring-ml-projects`
  rule 3 - caught by the user; corrected by locating the `real-world-ml` skill's ch07
  time-series feature-escalation framework and citing it retroactively (RESOURCES.md).
  That same exploration did catch a real methodological pitfall before it caused
  damage: a fixed 1-/3-month calendar lag only covers 72%/44% of real Test.csv rows
  (its 18 months aren't contiguous), which validating only against Train.csv's dense
  panel would have hidden. Fixed via a `prior_value`/`prior_age` design (generalises
  P1's `compute_anchor_age` pattern) - ~99% real coverage. Three candidates tested
  against `mask_augmented_horizon_matched_split` (5-seed mean): soil-moisture
  prior+age (flat, 0.7126->0.7126), SPEI_12 prior+age (0.7126->0.7117, ~0.1%, only
  3/5 seeds), SPEI_12 AR(1)-deviation escalation (0.7126->0.7138, net loses, 4/5
  seeds worse).
- 2026-09-05 — Initially called SPEI_12 prior+age "too weak to graduate" on the proxy
  signal alone (only 3/5 seeds, small margin) - the user challenged that call,
  correctly pointing out this project has never confirmed with a real submission
  whether a *negative*-looking Tier-A proxy verdict is actually trustworthy (the Gate
  policy's proxy-alone shortcut was only ever validated on a *positive* case,
  climatology). Generated `outputs/submission_spei12_prior.csv` (default pipeline +
  only `spei12_prior_value`/`spei12_prior_age`, single controlled change) to check.
  **Real Zindi score: 0.755129 RMSE** - beats the prior best (0.7579, anchor-age) by
  ~0.37%, confirming the weak/inconsistent proxy signal *was* real. **Graduated**:
  `src/features.py::compute_prior_reading` (generalised beyond just SPEI_12 - takes
  any never-masked column), wired unconditionally into `build_all_features` so every
  proxy variant in `evaluate.py` picks it up automatically, TDD unit tests added.
  `outputs/submission.csv` regenerated with this as the new default pipeline. Lesson
  for this project: a proxy-alone "not graduated" call on a Tier-A feature with a
  small-but-consistent-direction signal should default to a real-submission check
  before closing the investigation, not skip straight to "negative result" - see
  RESOURCES.md's `notebooks/09_own_cell_dynamics.ipynb` entry for the full reasoning.
- 2026-09-05 — **P3 (`notebooks/10_hyperparameter_tuning.ipynb`)**: random search over
  `HistGradientBoostingRegressor`'s `learning_rate`/`max_depth`/`min_samples_leaf`/
  `l2_regularization`/`max_leaf_nodes` (Bergstra & Bengio 2012; `max_iter` fixed at a
  generous ceiling and left to the model's own early stopping, per `hands-on-ml` ch07 -
  not searched directly). Round 1 (25 candidates, single-seed screen + 5-seed
  confirmation of the top 3): only one candidate beat the current defaults (0.7117 ->
  0.7106, ~0.15%, 4/5 seeds), and it landed at the search floor for `max_leaf_nodes` -
  per `real-world-ml` ch04's refinement rule, re-ran with `learning_rate` lowered
  further. Round 2: all 3 top candidates beat defaults by a clearer margin (best:
  0.7082, ~0.49%, 4/5 seeds) - again near a boundary (`learning_rate` close to the new
  floor). Rather than keep expanding, confirmed the best round-2 candidate with a real
  submission first (this project's established practice since P2).
  **Real Zindi score: 0.758156 RMSE - worse than the current best (0.755129)**, despite
  the proxy's clear win. A genuine proxy-real inversion on the hyperparameter axis.
  Working hypothesis (not independently verified): `HistGradientBoostingRegressor`'s
  built-in `early_stopping` picks its stopping point from a plain RANDOM internal
  holdout, not this project's horizon-matched/mask-aware validation structure - at a
  very low learning rate the exact stopping point matters a lot, and the internal
  random split's signal may not transfer to Test.csv's real deployment distribution.
  **Not graduated** - `src/model.py` unchanged, logged as a negative result.
- 2026-09-06 — **P3 retry (`notebooks/11_manual_early_stopping.ipynb`)**: tested P3's
  working hypothesis directly - verified against scikit-learn 1.7.2's actual source
  that `HistGradientBoostingRegressor`'s early stopping does draw a plain random IID
  holdout (`train_test_split(..., shuffle=True)`) when none is supplied, and that this
  sklearn version's `fit()` accepts `X_val`/`y_val` to override it. Implemented a
  manual, horizon-aware early-stopping holdout (a second, chronological
  `time_train_val_split` carved out of the fit portion) and re-ran P3's exact round-1
  search space (same seed) against it. **Initial reading was wrong**: on a 5-seed
  confirmation, `current_defaults` did lose to the default early-stopping baseline in
  5/5 seeds (0.7176 vs 0.7117) and no searched candidate beat it (best: 0.7163) - this
  was first logged as "hypothesis refuted." An Opus-model review (this project's
  established practice for a key decision point, per the 2026-09-04 precedent) found
  that comparison was **confounded**: the chronological inner split silently removes
  the ~17 most recent training months from the manual-ES arm only (the default-ES arm
  holds out a random 10% of *rows*, still spanning the full time range), so the
  observed loss is explained by "17 fewer months of recent training data" alone,
  without any contribution from the early-stopping mechanism - and the 5/5-seed
  uniformity, which read as robust, is actually consistent with a *constant* confound
  (the chronological cut is identical across all 5 seeds).
  The review also found a cleaner, confound-free fact already sitting in the same
  results: `current_defaults__default_es`'s `mean_n_iter=300` sits exactly at
  `make_baseline_model()`'s `max_iter` ceiling in all 5 seeds - **early stopping never
  fires for the production defaults**, so it cannot be responsible for anything in the
  current pipeline. Two follow-up checks confirmed this cleanly: (1) refitting P3's
  actual submitted round-2 candidate (`learning_rate=0.0039, max_depth=10,
  min_samples_leaf=63, l2_regularization=0.239, max_leaf_nodes=48`) at `max_iter=1000`
  with sklearn's default early stopping gave `n_iter_=1000` - also hit its ceiling
  exactly, so early stopping did not influence the one candidate that was actually
  real-world tested either. P3's real proxy-real inversion therefore still has no
  confirmed explanation - the leading remaining suspect is a genuine distribution
  shift between the proxy's validation window and Test.csv's real 2016-2018 period,
  not yet investigated. (2) Since early stopping never triggers anyway,
  `make_baseline_model()` silently discards 10% of training rows on a holdout that
  never does anything useful; testing `early_stopping=False` (recovering those rows,
  `max_iter=300` unchanged) gave a null result (mean 0.7120 vs. 0.7117) - not a
  meaningful improvement, within noise. **Not graduated** - `src/model.py` unchanged.
  Corrected takeaway: the *specific* nested-split procedure in this notebook loses
  (data-deletion confound, not an early-stopping effect), the early-stopping mechanism
  itself is now confirmed inert for both the production defaults and P3's actual
  submitted candidate, and the true cause of P3's inversion remains an open question.
- 2026-09-06 — **P5 (`notebooks/12_native_missing_value_handling.ipynb`)**: reviewed
  Kaggle competition write-ups (M5 Forecasting - Accuracy's 1st place solution;
  general GBM technique notebooks) for techniques this project might be missing, per
  user request. Most promising, cheapest-to-test idea: `make_baseline_model()` ran
  every feature through `SimpleImputer(median)` before fitting, discarding the "this
  was missing" signal for `tws_neighbour_mean`/`tws_local_deviation` (NaN when no
  observed neighbour that month), `tws_climatology_mean`/`tws_climatology_deviation`
  (NaN for a cell's first occurrence of a calendar month - measured real rate: **30%**
  for the deviation column, 19% for the mean, much higher than expected) and
  `spei12_prior_value`/`spei12_prior_age` (NaN with no earlier reading) -
  `HistGradientBoostingRegressor` has native missing-value support (confirmed in
  scikit-learn 1.7.2's own docstring: the tree grower learns per split whether missing
  values go left or right, based on potential gain - the same idea LightGBM/XGBoost
  use). Tested removing the imputer entirely on the 5-seed proxy: won 3/5 seeds
  clearly, within noise on the other 2 (mean 0.7117 -> 0.7101, ~0.23%) - a
  weak-but-consistent signal, same shape as P2's SPEI_12 prior-reading case, so per
  this project's established policy it was checked with a real submission rather than
  written off. **Real Zindi score: 0.753811 RMSE** - beats the prior best (0.755129)
  by ~0.17%, and unlike P3, the proxy's direction held on the real leaderboard.
  **Graduated**: `src/model.py::make_baseline_model` now returns a bare
  `HistGradientBoostingRegressor` (no `Pipeline`/`SimpleImputer`), `src/train.py`
  unchanged otherwise since `features.select_base_features` already returns NaN
  feature values as-is. `outputs/submission.csv` regenerated with this as the new
  default pipeline.

## Next steps

- [x] Run `python -m src.train`, confirm it reproduces the current leaderboard
      benchmark score before trusting any further comparison against it.
- [x] EDA notebook (`notebooks/01_eda.ipynb`): spatial patterns (GIS/dataviz),
      temporal dynamics per cell, single-feature correlation with `target` for
      each of `SPEI_01/03/06/12_t`, `SOIL_MOISTURE_t`.
- [x] Score the baseline separately on the 6 fully-observed test months vs. the 12
      near-fully-masked ones — real test targets are hidden, so
      `notebooks/03_masked_tws_fill.ipynb` simulates Test.csv's exact masking
      pattern on the validation split instead. Confirmed: masked rows carry ~55%
      more error than observed rows under the placeholder fill.
- [x] Implement the compliant masked-`TWS_t` fill (strictly-backward per-cell
      anchor) and validate it beats the current median-imputer placeholder.
      **Won** (`notebooks/03_masked_tws_fill.ipynb`): 11.2% RMSE improvement
      overall, 13.6% on masked rows, 0% change on observed rows. Graduated to
      `src/features.py`, wired into `src/train.py`, `outputs/submission.csv`
      regenerated with it.
- [x] Feature engineering: own-cell TWS/SPEI history (lags, rolling stats,
      climatology) — **partially done**. Climatology graduated (see 2026-09-04
      entries below); a TWS trend/rolling-stat variant was tried and rejected
      (confirmed negative on the real leaderboard). SPEI/soil-moisture lags not
      yet tried — see Next steps.
- [x] Feature engineering: spatial neighbourhood aggregates (GIS component).
      **Won** (`notebooks/04_spatial_neighbour_features.ipynb`): 1.3% RMSE
      improvement. Graduated to `src/features.py`, wired into `src/train.py`.
- [x] **Submit `outputs/submission.csv` to Zindi** — scored 0.7822 RMSE vs.
      leader's 0.559. Root-caused the gap from our 0.5994 internal estimate (see
      Progress log) — the internal proxy metric is now 0.6532 via
      `evaluate.horizon_matched_split`. **Use this metric, not the old pooled
      one, to gate every feature from here on.**
- [x] Feature engineering for long forecast horizons specifically (10-40 months,
      78% of Test.csv): seasonal climatology, longer-window trends per cell.
      Built and validated in `notebooks/06_long_horizon_features.ipynb` — mixed
      result (wins RMSE/R2, regresses MAE, see Progress log). Not graduated yet.
- [x] Upload `outputs/submission_climatology.csv`, `submission_trend.csv`,
      `submission_climatology_trend.csv` to Zindi and compare real leaderboard
      scores. **Climatology won (0.7822 → 0.7778), graduated.** Trend lost
      standalone and combined — confirmed negative result, not graduated.
- [x] **P0 — mask-aware validation** (Opus review recommendation, see Progress
      log): `notebooks/07_mask_aware_validation.ipynb` simulates Test.csv's real
      TWS_t masking pattern on the validation fold before computing features.
      Closed ~79% of the old proxy's gap to the real leaderboard; graduated as
      the new primary proxy (`evaluate.mask_aware_horizon_matched_split`,
      `src/train.py` updated). Interaction-ranking mystery only partly
      resolved — see that notebook's Gate decision.
- [x] **P1 — anchor-age feature + training-time masking augmentation**: built
      `features.compute_anchor_age` (`months_since_anchor`) and
      `evaluate.augment_with_simulated_masking`/`mask_augmented_horizon_matched_split`
      (also simulates masking on the FIT half, not just validation — otherwise
      the age feature is constant 0 in training and unlearnable). Validated over
      5 independent masking realisations in
      `notebooks/08_anchor_age_feature.ipynb`: fit augmentation alone beats the
      P0 reference (0.7469) in 5/5 seeds (mean 0.7201); adding
      `months_since_anchor` on top beats that in 5/5 seeds too (mean 0.7126) —
      clean wins, no MAE trade-offs, unlike the trend feature. **Won for real**
      (0.7778 → 0.7579, ~2.6%) — graduated as the default pipeline. Persistence
      sanity check (0.8864) confirms the model beats trivial persistence by a
      wide margin.
- [x] **P2 — own-cell dynamics on SPEI/soil moisture, not TWS**: investigated in
      `notebooks/09_own_cell_dynamics.ipynb` following `real-world-ml`'s
      time-series feature-escalation ladder. Found and fixed a coverage pitfall
      (fixed calendar lags cover only 72%/44% of real Test.csv rows; a
      `prior_value`/`prior_age` design generalising P1's anchor-age pattern
      fixes that to ~99%). Of 3 candidates tested, 2 (soil-moisture prior+age;
      SPEI_12 AR(1)-deviation) didn't clear the proxy and weren't worth a real
      submission; the third (SPEI_12 prior+age) had a weak/inconsistent proxy
      signal but was confirmed on the real leaderboard (0.7579 -> 0.755129,
      ~0.37%) after the user challenged the proxy-only "not graduated" call —
      **graduated** as `src/features.py::compute_prior_reading`, wired into
      `build_all_features`, now the default pipeline. A TWS trend computed over
      strictly *observed* (pre-mask) history only was not attempted — out of
      scope, since it reads TWS's own trajectory and would need a real Zindi
      submission under the Tier B policy, same as the already-rejected
      long-window trend feature.
- [x] **P3 — hyperparameter tuning**: random search (Bergstra & Bengio 2012) over
      `HistGradientBoostingRegressor`'s standard boosting levers, in
      `notebooks/10_hyperparameter_tuning.ipynb`. Two rounds both found real proxy
      improvements (round 2's best: 0.7117 -> 0.7082, ~0.49%, 4/5 seeds) but a real
      Zindi submission on the best candidate scored **worse** (0.758156 vs. the
      current best 0.755129) - a genuine proxy-real inversion; initial working
      hypothesis was `HistGradientBoostingRegressor`'s internal `early_stopping` using
      a random (non-horizon-aware) holdout - **checked directly in P3 retry below and
      ruled out**. **Not graduated** - `make_baseline_model` is unchanged, still the
      original starter-notebook defaults.
- [x] **P3 retry — early-stopping hypothesis, checked directly**: verified against
      scikit-learn 1.7.2 source that early stopping does use a random IID holdout by
      default, and that `X_val`/`y_val` can override it (`notebooks/11_manual_early_stopping.ipynb`).
      A manual horizon-aware version of that holdout initially looked like it refuted
      the hypothesis (lost 5/5 seeds) - an Opus-model review found that comparison
      confounded the fix with deleting the ~17 most recent training months, and
      pointed at a cleaner fact already in the results instead: `mean_n_iter=300` sits
      exactly at the ceiling for the production defaults, meaning early stopping never
      fires there at all. Confirmed early stopping also never fired for P3's actual
      submitted candidate (`n_iter_=1000`, also its ceiling) - so the mechanism is
      ruled out as P3's inversion cause for both cases that matter. A follow-up test
      (`early_stopping=False`, recovering the 10% of rows the inert holdout was
      discarding) gave a null result (0.7120 vs. 0.7117). **Not graduated** -
      `src/model.py` unchanged. P3's real proxy-real inversion still has no confirmed
      explanation; leading suspect is a distribution shift between the proxy's
      validation window and Test.csv's real 2016-2018 period.
- [ ] **P4 — external data phase (ERA5), paused**: investigated the compliance
      question 2026-09-05 (8 days before close). The 13 Aug organizer FAQ confirms
      ERA5T/final-ERA5-as-documented-proxy for non-TWS variables (source date ≤ t) is
      permitted for "historical validation" — safe on Train.csv today. But a live
      check of zindi.world/discussions confirmed 4 follow-up questions (24-29 Aug,
      plus a new one 4 Sep re: NASA GLDAS) asking to extend this explicitly to
      Test.csv rows, or to cover other GDO products/climate indices, all still show
      **0 answers** as of this check — not just missing from our local
      `docs/chats/` archive. A competitor (`uzbtrust`, 31 Aug) independently
      confirmed the same open gap while reporting ~0.70 RMSE using ERA5. Given this
      project's established norm (other participants in this same challenge
      explicitly wrote "treat as prohibited unless approved" when facing unanswered
      scope questions — see `docs/chats/Question for the orginizers.txt`, `worries
      questions.txt`) and two real prior disqualification-level leak incidents in
      this challenge, **user decision: do not build ERA5 features for
      Test.csv/submission until this is explicitly confirmed** — Train.csv-only
      exploration remains fine if useful later, but P4 is paused, not started, and
      P3 takes priority given the 8-day close. `docs/rules.txt` still constrains any
      future external-data work: must be available within one month of acquisition
      (rules out ERA5 final reanalysis for genuinely recent months; permits
      ERA5T/ERA5-Land near-real-time), and AutoML tools (FLAML/TPOT/auto-sklearn) are
      banned — plain random search over `HistGradientBoostingRegressor`'s params for
      P3 is fine.
- [x] **P5 — native missing-value handling**: reviewed Kaggle write-ups (M5
      Forecasting - Accuracy 1st place solution, general GBM technique notebooks) for
      techniques this project might be missing, per user request
      (`notebooks/12_native_missing_value_handling.ipynb`). Found `make_baseline_model`
      was median-imputing several features with genuine, informative missingness
      (climatology deviation: 30% NaN, climatology mean: 19%, measured directly - much
      higher than expected) instead of using `HistGradientBoostingRegressor`'s native
      NaN support. 5-seed proxy showed a weak-but-consistent win (3/5 seeds clear,
      0.7117 -> 0.7101) - checked with a real submission per this project's policy for
      this signal shape. **Real Zindi score: 0.753811 RMSE**, beating the prior best
      (0.755129) by ~0.17%, proxy direction held (unlike P3). **Graduated**:
      `src/model.py::make_baseline_model` returns a bare `HistGradientBoostingRegressor`
      (no `Pipeline`/`SimpleImputer`) - now the default pipeline.
