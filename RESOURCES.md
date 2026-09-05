# RESOURCES

One entry per source. Every function in `src/` that implements a technique cites
one of these in its docstring.

## Official challenge documents

- **Challenge overview** (`docs/overview.txt`)
  Why: defines the target (TWS at t+1), the RMSE leaderboard metric, and the
  no-future-information rule that every feature in `src/features.py` must respect.

- **Data dictionary** (`docs/data_dictionary.txt`)
  Why: defines the raw columns (`TWS_t`, `SPEI_01/03/06/12_t`, `SOIL_MOISTURE_t`) used
  in `src/config.py::BASE_FEATURE_COLS` / `OPTIONAL_FEATURE_COLS`.

- **Official starter notebook** (`notebooks/00_official_starter_reference.ipynb`)
  Why: source of the persistence baseline (`src/model.py::make_persistence_prediction`),
  the HistGradientBoostingRegressor baseline pipeline
  (`src/model.py::make_baseline_model`), the chronological fit/val split
  (`src/evaluate.py::time_train_val_split`), and the sample_id/ID column
  inconsistency handled in `src/data.py::_normalise_columns`.

- **Zindi clarification - "Neighbouring cells' past TWS - permitted or not"**
  (`docs/chats/Neighbouring cells' past TWS — permitted or not.txt`, answered 19 Aug)
  Why: rules `src/config.py::FORBIDDEN_MODEL_FEATURES` (lat/lon/ID banned as raw model
  inputs) and permits `src/features.py::build_spatial_adjacency`/
  `add_neighbourhood_features` - neighbourhood means/local deviations of TWS at
  months <= t are explicitly permitted, coordinate values and spatial embeddings
  are not (lat/lon are used only internally to build the neighbour graph).
  Validated in `notebooks/04_spatial_neighbour_features.ipynb`: 1.3% RMSE
  improvement (0.6073 -> 0.5994), no regression on MAE/R2. A forum participant's
  self-reported 0.637 -> 0.604 from the same idea motivated trying it, but was not
  itself used as a number.

- **Zindi clarification - "Clarification does the masked-row fill count as t+1
  information"** (`docs/chats/Clarification does the masked-row fill count as t+1
  information.txt`)
  Why: rules out filling a masked test `TWS_t` with any statistic that includes
  observed months later than that row's own month (leaks the test file's own future
  observations); only a strictly backward-looking per-cell anchor is compliant.
  Implemented as `src/features.py::backward_fill_tws`/`fill_masked_tws`, validated
  in `notebooks/03_masked_tws_fill.ipynb` (11.2% RMSE improvement overall, 13.6% on
  masked rows specifically, vs. the `SimpleImputer(median)` placeholder it replaced
  in `src/train.py`), unit-tested in `tests/test_features.py`.

- **Zindi update - "Competition Update" / "Data Update"**
  (`docs/chats/Competition Update.txt`, `docs/chats/Data Update.txt`)
  Why: documents the flat-CSV release (replacing the original NetCDF release that
  contained a hard target leak), the `TWS_t_masked` column, and the removal of
  forward-looking `*_tp1` columns - explains why `src/config.py` has no `_tp1` features.

## Library documentation

- **scikit-learn `TimeSeriesSplit`** (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
  Why: standard justification for chronological (not random) train/validation splitting
  on a forecasting task, applied in `src/evaluate.py::time_train_val_split`.

## This project's own investigations

- **`notebooks/05_leaderboard_gap_investigation.ipynb`**
  Why: source of `src/evaluate.py::compute_horizons`/`compute_test_horizons`/
  `horizon_matched_split`. Root-caused (via `systematic-debugging`) why the real
  Zindi score (0.7822) came in far worse than the pooled internal validation RMSE
  (0.5994): Test.csv's forecast-horizon distribution is disproportionately long
  (78% of rows 10-40 months from the training cutoff) while `time_train_val_split`
  pools horizons much more evenly, and validation RMSE nearly doubles in variance
  terms at long horizons. The horizon-matched split gives RMSE 0.6532 - the
  project's current honest internal proxy metric; use it, not the plain split, to
  gate future features.

- **`notebooks/06_long_horizon_features.ipynb`**
  Why: built and validated seasonal climatology (mean + z-score deviation) and a
  long-window per-cell trend slope, targeting the long-horizon regime identified in
  notebook 05. Mixed result against `horizon_matched_split`: every climatology
  variant improved RMSE and R2 but regressed MAE ~1.4-1.7% consistently - a real
  trade-off, not noise - so this project deferred the graduation call to real Zindi
  submissions rather than the internal proxy alone (project decision, 2026-09-04).
  **Real leaderboard results (2026-09-04):** climatology alone 0.7778 RMSE (beats
  the prior best of 0.7822 by 0.56% - graduated to
  `src/features.py::add_seasonal_climatology_features`, unit-tested, wired into
  `src/train.py`); trend alone 0.7834 (worse than baseline); climatology+trend
  combined 0.7845 (worse than baseline AND worse than trend alone - confirmed
  anti-synergy). The internal proxy had predicted climatology+trend combined as
  the best option (1.06% improvement) - the real leaderboard inverted that
  ranking entirely, validating the decision not to trust the proxy's *interaction
  effects* even though its individual-feature signs were directionally right.
  Trend is a confirmed negative result, not graduated.

- **`notebooks/07_mask_aware_validation.ipynb`**
  Why: source of `src/evaluate.py::measure_masking_pattern`/`simulate_masking`/
  `mask_aware_horizon_matched_split` and `src/features.py::build_all_features`
  (extracted from `src/train.py` so validation and production run the identical
  feature pipeline). Followed an Opus-model review (project decision, 2026-09-04)
  of notebook 06's real-vs-proxy inversion, which found that 66.5% of Test.csv's
  `TWS_t` is masked-then-backward-filled (frozen at its last observed value) while
  `horizon_matched_split` always validated on fully-observed `TWS_t` - a plausible
  train/serve skew, especially damaging to trend-like features that read the
  recent TWS_t trajectory. Simulating Test.csv's real masking pattern (measured,
  not guessed: 66.7% of months near-fully masked, 99.8% of rows within them) onto
  the validation fold before feature computation closed ~79% of the old proxy's
  gap to the real leaderboard, and correctly ranks climatology as the best single
  feature - but still ranks climatology+trend combined as 2nd-best when reality
  says it's the worst option, so the interaction-inversion mystery is only
  partly resolved. Graduated as the project's new primary internal proxy anyway
  (the absolute-calibration win stands on its own); `src/train.py` reports it.

- **`notebooks/08_anchor_age_feature.ipynb`**
  Why: source of `src/features.py::compute_anchor_age` (`months_since_anchor`)
  and `src/evaluate.py::augment_with_simulated_masking`/
  `mask_augmented_horizon_matched_split`. Implements the Opus review's P1
  recommendation - `months_since_anchor` is only learnable if the model sees it
  vary during training, which requires simulating masking on the FIT half too
  (not just validation, as notebook 07's fix did), since Train.csv is otherwise
  always fully observed. Validated over 5 independent masking realisations: fit
  augmentation alone beats the P0 reference (0.7469) in 5/5 seeds (mean 0.7201);
  adding the age feature on top beats that in 5/5 seeds too (mean 0.7126) - no
  MAE trade-offs in any seed, unlike the trend feature. **Real leaderboard
  result (2026-09-04): 0.7579 RMSE**, beating the prior best (0.7778,
  climatology) by ~2.6% - the largest confirmed win since masked-fill, and
  unlike trend, no proxy-inversion surprise. Graduated as the default pipeline:
  `src/train.py`'s final fit now trains on masking-augmented Train.csv with
  `months_since_anchor` included, and reports
  `mask_augmented_horizon_matched_split`'s mean over 5 seeds as the primary
  proxy.

- **`notebooks/09_own_cell_dynamics.ipynb`**
  Why: P2 investigation - own-cell lag/rolling/AR(1) features on `SPEI_01/03/06/12_t`/
  `SOIL_MOISTURE_t` (never masked in Test.csv, so gateable on the proxy alone per the
  Gate policy). Found and fixed a coverage pitfall before building anything: a fixed
  1-/3-month calendar lag only covers 72%/44% of real Test.csv rows (Test.csv's 18
  months are not contiguous - some calendar months have zero rows for any cell in
  Train+Test combined), which a naive notebook validated purely against Train.csv's
  dense contiguous panel would not have caught (the same proxy-optimism risk already
  hit twice in this project). Fixed via a `prior_value`/`prior_age` design (most
  recent available reading + elapsed months), generalising `compute_anchor_age`'s
  already-validated pattern - ~99% real coverage. Tested 3 candidates against
  `mask_augmented_horizon_matched_split` (5-seed mean): soil-moisture prior+age
  (flat, noise-level - not worth a real submission), SPEI_12 AR(1)-deviation (net
  **loses**, 4/5 seeds worse - a `real-world-ml` ch07-style "classical model as
  feature generator" escalation that backfired, since SPEI_12's near-unit-root AR(1)
  coefficient ~0.93 left little room for the mean-reversion correction to help while
  adding two more estimated quantities' noise - not worth a real submission either),
  and SPEI_12 prior+age (~0.1% mean proxy improvement, only 3/5 seeds - initially
  judged too weak given past proxy-vs-real divergences, but the user challenged that
  call: the Gate policy's "proxy alone is enough for Tier-A features" shortcut had
  only ever been validated on a *positive* verdict (climatology), never a
  *negative*/weak one). **Real Zindi score on SPEI_12 prior+age: 0.755129 RMSE**
  (beats the prior best 0.7579 by ~0.37%) - the weak proxy signal was real.
  **Graduated**: `src/features.py::compute_prior_reading`, wired unconditionally into
  `build_all_features` (project decision, 2026-09-05). The other two candidates
  remain in the notebook only, not graduated and not real-submission-tested (their
  proxy signal gave no reason to expect a different real-world outcome).

- **`notebooks/10_hyperparameter_tuning.ipynb`**
  Why: P3 investigation - random search (Bergstra & Bengio 2012) over
  `HistGradientBoostingRegressor`'s standard boosting levers. Round 1 (25 candidates):
  only one beat the current defaults (0.7117 -> 0.7106, ~0.15%, 4/5 seeds), landing at
  the search floor for `max_leaf_nodes` - per `real-world-ml` ch04's refinement rule,
  re-ran with `learning_rate`'s range lowered. Round 2: clearer win (best: 0.7082,
  ~0.49%, 4/5 seeds), again near a boundary (`learning_rate`). Confirmed the best
  round-2 candidate with a real submission before expanding further (project's
  established practice since the P2 SPEI_12 case) - **real score 0.758156 RMSE,
  worse than the current best (0.755129)**, a genuine proxy-real inversion on the
  hyperparameter axis. Working hypothesis: `HistGradientBoostingRegressor`'s
  `early_stopping` picks its stopping point from a random (non-horizon-aware)
  internal holdout, which may not transfer to Test.csv's real distribution at low
  learning rates where the exact stopping point matters more. **Not graduated** -
  `src/model.py` unchanged.

## Comparable projects

- **DrivenData seasonal streamflow forecasting - winner write-up**
  (`docs/DrivenData - Seasonal streamflow forecasting winner writeup.pdf` - a
  different DrivenData hydrology competition, not this one)
  Why: source of the z-score/"deviation" design - `(value - group_mean) / group_std`,
  grouped by site and time-of-year - used in
  `notebooks/06_long_horizon_features.ipynb`'s `tws_climatology_deviation`. The
  winner's own static features (lat/lon/elevation/site_id as direct model inputs) are
  NOT applicable here - they would violate `config.FORBIDDEN_MODEL_FEATURES`.

- **featuretools time series guide** (https://featuretools.alteryx.com/en/v1.31.0/guides/time_series.html)
  Why: source of the `gap`/`window_length` design pattern (exclude the current/future
  observations from a rolling aggregate) used in
  `notebooks/06_long_horizon_features.ipynb`'s long-window trend feature. The library
  itself was evaluated and not adopted (project decision, 2026-09-04): it targets
  relational multi-table entity sets via Deep Feature Synthesis, and this project is a
  single flat panel table - modelling an EntitySet for two features was judged not
  worth the dependency. Implemented in plain pandas instead, consistent with the rest
  of `src/features.py`.

## Textbooks

- **Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*** (https://otexts.com/fpp3/)
  Why: standard reference for the seasonal-naive/climatology forecasting method -
  source for `tws_climatology_mean` in `notebooks/06_long_horizon_features.ipynb`
  (each cell's mean `TWS_t` for the same calendar month in all strictly earlier
  years).

- **Géron, *Hands-On Machine Learning with Scikit-Learn and TensorFlow* (O'Reilly,
  2017), ch. 7 "Ensemble Learning and Random Forests"** (`hands-on-ml` skill)
  Why: source of `notebooks/10_hyperparameter_tuning.ipynb`'s (P3) decision not to
  search `max_iter` directly - "manually grid-searching Gradient Boosting's
  `n_estimators` by training many separate ensembles from scratch [is] wasteful...
  use `staged_predict()` or a `warm_start` early-stopping loop instead" - the
  equivalent for `HistGradientBoostingRegressor` is its built-in
  `early_stopping`/`n_iter_no_change` API. Also source for treating `learning_rate`
  and the tree count as "a paired tuning surface" and gradient boosting's other
  standard levers (`max_depth`, `min_samples_leaf`, regularization).

- **Brink, Richards & Fetherolf, *Real-World Machine Learning* (Manning, 2017),
  ch. 4 "Model Evaluation and Optimization"** (`real-world-ml` skill)
  Why: source of the grid-search refinement rule applied when reading
  `notebooks/10_hyperparameter_tuning.ipynb`'s (P3) results - "boundary optimum ->
  expand the grid; high sensitivity -> densify/log-scale; low sensitivity ->
  coarsen" - and confirms boosting's standard tuning-parameter set ("number of
  trees, learning rate, max depth, splitting criterion, min samples to split").

- **Brink, Richards & Fetherolf, *Real-World Machine Learning* (Manning, 2017),
  ch. 7 "Advanced Feature Engineering"** (`real-world-ml` skill)
  Why: source of the **Classical Time-Series Feature Escalation** ladder (marginal
  stats -> windowed stats/differences -> autocorrelation/Fourier -> classical model
  fits as feature generators) used to structure
  `notebooks/09_own_cell_dynamics.ipynb`'s P2 investigation, and of the
  "point-process / time since last event" framing that justified the
  `prior_value`/`prior_age` design once a fixed calendar lag was found to have poor
  real-Test.csv coverage. Consulted only *after* the coverage problem was already
  found empirically (2026-09-05) - the project's own retrospective note on that
  ordering gap is in the README Progress log; going forward this citation step
  should happen before writing candidate-feature code, per
  `structuring-ml-projects` rule 3.

- **Bergstra & Bengio, "Random Search for Hyper-Parameter Optimization"** (JMLR 2012)
  Why: justifies random search over grid search for `notebooks/10_hyperparameter_tuning.ipynb`'s
  (P3) multi-dimensional search over `HistGradientBoostingRegressor`'s `learning_rate`/
  `max_depth`/`min_samples_leaf`/`l2_regularization`/`max_leaf_nodes` - for the same
  compute budget, random search explores each dimension's marginal effect more
  finely than a grid, which wastes evaluations on unimportant dimensions.

## Library documentation (P3)

- **scikit-learn `HistGradientBoostingRegressor`** (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
  Why: source of the `early_stopping`/`validation_fraction`/`n_iter_no_change` API used
  in `notebooks/10_hyperparameter_tuning.ipynb` to let the model pick its own `max_iter`
  per `hands-on-ml` ch07's guidance below, instead of searching it directly.
