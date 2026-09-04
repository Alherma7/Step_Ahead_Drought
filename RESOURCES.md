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
  MAE trade-offs in any seed, unlike the trend feature. Not yet wired into
  `src/train.py`'s default pipeline pending real-leaderboard confirmation
  (`outputs/submission_anchor_age.csv` generated and queued).

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

## Pending (add before use)

- Random search vs. grid search for hyperparameter tuning: Bergstra & Bengio, JMLR 2012
  ("Random Search for Hyper-Parameter Optimization") - cite in full once tuning starts.
