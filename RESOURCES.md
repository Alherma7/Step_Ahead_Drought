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

## Pending (add before use)

- Own-cell historical/lag features (TWS/SPEI trends, climatology): not yet sourced or
  attempted.
- Random search vs. grid search for hyperparameter tuning: Bergstra & Bengio, JMLR 2012
  ("Random Search for Hyper-Parameter Optimization") - cite in full once tuning starts.
