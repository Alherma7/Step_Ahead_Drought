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
- **Our first real submission scored 0.7822 RMSE** (2026-09-03) — see Progress log
  and `notebooks/05_leaderboard_gap_investigation.ipynb` for why this came in much
  worse than the internal validation number we had at the time (0.5994), and what
  the corrected internal proxy metric now is (0.6532, via
  `evaluate.horizon_matched_split`).

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
   derived feature needs a per-row source-date ≤ t audit before use.

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
- [ ] Feature engineering: own-cell TWS/SPEI history (lags, rolling stats,
      climatology) — validate each in notebook before graduating to
      `src/features.py`.
- [x] Feature engineering: spatial neighbourhood aggregates (GIS component).
      **Won** (`notebooks/04_spatial_neighbour_features.ipynb`): 1.3% RMSE
      improvement. Graduated to `src/features.py`, wired into `src/train.py`.
- [x] **Submit `outputs/submission.csv` to Zindi** — scored 0.7822 RMSE vs.
      leader's 0.559. Root-caused the gap from our 0.5994 internal estimate (see
      Progress log) — the internal proxy metric is now 0.6532 via
      `evaluate.horizon_matched_split`. **Use this metric, not the old pooled
      one, to gate every feature from here on.**
- [ ] Feature engineering for long forecast horizons specifically (10-40 months,
      78% of Test.csv): seasonal climatology, longer-window trends per cell —
      persistence/neighbour-style signals decay fastest with horizon, so this is
      the natural next lever now that the metric honestly reflects deployment.
- [ ] External data phase: Copernicus/ERA5 ingestion module + per-feature
      source-date audit, once the provided-columns pipeline is validated.
- [ ] Hyperparameter tuning (random search) — after feature work, not before.
