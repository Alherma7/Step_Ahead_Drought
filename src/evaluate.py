"""This project's evaluation metric (RMSE, per docs/overview.txt "Evaluation") and a
chronological train/validation split for the time-series-shaped TWS forecasting task.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config, features


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error - the exact leaderboard metric (docs/overview.txt)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_metrics(y_true, y_pred) -> dict:
    """RMSE (the leaderboard metric) plus MAE/R2 as diagnostics."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def time_train_val_split(
    df: pd.DataFrame, val_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically: the last val_fraction of calendar months become the
    validation set, all earlier months are the fit set.

    This is a forward-looking forecasting task (predict t+1 from t), so a random
    row-wise split would leak future information into training via near-duplicate
    rows at nearby times/locations. Chronological splitting is the standard fix for
    time-series evaluation (see scikit-learn's TimeSeriesSplit docs, cited in
    RESOURCES.md) and matches the split already used in
    notebooks/00_official_starter_reference.ipynb.
    """
    unique_times = np.sort(df[config.TIME_COL].unique())
    split_idx = int(len(unique_times) * (1 - val_fraction))
    fit_times = unique_times[:split_idx]
    val_times = unique_times[split_idx:]

    fit_df = df[df[config.TIME_COL].isin(fit_times)].copy().reset_index(drop=True)
    val_df = df[df[config.TIME_COL].isin(val_times)].copy().reset_index(drop=True)
    return fit_df, val_df


def compute_horizons(reference_time: pd.Timestamp, times) -> pd.Series:
    """Whole months from reference_time to each timestamp in times (can be negative)."""
    times = pd.to_datetime(pd.Series(times))
    return (times.dt.year - reference_time.year) * 12 + (times.dt.month - reference_time.month)


def compute_test_horizons(train: pd.DataFrame, test: pd.DataFrame) -> set[int]:
    """The exact forecast-horizon distribution (months from train's cutoff to each
    unique test month) that the deployed model actually faces at submission time.
    """
    cutoff = train[config.TIME_COL].max()
    return set(compute_horizons(cutoff, test[config.TIME_COL].unique()).tolist())


def horizon_matched_split(
    train: pd.DataFrame, target_horizons: set[int], val_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological fit/val split (as time_train_val_split), then restrict val_df to
    only the months whose horizon-from-fit-cutoff is in target_horizons - so the
    validation set's horizon mix matches a real deployment scenario (e.g. Test.csv's
    actual horizon-from-train-cutoff distribution, from compute_test_horizons)
    instead of pooling every horizon roughly evenly.

    Source: notebooks/05_leaderboard_gap_investigation.ipynb - the plain, unweighted
    time_train_val_split's pooled RMSE (0.5994) was a poor predictor of the real
    leaderboard score (0.7822): validation RMSE nearly doubled in variance terms for
    the longest-horizon 25% of rows (0.494 -> 0.732), and Test.csv is
    disproportionately long-horizon (77.7% of rows >=10 months from the training
    cutoff) relative to a plain chronological split, which pools horizons much more
    evenly.
    """
    fit_df, val_df = time_train_val_split(train, val_fraction)
    cutoff = fit_df[config.TIME_COL].max()
    horizons = compute_horizons(cutoff, val_df[config.TIME_COL])
    matched = val_df[horizons.isin(target_horizons).to_numpy()].reset_index(drop=True)
    return fit_df, matched


def measure_masking_pattern(test_df: pd.DataFrame) -> tuple[float, float]:
    """Measure Test.csv's real TWS_t masking pattern as (masked_month_fraction,
    masked_row_fraction): the share of months that are (near-)fully masked, and the
    average masked-row share within just those months.

    Source: notebooks/01_eda.ipynb - Test.csv's masking is bimodal, not a uniform
    66.5% per month: some months are ~0% masked, others >99.5%. A month counts as
    "masked" here if more than half its rows are masked - comfortably separates the
    two modes without hardcoding an exact threshold like 99.5%.
    """
    per_month = test_df.groupby(config.TIME_COL)[config.TWS_MASKED_COL].mean()
    masked_months = per_month[per_month > 0.5]
    masked_month_fraction = len(masked_months) / len(per_month)
    masked_row_fraction = float(masked_months.mean()) if len(masked_months) else 0.0
    return masked_month_fraction, masked_row_fraction


def simulate_masking(
    df: pd.DataFrame,
    masked_month_fraction: float,
    masked_row_fraction: float,
    seed: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Simulate Test.csv's bimodal TWS_t masking pattern on any dataframe with known
    true values (e.g. a validation split), so a validation metric can be computed
    under the same masked-input regime the model faces at real inference time
    instead of on fully-observed data.

    Source: notebooks/03_masked_tws_fill.ipynb, generalised with parameters from
    measure_masking_pattern instead of hardcoded literals. Randomly designates
    masked_month_fraction of df's unique months as "masked months", then masks
    masked_row_fraction of their rows' TWS_t (set to NaN). Deterministic given seed.
    """
    rng = np.random.RandomState(seed)
    out = df.copy()
    months = np.sort(out[config.TIME_COL].unique())
    n_masked_months = round(len(months) * masked_month_fraction)
    masked_months = (
        set(rng.choice(months, size=n_masked_months, replace=False))
        if n_masked_months else set()
    )

    is_masked_month = out[config.TIME_COL].isin(masked_months)
    row_mask = (is_masked_month & (rng.random_sample(len(out)) < masked_row_fraction)).to_numpy()
    out.loc[row_mask, config.TWS_COL] = np.nan
    return out


def mask_aware_horizon_matched_split(
    raw_train: pd.DataFrame,
    target_horizons: set[int],
    masked_month_fraction: float,
    masked_row_fraction: float,
    val_fraction: float = 0.2,
    seed: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """horizon_matched_split, but the validation half additionally gets Test.csv's
    real TWS_t masking pattern simulated onto it (simulate_masking) before the
    derived-feature pipeline (features.build_all_features) runs - so validation
    reflects the masked-input regime the model actually faces at ~66.5% of real
    Test.csv rows, not fully-observed data like the plain horizon_matched_split.

    Source: project decision, 2026-09-04, after seasonal-climatology/trend
    graduation - horizon_matched_split alone still validated every feature on
    fully-observed TWS_t, the opposite of Test.csv's dominant regime, which is
    suspected of inverting the climatology+trend interaction ranking on the real
    leaderboard (see notebooks/07_mask_aware_validation.ipynb for the fix's own
    validation).

    `raw_train` must be RAW - not already run through build_all_features - since
    masking must happen before derived features are computed from TWS_t, exactly
    mirroring Test.csv's own pipeline order (masked in the raw file, features
    computed after). Pass measure_masking_pattern(test)'s output as
    masked_month_fraction/masked_row_fraction to match Test.csv's real pattern.
    """
    fit_raw, val_raw = time_train_val_split(raw_train, val_fraction)
    cutoff = fit_raw[config.TIME_COL].max()
    horizons = compute_horizons(cutoff, val_raw[config.TIME_COL])
    val_matched_raw = val_raw[horizons.isin(target_horizons).to_numpy()].reset_index(drop=True)

    val_masked_raw = simulate_masking(
        val_matched_raw, masked_month_fraction, masked_row_fraction, seed
    )
    return features.build_all_features(fit_raw, val_masked_raw)


def augment_with_simulated_masking(
    raw_df: pd.DataFrame,
    masked_month_fraction: float,
    masked_row_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Simulate Test.csv's masking pattern on raw_df's own TWS_t, backward-fill it
    from raw_df's own earlier history (self-referential), and compute each row's
    resulting anchor age - a training-time augmentation so a model can learn from
    examples with a stale/frozen anchor, not just fully-observed rows (all that
    Train.csv otherwise ever provides).

    Source: notebooks/08_anchor_age_feature.ipynb (P1, Opus-model review,
    2026-09-04) - validated over 5 independent masking realisations, beating an
    unaugmented fit on every metric component in every seed.

    Returns (augmented_df, anchor_age): augmented_df is a copy of raw_df with
    TWS_t replaced by its masked-then-backward-filled version; anchor_age is a
    Series aligned to augmented_df giving each row's months-since-anchor (0 for
    rows that were never masked, NaN for a cell with no earlier anchor at all).
    """
    masked = simulate_masking(raw_df, masked_month_fraction, masked_row_fraction, seed)
    anchor_age = features.compute_anchor_age(masked.iloc[:0], masked)
    augmented = raw_df.copy()
    augmented[config.TWS_COL] = features.backward_fill_tws(masked).to_numpy()
    return augmented, anchor_age


def mask_augmented_horizon_matched_split(
    raw_train: pd.DataFrame,
    target_horizons: set[int],
    masked_month_fraction: float,
    masked_row_fraction: float,
    val_fraction: float = 0.2,
    fit_seed: int = config.RANDOM_STATE,
    val_seed: int = config.RANDOM_STATE + 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """mask_aware_horizon_matched_split, but the fit half is ALSO augmented with
    simulated masking (self-referentially filled from its own earlier history),
    and both halves get a `months_since_anchor` column (features.compute_anchor_age).

    Source: project decision 2026-09-04 (P1, Opus-model review) - a
    `months_since_anchor` feature is only learnable if training data shows it
    varying; Train.csv is otherwise always fully observed, so months_since_anchor
    would be constant 0 for every fit row without this augmentation. Validated
    (5-seed robustness check, `notebooks/08_anchor_age_feature.ipynb`): both the
    fit-augmentation itself and the anchor-age feature on top each improve every
    metric component, in every seed tried, over the plain mask_aware_horizon_
    matched_split.

    Fit's own masking simulation is filled from fit's OWN earlier history
    (self-referential, via backward_fill_tws directly) - it must not see val's
    true values. Val's masking (as in mask_aware_horizon_matched_split) is filled
    from fit_raw's TRUE, unaugmented history - exactly mirroring how real
    Test.csv is filled from the always-fully-observed Train.csv, not from an
    augmented copy of it.
    """
    fit_raw, val_raw = time_train_val_split(raw_train, val_fraction)
    cutoff = fit_raw[config.TIME_COL].max()
    horizons = compute_horizons(cutoff, val_raw[config.TIME_COL])
    val_matched_raw = val_raw[horizons.isin(target_horizons).to_numpy()].reset_index(drop=True)

    fit_filled, fit_anchor_age = augment_with_simulated_masking(
        fit_raw, masked_month_fraction, masked_row_fraction, fit_seed
    )

    val_masked_raw = simulate_masking(val_matched_raw, masked_month_fraction, masked_row_fraction, val_seed)
    val_anchor_age = features.compute_anchor_age(fit_raw, val_masked_raw)

    fit_out, val_out = features.build_all_features(fit_filled, val_masked_raw)
    fit_out["months_since_anchor"] = fit_anchor_age.to_numpy()
    val_out["months_since_anchor"] = val_anchor_age.to_numpy()
    return fit_out, val_out
