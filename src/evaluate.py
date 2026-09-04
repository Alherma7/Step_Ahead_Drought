"""This project's evaluation metric (RMSE, per docs/overview.txt "Evaluation") and a
chronological train/validation split for the time-series-shaped TWS forecasting task.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config


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
