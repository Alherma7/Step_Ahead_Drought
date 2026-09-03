import numpy as np
import pandas as pd

from src.evaluate import (
    compute_horizons,
    compute_metrics,
    compute_test_horizons,
    horizon_matched_split,
    rmse,
    time_train_val_split,
)


def test_rmse_zero_for_perfect_predictions():
    assert rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_rmse_known_value():
    # errors of [1, -1] -> RMSE = sqrt(mean([1, 1])) = 1.0
    assert rmse([0.0, 0.0], [1.0, -1.0]) == 1.0


def test_compute_metrics_returns_rmse_mae_r2_keys():
    result = compute_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    assert set(result.keys()) == {"rmse", "mae", "r2"}


def test_time_train_val_split_is_chronological_and_non_overlapping():
    times = pd.to_datetime(
        ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01"]
    )
    df = pd.DataFrame({"time": np.repeat(times, 2)})

    fit_df, val_df = time_train_val_split(df, val_fraction=0.2)

    assert fit_df["time"].max() < val_df["time"].min()
    assert len(fit_df) + len(val_df) == len(df)


def test_time_train_val_split_respects_val_fraction():
    times = pd.to_datetime([f"2020-{m:02d}-01" for m in range(1, 11)])
    df = pd.DataFrame({"time": times})

    fit_df, val_df = time_train_val_split(df, val_fraction=0.3)

    assert len(val_df) == 3
    assert len(fit_df) == 7


def test_compute_horizons_counts_whole_months_including_negative():
    reference = pd.Timestamp("2020-06-01")
    times = pd.to_datetime(["2020-06-01", "2020-07-01", "2021-01-01", "2020-05-01"])

    horizons = compute_horizons(reference, times)

    assert horizons.tolist() == [0, 1, 7, -1]


def test_compute_test_horizons_matches_gap_between_train_and_test():
    train = pd.DataFrame({"time": pd.to_datetime(["2020-01-01", "2020-02-01"])})
    test = pd.DataFrame({"time": pd.to_datetime(["2020-03-01", "2020-05-01", "2020-05-01"])})

    horizons = compute_test_horizons(train, test)

    assert horizons == {1, 3}  # unique test months only, duplicates collapse


def test_horizon_matched_split_keeps_only_matching_horizon_months():
    times = pd.to_datetime([f"2020-{m:02d}-01" for m in range(1, 11)])
    df = pd.DataFrame({"time": times})
    # val_fraction=0.3 -> fit ends at month 7, val = months 8, 9, 10 (horizons 1, 2, 3)

    fit_df, matched = horizon_matched_split(df, target_horizons={1, 3}, val_fraction=0.3)

    matched_months = sorted(matched["time"].dt.month.tolist())
    assert matched_months == [8, 10]  # horizon 2 (month 9) dropped
    assert fit_df["time"].max() < matched["time"].min()
