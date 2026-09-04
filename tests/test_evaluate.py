import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    augment_with_simulated_masking,
    compute_horizons,
    compute_metrics,
    compute_test_horizons,
    horizon_matched_split,
    mask_augmented_horizon_matched_split,
    mask_aware_horizon_matched_split,
    measure_masking_pattern,
    rmse,
    simulate_masking,
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


def test_measure_masking_pattern_finds_the_bimodal_split():
    # 3 months: one 0% masked, two ~100% masked - matches Test.csv's real bimodal
    # pattern (some months fully observed, others near-fully masked).
    times = pd.to_datetime(["2020-01-01"] * 2 + ["2020-02-01"] * 2 + ["2020-03-01"] * 2)
    test = pd.DataFrame({
        "time": times,
        "TWS_t_masked": [False, False, True, True, True, True],
    })

    month_fraction, row_fraction = measure_masking_pattern(test)

    assert month_fraction == pytest.approx(2 / 3)
    assert row_fraction == pytest.approx(1.0)


def test_simulate_masking_masks_only_designated_months():
    times = pd.to_datetime(["2020-01-01"] * 2 + ["2020-02-01"] * 2)
    df = pd.DataFrame({
        "lat": [1.0] * 4, "lon": [1.0] * 4,
        "time": times,
        "TWS_t": [1.0, 2.0, 3.0, 4.0],
    })

    out = simulate_masking(df, masked_month_fraction=0.5, masked_row_fraction=1.0, seed=0)

    # Exactly one of the two months is fully masked, the other untouched.
    per_month_masked = out.groupby("time")["TWS_t"].apply(lambda s: s.isna().all())
    assert per_month_masked.sum() == 1
    untouched_month = per_month_masked[~per_month_masked].index[0]
    assert out.loc[out["time"] == untouched_month, "TWS_t"].notna().all()


def test_simulate_masking_is_a_no_op_with_zero_masked_fraction():
    df = pd.DataFrame({
        "lat": [1.0, 1.0], "lon": [1.0, 1.0],
        "time": pd.to_datetime(["2020-01-01", "2020-02-01"]),
        "TWS_t": [1.0, 2.0],
    })

    out = simulate_masking(df, masked_month_fraction=0.0, masked_row_fraction=1.0, seed=0)

    assert out["TWS_t"].tolist() == [1.0, 2.0]


def test_mask_aware_horizon_matched_split_produces_engineered_features_with_some_masking():
    # Two cells, enough monthly history for climatology/neighbourhood to be
    # well-defined, and a val split whose only month gets masked.
    months = pd.date_range("2020-01-01", periods=8, freq="MS")
    rows = []
    for lat in [0.0, 1.0]:
        for i, t in enumerate(months):
            rows.append({"lat": lat, "lon": 0.0, "time": t, "TWS_t": float(i + (0 if lat == 0.0 else 100))})
    raw = pd.DataFrame(rows)

    fit_df, val_df = mask_aware_horizon_matched_split(
        raw, target_horizons={1}, masked_month_fraction=1.0, masked_row_fraction=1.0,
        val_fraction=1 / 8, seed=0,
    )

    assert "tws_neighbour_mean" in val_df.columns
    assert "tws_climatology_mean" in val_df.columns
    # The val month's TWS_t was masked and then backward-filled - it must equal the
    # cell's own last fit-period value, not the true (higher) value at that month.
    val_row_lat0 = val_df[val_df["lat"] == 0.0].iloc[0]
    fit_last_lat0 = fit_df[fit_df["lat"] == 0.0].sort_values("time")["TWS_t"].iloc[-1]
    assert val_row_lat0["TWS_t"] == fit_last_lat0


def test_mask_augmented_horizon_matched_split_augments_both_fit_and_val():
    months = pd.date_range("2020-01-01", periods=10, freq="MS")
    rows = []
    for lat in [0.0, 1.0]:
        for i, t in enumerate(months):
            rows.append({"lat": lat, "lon": 0.0, "time": t,
                         "TWS_t": float(i + (0 if lat == 0.0 else 100))})
    raw = pd.DataFrame(rows)

    fit_df, val_df = mask_augmented_horizon_matched_split(
        raw, target_horizons={1}, masked_month_fraction=0.5, masked_row_fraction=1.0,
        val_fraction=1 / 10, fit_seed=1, val_seed=0,
    )

    assert "months_since_anchor" in fit_df.columns
    assert "months_since_anchor" in val_df.columns
    # Fit was augmented with simulated masking, so its own anchor age must vary,
    # not stay constant at 0 (which would make the feature unlearnable).
    assert (fit_df["months_since_anchor"] > 0).any()
    assert fit_df["months_since_anchor"].notna().any()


def test_augment_with_simulated_masking_returns_matching_shapes():
    df = pd.DataFrame({
        "lat": [1.0] * 6, "lon": [1.0] * 6,
        "time": pd.date_range("2020-01-01", periods=6, freq="MS"),
        "TWS_t": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })

    augmented, age = augment_with_simulated_masking(df, 0.5, 1.0, seed=0)

    assert len(augmented) == len(df) == len(age)
    assert list(augmented.columns) == list(df.columns)


def test_augment_with_simulated_masking_is_a_no_op_with_zero_masked_fraction():
    df = pd.DataFrame({
        "lat": [1.0, 1.0], "lon": [1.0, 1.0],
        "time": pd.to_datetime(["2020-01-01", "2020-02-01"]),
        "TWS_t": [1.0, 2.0],
    })

    augmented, age = augment_with_simulated_masking(df, 0.0, 1.0, seed=0)

    assert augmented["TWS_t"].tolist() == [1.0, 2.0]
    assert age.tolist() == [0, 0]


def test_augment_with_simulated_masking_ages_match_actual_masked_rows():
    df = pd.DataFrame({
        "lat": [1.0] * 6, "lon": [1.0] * 6,
        "time": pd.date_range("2020-01-01", periods=6, freq="MS"),
        "TWS_t": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })

    augmented, age = augment_with_simulated_masking(df, 0.5, 1.0, seed=0)

    was_masked = ~np.isclose(augmented["TWS_t"].to_numpy(), df["TWS_t"].to_numpy())
    assert was_masked.any()  # something was actually masked given these params
    for i in range(6):
        if was_masked[i]:
            if pd.isna(augmented["TWS_t"].iloc[i]):
                assert pd.isna(age.iloc[i])  # no earlier anchor at all
            else:
                assert age.iloc[i] > 0
                assert augmented["TWS_t"].iloc[i] in df["TWS_t"].iloc[:i].to_numpy()
        else:
            assert age.iloc[i] == 0
