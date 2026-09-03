import numpy as np
import pandas as pd
import pytest

from src.features import (
    add_neighbourhood_features,
    backward_fill_tws,
    build_spatial_adjacency,
    fill_masked_tws,
    select_base_features,
)


def test_select_base_features_returns_float32_array():
    df = pd.DataFrame({"TWS_t": [1.0, 2.0], "month_sin": [0.5, -0.5]})
    X = select_base_features(df, ["TWS_t", "month_sin"])
    assert X.dtype == np.float32
    assert X.shape == (2, 2)


def test_select_base_features_rejects_forbidden_columns():
    df = pd.DataFrame({"lat": [1.0], "TWS_t": [1.0]})
    with pytest.raises(ValueError, match="Forbidden"):
        select_base_features(df, ["lat", "TWS_t"])


def _series(times, tws):
    return pd.DataFrame({
        "lat": [1.0] * len(times),
        "lon": [1.0] * len(times),
        "time": pd.to_datetime(times),
        "TWS_t": tws,
    })


def test_backward_fill_tws_fills_from_the_last_earlier_observed_value():
    df = _series(
        ["2020-01-01", "2020-02-01", "2020-03-01"],
        [1.0, np.nan, np.nan],
    )
    filled = backward_fill_tws(df)
    assert filled.tolist() == [1.0, 1.0, 1.0]


def test_backward_fill_tws_never_uses_a_later_value():
    # masked row has no earlier observation, only a later one - must stay NaN,
    # not be filled from the future (that would be the confirmed leak).
    df = _series(["2020-01-01", "2020-02-01"], [np.nan, 5.0])
    filled = backward_fill_tws(df)
    assert np.isnan(filled.iloc[0])
    assert filled.iloc[1] == 5.0


def test_backward_fill_tws_leaves_observed_values_untouched():
    df = _series(["2020-01-01", "2020-02-01"], [1.0, 2.0])
    filled = backward_fill_tws(df)
    assert filled.tolist() == [1.0, 2.0]


def test_backward_fill_tws_is_per_cell():
    df = pd.DataFrame({
        "lat": [1.0, 1.0, 2.0],
        "lon": [1.0, 1.0, 2.0],
        "time": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-02-01"]),
        "TWS_t": [3.0, np.nan, np.nan],
    })
    filled = backward_fill_tws(df)
    assert filled.iloc[1] == 3.0  # same cell as the earlier observed row
    assert np.isnan(filled.iloc[2])  # different cell, no history at all


def test_fill_masked_tws_uses_history_df_for_a_masked_first_row():
    history = _series(["2020-01-01"], [7.0])
    target = _series(["2020-02-01"], [np.nan])

    out = fill_masked_tws(history, target)

    assert out["TWS_t"].tolist() == [7.0]
    assert len(out) == len(target)


def _cell_index(rows):
    # rows: list of (lat, lon)
    return pd.DataFrame(rows, columns=["lat", "lon"])


def test_build_spatial_adjacency_connects_nearby_cells_only():
    # A-B ~111km apart (1 degree of latitude), C ~2200km from A - well outside 500km.
    cells = _cell_index([(0.0, 0.0), (1.0, 0.0), (20.0, 0.0)])
    cell_id, adjacency, n_cells = build_spatial_adjacency(cells, radius_km=500.0)

    assert n_cells == 3
    a, b, c = cell_id[(0.0, 0.0)], cell_id[(1.0, 0.0)], cell_id[(20.0, 0.0)]
    assert adjacency[a, b] == 1
    assert adjacency[b, a] == 1
    assert adjacency[a, c] == 0
    assert adjacency[a, a] == 0  # never a self-loop


def test_add_neighbourhood_features_averages_only_valid_neighbours():
    # A has two neighbours, B (observed) and D (masked) - D must be excluded.
    cells = _cell_index([(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (30.0, 0.0)])
    cell_id, adjacency, n_cells = build_spatial_adjacency(cells, radius_km=500.0)

    df = pd.DataFrame({
        "lat": [0.0, 1.0, -1.0, 30.0],
        "lon": [0.0, 0.0, 0.0, 0.0],
        "time": pd.to_datetime(["2020-01-01"] * 4),
        "TWS_t": [5.0, 2.0, np.nan, 100.0],  # A=5.0, B=2.0, D=NaN (masked), far cell=100.0
    })

    out = add_neighbourhood_features(df, cell_id, adjacency, n_cells)
    a_row = out[(out["lat"] == 0.0) & (out["lon"] == 0.0)].iloc[0]

    assert a_row["tws_neighbour_mean"] == 2.0  # only B counted, D excluded
    assert a_row["tws_local_deviation"] == 5.0 - 2.0


def test_add_neighbourhood_features_is_nan_for_an_isolated_cell():
    cells = _cell_index([(0.0, 0.0), (40.0, 0.0)])  # far apart, no neighbours
    cell_id, adjacency, n_cells = build_spatial_adjacency(cells, radius_km=500.0)

    df = pd.DataFrame({
        "lat": [0.0, 40.0],
        "lon": [0.0, 0.0],
        "time": pd.to_datetime(["2020-01-01"] * 2),
        "TWS_t": [1.0, 2.0],
    })

    out = add_neighbourhood_features(df, cell_id, adjacency, n_cells)
    assert out["tws_neighbour_mean"].isna().all()
    assert out["tws_local_deviation"].isna().all()
