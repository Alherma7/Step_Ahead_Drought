"""Feature engineering.

Per this project's workflow (see README.md), a feature only belongs here once
it has been built and validated against src/evaluate.py's RMSE gate in a
notebook (see structuring-ml-projects skill's "graduation rule"). Candidate
features still under investigation - own-cell TWS/SPEI lag history, spatial
neighbourhood aggregates (permitted per docs/chats/Neighbouring cells' past
TWS - permitted or not.txt) - live in notebooks/ until they win.

select_base_features() is not itself an engineered feature - it is the column
selection already used, unmodified, by the official starter notebook
(notebooks/00_official_starter_reference.ipynb), so it graduated directly.
backward_fill_tws()/fill_masked_tws() graduated from
notebooks/03_masked_tws_fill.ipynb after beating the median-imputer
placeholder on every metric component (see that notebook's Gate decision).
build_spatial_adjacency()/add_neighbourhood_features() graduated from
notebooks/04_spatial_neighbour_features.ipynb after a 1.3% RMSE improvement
with no regression on MAE/R2.
add_seasonal_climatology_features() graduated from
notebooks/06_long_horizon_features.ipynb after a real Zindi leaderboard
improvement (0.7822 -> 0.7778 RMSE), despite a regression on the internal MAE
diagnostic - see that notebook's Gate decision and RESOURCES.md for why the
real leaderboard score, not the internal proxy, drove this graduation. The
long-window trend feature explored in the same notebook was NOT graduated -
confirmed negative result, worse standalone and worse combined with
climatology on the real leaderboard (0.7834 and 0.7845 respectively).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

from src import config

_R_EARTH_KM = 6371.0


def select_base_features(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Select model-input columns as a float32 array.

    Source: notebooks/00_official_starter_reference.ipynb (build_xy). Raises if any
    forbidden column (lat/lon/ID - see config.FORBIDDEN_MODEL_FEATURES) is requested,
    since the challenge rules disallow coordinates as predictors.
    """
    forbidden = config.FORBIDDEN_MODEL_FEATURES & set(feature_cols)
    if forbidden:
        raise ValueError(f"Forbidden columns requested as model features: {forbidden}")
    return df[feature_cols].to_numpy(dtype=np.float32)


def backward_fill_tws(df: pd.DataFrame) -> pd.Series:
    """Fill masked/missing TWS_t using each (lat, lon) cell's last observed value
    at or before that row's own time - never a later one.

    Source: docs/chats/Clarification does the masked-row fill count as t+1
    information.txt (Zindi ruling) - averaging over all observed months
    (including later ones) is a confirmed leak; only a strictly-backward
    per-cell anchor is compliant. `df` must contain every earlier observation for
    the cells being filled (e.g. Train.csv concatenated with Test.csv) - this
    function sorts by (lat, lon, time) internally and returns a Series aligned to
    df's original row order.

    Validated in notebooks/03_masked_tws_fill.ipynb: on a simulated test-like
    masking pattern applied to the validation split, this cut RMSE from 0.7791 to
    0.6919 overall (0.8659 -> 0.7481 on masked rows specifically), with zero
    change on already-observed rows.
    """
    ordered = df.sort_values([config.LAT_COL, config.LON_COL, config.TIME_COL])
    filled = ordered.groupby([config.LAT_COL, config.LON_COL])[config.TWS_COL].ffill()
    return filled.reindex(df.index)


def fill_masked_tws(history_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of target_df with masked/missing TWS_t rows filled via
    backward_fill_tws, using history_df (e.g. Train.csv, always fully observed)
    plus target_df's own observed rows as that per-cell chronological series.
    """
    cols = [config.LAT_COL, config.LON_COL, config.TIME_COL, config.TWS_COL]
    combined = pd.concat([history_df[cols], target_df[cols]], ignore_index=True)
    filled = backward_fill_tws(combined)
    out = target_df.copy()
    out[config.TWS_COL] = filled.iloc[len(history_df):].to_numpy()
    return out


def _latlon_to_unit_sphere(lat_deg, lon_deg) -> np.ndarray:
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.column_stack([x, y, z])


def build_spatial_adjacency(cell_index: pd.DataFrame, radius_km: float = config.NEIGHBOUR_RADIUS_KM):
    """Build a (lat, lon) neighbour graph: cells within radius_km of each other
    (excluding self), via great-circle distance on a unit sphere - exact at
    every latitude, unlike a naive lat/lon Euclidean distance.

    Source: docs/chats/Neighbouring cells' past TWS - permitted or not.txt
    (Zindi ruling) - neighbourhood aggregation of TWS at months <= t is
    explicitly permitted, though lat/lon may never be a direct model input;
    this graph only ever feeds add_neighbourhood_features(), never the model
    directly.

    cell_index must have one row per unique (lat, lon) cell. Returns
    (cell_id, adjacency, n_cells): cell_id maps (lat, lon) -> row/col index into
    the sparse (n_cells, n_cells) 0/1 adjacency matrix.
    """
    cell_id = {
        (lat, lon): i
        for i, (lat, lon) in enumerate(zip(cell_index[config.LAT_COL], cell_index[config.LON_COL]))
    }
    n_cells = len(cell_index)

    xyz = _latlon_to_unit_sphere(
        cell_index[config.LAT_COL].to_numpy(), cell_index[config.LON_COL].to_numpy()
    )
    chord_threshold = 2 * np.sin(radius_km / (2 * _R_EARTH_KM))

    tree = cKDTree(xyz)
    neighbour_lists = tree.query_ball_point(xyz, r=chord_threshold)

    rows, cols = [], []
    for i, neighbours in enumerate(neighbour_lists):
        for j in neighbours:
            if j != i:
                rows.append(i)
                cols.append(j)

    adjacency = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n_cells, n_cells)
    )
    return cell_id, adjacency, n_cells


def add_neighbourhood_features(
    df: pd.DataFrame, cell_id: dict, adjacency: sp.csr_matrix, n_cells: int
) -> pd.DataFrame:
    """Add tws_neighbour_mean (same-month spatial average of TWS_t over cells
    within build_spatial_adjacency's radius, excluding self) and
    tws_local_deviation (TWS_t - tws_neighbour_mean).

    Source: same Zindi ruling as build_spatial_adjacency. NaN/masked TWS_t
    values are excluded from each neighbourhood average rather than treated as
    zero. Validated in notebooks/04_spatial_neighbour_features.ipynb: 1.3% RMSE
    improvement (0.6073 -> 0.5994) with no regression on MAE/R2.
    """
    out = df.reset_index(drop=True).copy()
    out["_cell_idx"] = [
        cell_id[(lat, lon)] for lat, lon in zip(out[config.LAT_COL], out[config.LON_COL])
    ]

    neighbour_mean = np.full(len(out), np.nan)
    for _, grp in out.groupby(config.TIME_COL):
        idx = grp["_cell_idx"].to_numpy()
        values = grp[config.TWS_COL].to_numpy()
        valid = ~np.isnan(values)

        month_vec = np.zeros(n_cells)
        month_vec[idx[valid]] = values[valid]
        present = np.zeros(n_cells)
        present[idx[valid]] = 1.0

        weighted_sum = adjacency @ month_vec
        weight = adjacency @ present
        with np.errstate(invalid="ignore", divide="ignore"):
            cell_mean = np.where(weight > 0, weighted_sum / weight, np.nan)

        neighbour_mean[grp.index.to_numpy()] = cell_mean[idx]

    out[config.NEIGHBOUR_MEAN_COL] = neighbour_mean
    out[config.LOCAL_DEVIATION_COL] = out[config.TWS_COL] - out[config.NEIGHBOUR_MEAN_COL]
    return out.drop(columns=["_cell_idx"])


def add_seasonal_climatology_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add each row's per-cell seasonal climatology: the mean/std of that cell's own
    TWS_t in the same calendar month across all strictly earlier years, plus a
    standardised deviation from it.

    A calendar month occurs once per year, so "all earlier years' same month" is
    automatically < t - no explicit gap parameter needed, unlike a same-resolution
    rolling window. Cells with no earlier occurrence of that month (typically a
    cell's first year in the dataset) get NaN, left for the pipeline's median
    imputer. To include a test row's own earlier test-period history (not just
    Train.csv), call this on a concatenated frame and slice the result - the same
    pattern backward_fill_tws/fill_masked_tws use.

    Source (climatology mean): Hyndman & Athanasopoulos, "Forecasting: Principles
    and Practice" (seasonal-naive/climatology method). Source (standardised
    deviation design): a DrivenData seasonal-streamflow-forecasting competition
    winner write-up (docs/DrivenData - Seasonal streamflow forecasting winner
    writeup.pdf) - z-scoring a physical measurement by location and time-of-year.

    Validated in notebooks/06_long_horizon_features.ipynb: real Zindi leaderboard
    RMSE improved 0.7822 -> 0.7778 despite a regression on the internal MAE
    diagnostic - graduated on the real leaderboard result, not the internal proxy
    (see that notebook's Gate decision).
    """
    out = df.copy()
    ordered = out.sort_values([config.LAT_COL, config.LON_COL, config.TIME_COL])
    month = ordered[config.TIME_COL].dt.month
    grp = ordered.groupby([ordered[config.LAT_COL], ordered[config.LON_COL], month])[config.TWS_COL]
    clim_mean = grp.transform(lambda s: s.expanding().mean().shift(1))
    clim_std = grp.transform(lambda s: s.expanding().std().shift(1))

    out[config.CLIMATOLOGY_MEAN_COL] = clim_mean.reindex(out.index)
    out[config.CLIMATOLOGY_STD_COL] = clim_std.reindex(out.index)
    with np.errstate(invalid="ignore", divide="ignore"):
        deviation = (
            (out[config.TWS_COL] - out[config.CLIMATOLOGY_MEAN_COL])
            / out[config.CLIMATOLOGY_STD_COL]
        )
    out[config.CLIMATOLOGY_DEVIATION_COL] = deviation.replace([np.inf, -np.inf], np.nan)
    return out


def compute_anchor_age(history_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.Series:
    """For each target_df row, whole months since that cell's last observed
    (non-null) TWS_t at or before this row's own time - 0 for an already-observed
    row, NaN if the cell has no observed history at all yet.

    Source: notebooks/05_leaderboard_gap_investigation.ipynb's "anchor staleness"
    diagnostic (average 2.6, max 6 months on real Test.csv) - computed there only
    to rule out staleness as the leaderboard-gap cause, never used as a model
    input. Promoted to an actual feature per the Opus-model review's P1
    recommendation, 2026-09-04: the model previously had no way to distinguish a
    1-month-ahead forecast from a 4-month-ahead one off the same frozen
    backward-filled anchor. Returns a Series aligned to target_df's row order
    (same calling convention as fill_masked_tws).
    """
    cols = [config.LAT_COL, config.LON_COL, config.TIME_COL, config.TWS_COL]
    combined = pd.concat([history_df[cols], target_df[cols]], ignore_index=True)
    ordered = combined.sort_values([config.LAT_COL, config.LON_COL, config.TIME_COL]).copy()
    ordered["_anchor_time"] = ordered[config.TIME_COL].where(ordered[config.TWS_COL].notna())
    ordered["_anchor_time"] = (
        ordered.groupby([config.LAT_COL, config.LON_COL])["_anchor_time"].ffill()
    )
    months_since = (
        (ordered[config.TIME_COL].dt.year - ordered["_anchor_time"].dt.year) * 12
        + (ordered[config.TIME_COL].dt.month - ordered["_anchor_time"].dt.month)
    )
    months_since = months_since.reindex(combined.index)
    return months_since.iloc[len(history_df):].reset_index(drop=True)


def build_all_features(
    history_df: pd.DataFrame, target_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run this project's full derived-feature pipeline - masked-TWS_t fill, spatial
    neighbourhood, seasonal climatology - identically for any (history, target) pair.

    Extracted from src/train.py so that the exact same feature computation used for
    (Train.csv, Test.csv) at submission time can also be used for (fit, validation)
    splits during evaluation - see evaluate.py's mask-aware validation, which needs
    validation-set features computed the same way Test.csv's are (after masking and
    backward-fill), not from history_df/target_df's original, fully-observed columns.
    `history_df` is assumed already fully observed (e.g. Train.csv, or a fit split -
    never masked); `target_df` may have masked/missing TWS_t rows.
    """
    target_filled = fill_masked_tws(history_df, target_df)

    cell_index = (
        pd.concat([history_df[[config.LAT_COL, config.LON_COL]],
                   target_df[[config.LAT_COL, config.LON_COL]]])
        .drop_duplicates()
    )
    cell_id, adjacency, n_cells = build_spatial_adjacency(cell_index)
    history_nb = add_neighbourhood_features(history_df, cell_id, adjacency, n_cells)
    target_nb = add_neighbourhood_features(target_filled, cell_id, adjacency, n_cells)

    n_history = len(history_nb)
    combined = pd.concat([history_nb, target_nb], ignore_index=True)
    combined = add_seasonal_climatology_features(combined)
    history_out = combined.iloc[:n_history].reset_index(drop=True)
    target_out = combined.iloc[n_history:].reset_index(drop=True)
    return history_out, target_out
