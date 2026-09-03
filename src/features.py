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
