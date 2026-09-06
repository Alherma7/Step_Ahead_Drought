"""Model builders.

make_persistence_prediction() reproduces, unmodified, the official benchmark from
notebooks/00_official_starter_reference.ipynb - the bar every validated change in
this project must clear (see RESOURCES.md).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src import config


def make_persistence_prediction(df: pd.DataFrame) -> np.ndarray:
    """TWS(t+1) ~= TWS(t). The naive baseline every model must beat."""
    return df[config.TWS_COL].to_numpy(dtype=np.float32)


def make_baseline_model() -> HistGradientBoostingRegressor:
    """HistGradientBoostingRegressor over the base+optional feature columns.

    Source: notebooks/00_official_starter_reference.ipynb for the hyperparameters.
    Feeds NaN feature values (masked TWS_t and never-observed neighbourhood/
    climatology/prior-reading columns) directly to the model rather than
    median-imputing them first - HistGradientBoostingRegressor has native missing-
    value support (scikit-learn 1.7.2 docs: the tree grower learns per split whether
    missing values go left or right, based on potential gain), which keeps the "this
    was missing" signal instead of discarding it. Validated in
    notebooks/12_native_missing_value_handling.ipynb: real Zindi leaderboard RMSE
    improved 0.755129 -> 0.753811 (~0.17%), consistent with the 5-seed proxy's
    directionally-similar (weaker) signal.
    """
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=300,
        max_depth=8,
        min_samples_leaf=50,
        l2_regularization=1.0,
        random_state=config.RANDOM_STATE,
    )


def predict(model: HistGradientBoostingRegressor, X: np.ndarray) -> np.ndarray:
    """Score new, unseen rows with a fitted model."""
    return model.predict(X)
