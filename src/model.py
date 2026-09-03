"""Pipeline builders.

make_persistence_prediction() and make_baseline_model() reproduce, unmodified, the
official benchmark from notebooks/00_official_starter_reference.ipynb - the bar every
validated change in this project must clear (see RESOURCES.md).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src import config


def make_persistence_prediction(df: pd.DataFrame) -> np.ndarray:
    """TWS(t+1) ~= TWS(t). The naive baseline every model must beat."""
    return df[config.TWS_COL].to_numpy(dtype=np.float32)


def make_baseline_model() -> Pipeline:
    """HistGradientBoostingRegressor over the base+optional feature columns.

    Source: notebooks/00_official_starter_reference.ipynb. Median-imputes TWS_t for
    the ~66.5% of test rows where it is masked (config.TWS_MASKED_COL).
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("gbr", HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=300,
            max_depth=8,
            min_samples_leaf=50,
            l2_regularization=1.0,
            random_state=config.RANDOM_STATE,
        )),
    ])


def predict(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Score new, unseen rows with a fitted pipeline."""
    return model.predict(X)
