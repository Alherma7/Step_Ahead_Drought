"""Model builders.

make_persistence_prediction() reproduces, unmodified, the official benchmark from
notebooks/00_official_starter_reference.ipynb - the bar every validated change in
this project must clear (see RESOURCES.md).
"""
import lightgbm as lgb
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


def make_lightgbm_model() -> lgb.LGBMRegressor:
    """LightGBM regressor - a second, genuinely different gradient-boosting
    implementation from make_baseline_model()'s HistGradientBoostingRegressor,
    for the two-model ensemble in predict_ensemble(). Same boosting levers
    (learning_rate/depth/min samples per leaf/L2) as make_baseline_model, so
    the ensemble's diversity comes from the two libraries' different tree-growth
    and split-finding algorithms, not from a deliberately different hyperparameter
    budget.

    Native missing-value handling (LightGBM docs, "Missing Value Handling":
    NaN is treated as missing and both split directions are tried during
    training to pick the better one - the same default-direction-learning
    idea as HistGradientBoostingRegressor's), so it can ingest the same raw
    NaN-containing feature matrix as the baseline model - preserves this
    project's P5 finding that missingness in the climatology/anchor-age
    columns is itself informative (notebooks/12_native_missing_value_handling.ipynb).

    Source: DrivenData "Water Supply Forecast Rodeo" winner solutions repo
    (P6, notebooks/13_ensemble.ipynb) - motivated combining a second, genuinely
    different model family after notebooks/13's same-model seed-averaging
    variant showed no real signal. LightGBM specifically is the model family
    used by the M5 Forecasting - Accuracy 1st place solution already cited for
    P5 (RESOURCES.md).
    """
    return lgb.LGBMRegressor(
        objective="regression",
        learning_rate=0.05,
        n_estimators=300,
        max_depth=8,
        min_child_samples=50,
        reg_lambda=1.0,
        random_state=config.RANDOM_STATE,
        verbosity=-1,
    )


def predict_ensemble(models: list, X: np.ndarray) -> np.ndarray:
    """Average predictions across a list of already-fitted models (any mix of
    model classes, as long as each exposes .predict(X))."""
    predictions = np.column_stack([m.predict(X) for m in models])
    return predictions.mean(axis=1)
