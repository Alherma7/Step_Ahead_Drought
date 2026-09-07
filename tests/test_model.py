import numpy as np

from src.model import make_baseline_model, make_lightgbm_model, predict_ensemble


def test_make_lightgbm_model_returns_lgbm_regressor():
    import lightgbm as lgb

    m = make_lightgbm_model()
    assert isinstance(m, lgb.LGBMRegressor)


def test_make_lightgbm_model_shares_boosting_levers_with_baseline():
    baseline_params = make_baseline_model().get_params()
    lgbm_params = make_lightgbm_model().get_params()
    assert lgbm_params["learning_rate"] == baseline_params["learning_rate"]
    assert lgbm_params["max_depth"] == baseline_params["max_depth"]
    assert lgbm_params["random_state"] == baseline_params["random_state"]


class _FixedPredictModel:
    def __init__(self, predictions):
        self._predictions = np.asarray(predictions)

    def predict(self, X):
        return self._predictions


def test_predict_ensemble_averages_across_different_model_classes():
    models = [_FixedPredictModel([1.0, 2.0, 3.0]), _FixedPredictModel([3.0, 4.0, 5.0])]
    result = predict_ensemble(models, X=None)
    assert np.allclose(result, [2.0, 3.0, 4.0])


def test_predict_ensemble_single_model_matches_its_own_prediction():
    models = [_FixedPredictModel([1.0, -2.0, 0.5])]
    result = predict_ensemble(models, X=None)
    assert np.allclose(result, [1.0, -2.0, 0.5])
