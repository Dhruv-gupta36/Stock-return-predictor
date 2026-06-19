import numpy as np
import pandas as pd
import pytest

from src.models.random_forest import train_random_forest, predict_random_forest
from src.evaluation.metrics import (
    compute_regression_metrics,
    directional_accuracy,
    information_coefficient,
)


def _make_xy(n: int = 500, n_feat: int = 10, seed: int = 0):
    np.random.seed(seed)
    X = pd.DataFrame(
        np.random.randn(n, n_feat),
        columns=[f"f{i}" for i in range(n_feat)],
    )
    y = pd.Series(0.3 * X["f0"] + np.random.randn(n) * 0.5)
    return X, y


class TestRandomForest:

    def test_train_and_predict_shape(self, tmp_path):
        X, y = _make_xy(300)
        train_n = 200
        model = train_random_forest(
            X.iloc[:train_n], y.iloc[:train_n],
            params={"n_estimators": 10, "n_jobs": 1, "random_state": 0},
            model_dir=str(tmp_path),
            cv_folds=2,
        )
        preds = predict_random_forest(model, X.iloc[train_n:])
        assert preds.shape == (len(X) - train_n,)

    def test_model_artifact_saved(self, tmp_path):
        X, y = _make_xy(200)
        train_random_forest(
            X.iloc[:150], y.iloc[:150],
            params={"n_estimators": 5, "n_jobs": 1, "random_state": 0},
            model_dir=str(tmp_path),
            cv_folds=2,
        )
        assert (tmp_path / "rf_model.pkl").exists()


class TestMetrics:

    def test_directional_accuracy_perfect(self):
        y = np.array([0.01, -0.02, 0.03, -0.01, 0.005])
        assert directional_accuracy(y, y) == 1.0

    def test_directional_accuracy_opposite(self):
        y = np.array([0.01, -0.02, 0.03])
        assert directional_accuracy(y, -y) == 0.0

    def test_ic_perfect_rank_correlation(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ic = information_coefficient(y, y)
        assert abs(ic - 1.0) < 1e-6

    def test_compute_regression_metrics_keys(self):
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1
        metrics = compute_regression_metrics(y_true, y_pred)
        for key in ["RMSE", "MAE", "R2", "Directional_Accuracy", "IC"]:
            assert key in metrics

    def test_rmse_is_non_negative(self):
        y = np.random.randn(50)
        metrics = compute_regression_metrics(y, y + 0.1)
        assert metrics["RMSE"] >= 0

    def test_handles_nan_inputs_gracefully(self):
        y_true = np.array([0.01, np.nan, 0.02, -0.01])
        y_pred = np.array([0.005, 0.01, np.nan, -0.005])
        metrics = compute_regression_metrics(y_true, y_pred)
        assert "RMSE" in metrics
