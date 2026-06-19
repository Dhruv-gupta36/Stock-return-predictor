import warnings
from pathlib import Path
from typing import Dict, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.utils.logging_config import get_logger
from src.utils.helpers import save_model

warnings.filterwarnings("ignore", category=UserWarning)
logger = get_logger(__name__)


def _lgb_objective(trial, X_train, y_train, cv_folds):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "n_estimators": 1000,
        "random_state": 42,
    }

    tscv = TimeSeriesSplit(n_splits=cv_folds)
    rmse_scores = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        mdl = lgb.LGBMRegressor(**params)
        mdl.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        preds = mdl.predict(X_val)
        rmse_scores.append(np.sqrt(np.mean((preds - y_val.values) ** 2)))

    return float(np.mean(rmse_scores))


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Optional[Dict] = None,
    use_optuna: bool = True,
    n_trials: int = 50,
    cv_folds: int = 5,
    model_dir: str = "models",
) -> lgb.LGBMRegressor:
    if use_optuna:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logger.info(f"Starting Optuna search ({n_trials} trials)...")
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: _lgb_objective(trial, X_train, y_train, cv_folds),
            n_trials=n_trials,
            show_progress_bar=True,
        )
        best_params = study.best_params
        best_params.update({
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "n_estimators": 2000,
            "random_state": 42,
        })
        logger.info(f"Best Optuna params: {best_params}")
        logger.info(f"Best CV RMSE: {study.best_value:.6f}")
    else:
        best_params = params or {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "n_estimators": 1000,
            "random_state": 42,
        }

    logger.info("Training final LightGBM model...")
    model = lgb.LGBMRegressor(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    val_preds = model.predict(X_val)
    val_rmse = np.sqrt(np.mean((val_preds - y_val.values) ** 2))
    logger.info(f"LightGBM Validation RMSE: {val_rmse:.6f}")

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    save_model(model, Path(model_dir) / "lgbm_model.pkl")
    logger.info("LightGBM training complete.")
    return model


def predict_lightgbm(model: lgb.LGBMRegressor, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)
