from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from src.utils.logging_config import get_logger
from src.utils.helpers import save_model

logger = get_logger(__name__)


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[Dict] = None,
    model_dir: str = "models",
    cv_folds: int = 5,
) -> RandomForestRegressor:
    default_params = {
        "n_estimators": 500,
        "max_depth": 8,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    logger.info(f"Training Random Forest with params: {default_params}")

    tscv = TimeSeriesSplit(n_splits=cv_folds)
    cv_rmse = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_v = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_v = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        fold_model = RandomForestRegressor(**default_params)
        fold_model.fit(X_tr, y_tr)
        preds = fold_model.predict(X_v)
        rmse = np.sqrt(np.mean((preds - y_v.values) ** 2))
        cv_rmse.append(rmse)
        logger.debug(f"  Fold {fold+1}: RMSE={rmse:.6f}")

    logger.info(
        f"Random Forest CV RMSE: {np.mean(cv_rmse):.6f} +/- {np.std(cv_rmse):.6f}"
    )

    model = RandomForestRegressor(**default_params)
    model.fit(X_train, y_train)

    if X_val is not None and y_val is not None:
        val_preds = model.predict(X_val)
        val_rmse = np.sqrt(np.mean((val_preds - y_val.values) ** 2))
        logger.info(f"Random Forest Validation RMSE: {val_rmse:.6f}")

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    save_model(model, Path(model_dir) / "rf_model.pkl")
    logger.info("Random Forest training complete.")
    return model


def predict_random_forest(
    model: RandomForestRegressor,
    X: pd.DataFrame,
) -> np.ndarray:
    return model.predict(X)
