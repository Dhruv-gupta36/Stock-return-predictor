import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from src.utils.logging_config import get_logger
from src.utils.helpers import load_config, load_model
from src.data.preprocess import preprocess_ticker
from src.features.technical import add_technical_features, FEATURE_COLS

logger = get_logger(__name__)

_model_cache: Dict[str, object] = {}
_config: Optional[dict] = None
_MODEL_BASE_DIR = Path(os.environ.get("MODEL_DIR", "models"))
_CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/config.yaml")


def get_config() -> dict:
    global _config
    if _config is None:
        _config = load_config(_CONFIG_PATH)
    return _config


def get_available_tickers() -> List[str]:
    if not _MODEL_BASE_DIR.exists():
        return []
    return [
        d.name
        for d in _MODEL_BASE_DIR.iterdir()
        if d.is_dir() and (d / "lgbm_model.pkl").exists()
    ]


def load_model_for_ticker(ticker: str) -> object:
    if ticker not in _model_cache:
        model_path = _MODEL_BASE_DIR / ticker / "lgbm_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found for {ticker} at {model_path}. "
                f"Run: python train.py --ticker {ticker}"
            )
        _model_cache[ticker] = load_model(model_path)
        logger.info(f"Model loaded for {ticker} and cached.")
    return _model_cache[ticker]


def preload_all_models() -> None:
    available = get_available_tickers()
    if not available:
        logger.warning("No trained models found. Run train.py before starting the API.")
        return
    for ticker in available:
        try:
            load_model_for_ticker(ticker)
        except Exception as exc:
            logger.warning(f"Could not preload model for {ticker}: {exc}")
    logger.info(f"Preloaded models for: {available}")


def predict_for_ticker(ticker: str) -> dict:
    cfg = get_config()
    feat_cfg = cfg["features"]
    model = load_model_for_ticker(ticker)

    logger.info(f"Fetching market data for {ticker}...")
    raw = yf.download(ticker, period="300d", auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"yfinance returned no data for ticker '{ticker}'.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    clean = preprocess_ticker(raw, macro=None, ticker=ticker)

    featured = add_technical_features(
        clean,
        ma_windows=feat_cfg["ma_windows"],
        ema_spans=feat_cfg["ema_spans"],
        rsi_period=feat_cfg["rsi_period"],
        atr_period=feat_cfg["atr_period"],
        bb_window=feat_cfg["bb_window"],
        bb_std=feat_cfg["bb_std"],
        vol_windows=feat_cfg["vol_windows"],
        lag_windows=feat_cfg["lag_windows"],
        return_windows=feat_cfg["return_windows"],
        volume_ma_window=feat_cfg["volume_ma_window"],
    )
    featured = featured.dropna()

    if featured.empty:
        raise ValueError(
            f"Could not compute features for {ticker}. Insufficient historical data."
        )

    available_features = [c for c in FEATURE_COLS if c in featured.columns]
    latest = featured[available_features].iloc[[-1]]
    prediction_date = featured.index[-1].strftime("%Y-%m-%d")
    current_price = float(clean["Close"].iloc[-1])

    predicted_return = float(model.predict(latest)[0])
    direction = "UP" if predicted_return > 0 else "DOWN"
    confidence = min(abs(predicted_return) / 0.05, 1.0)

    return {
        "ticker": ticker,
        "prediction_date": prediction_date,
        "predicted_return": round(predicted_return, 6),
        "predicted_return_pct": f"{predicted_return * 100:+.3f}%",
        "predicted_direction": direction,
        "confidence": round(confidence, 4),
        "current_price": round(current_price, 2),
    }
