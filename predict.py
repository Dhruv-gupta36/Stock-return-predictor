import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils.logging_config import setup_logging, get_logger
from src.utils.helpers import load_config, load_model
from src.features.technical import add_technical_features, FEATURE_COLS
from src.data.preprocess import preprocess_ticker

setup_logging(level="INFO", log_dir=None)
logger = get_logger("predict")


def load_and_predict(ticker: str, model_dir: str, config_path: str) -> dict:
    cfg = load_config(config_path)
    feat_cfg = cfg["features"]

    model_path = Path(model_dir) / "lgbm_model.pkl"
    model = load_model(model_path)

    logger.info(f"Downloading recent data for {ticker}...")
    raw = yf.download(ticker, period="300d", auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No price data found for ticker: {ticker}")

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
        raise ValueError(f"Not enough data to compute features for {ticker}.")

    available_features = [c for c in FEATURE_COLS if c in featured.columns]
    latest_features = featured[available_features].iloc[[-1]]
    prediction_date = featured.index[-1].strftime("%Y-%m-%d")
    current_price = float(clean["Close"].iloc[-1])

    predicted_return = float(model.predict(latest_features)[0])
    direction = "UP" if predicted_return > 0 else "DOWN"
    confidence = min(abs(predicted_return) / 0.05, 1.0)

    return {
        "ticker": ticker.upper(),
        "prediction_date": prediction_date,
        "predicted_return": round(predicted_return, 6),
        "predicted_return_pct": f"{predicted_return * 100:+.3f}%",
        "predicted_direction": direction,
        "confidence": round(confidence, 4),
        "current_price": round(current_price, 2),
        "model": str(model_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Stock Return Predictor - CLI")
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--output", type=str, choices=["table", "json"], default="table")
    return parser.parse_args()


def main():
    args = parse_args()
    ticker = args.ticker.upper()
    model_dir = args.model_dir or f"models/{ticker}"

    try:
        result = load_and_predict(ticker, model_dir, args.config)

        if args.output == "json":
            print(json.dumps(result, indent=2))
        else:
            print("\n" + "=" * 50)
            print(f"  Stock Return Predictor - {ticker}")
            print("=" * 50)
            for k, v in result.items():
                if k != "model":
                    print(f"  {k:<28s} {v}")
            print("=" * 50 + "\n")

    except FileNotFoundError as exc:
        logger.error(f"Model not found: {exc}")
        logger.error("Run train.py first.")
        sys.exit(1)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Prediction failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
