import argparse
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from src.utils.logging_config import setup_logging, get_logger
from src.utils.helpers import load_config, set_seed, ensure_dirs, Timer
from src.data.download import download_stock_data, download_macro_data, load_raw_ticker, load_macro
from src.data.validate import validate_dataframe
from src.data.preprocess import preprocess_ticker, train_val_test_split
from src.features.technical import add_technical_features, FEATURE_COLS
from src.features.target import add_target, drop_target_nans
from src.models.random_forest import train_random_forest
from src.models.lightgbm_model import train_lightgbm
from src.models.lstm_model import train_lstm, predict_lstm
from src.evaluation.metrics import compute_regression_metrics, print_metrics
from src.evaluation.backtest import long_short_backtest, compute_strategy_metrics, plot_backtest
from src.evaluation.monte_carlo import monte_carlo_backtest, plot_monte_carlo
from src.explainability.shap_analysis import generate_shap_analysis

CFG_PATH = "configs/config.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stock Return Predictor")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--no-optuna", action="store_true")
    parser.add_argument("--no-lstm", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--config", type=str, default=CFG_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    setup_logging(
        level=cfg["logging"]["level"],
        log_dir=cfg["logging"]["log_dir"],
        log_file="train.log",
    )
    logger = get_logger("train")
    logger.info("=" * 60)
    logger.info("  Stock Return Predictor - Training Pipeline")
    logger.info("=" * 60)

    set_seed(42)

    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]
    train_cfg = cfg["training"]
    eval_cfg = cfg["evaluation"]
    mc_cfg = cfg["monte_carlo"]

    tickers = [args.ticker] if args.ticker else data_cfg["tickers"]
    use_optuna = not args.no_optuna
    use_lstm = not args.no_lstm
    use_shap = not args.no_shap

    ensure_dirs(
        data_cfg["raw_dir"],
        data_cfg["processed_dir"],
        data_cfg["external_dir"],
        train_cfg["model_dir"],
        "reports/figures",
        "logs",
    )

    if not args.no_download:
        with Timer("Data download"):
            download_stock_data(
                tickers=tickers,
                start=data_cfg["start_date"],
                end=data_cfg["end_date"],
                raw_dir=data_cfg["raw_dir"],
            )
            download_macro_data(
                start=data_cfg["start_date"],
                end=data_cfg["end_date"],
                external_dir=data_cfg["external_dir"],
            )
    else:
        logger.info("Skipping download - using existing data files.")

    macro = load_macro(data_cfg["external_dir"])
    all_results = {}

    for ticker in tickers:
        logger.info(f"\n{'-' * 50}")
        logger.info(f"  Processing: {ticker}")
        logger.info(f"{'-' * 50}")

        try:
            with Timer(f"{ticker} full pipeline"):

                raw = load_raw_ticker(ticker, data_cfg["raw_dir"])

                val_result = validate_dataframe(raw, ticker=ticker)
                if not val_result.passed:
                    logger.error(f"Validation failed for {ticker} - skipping.")
                    continue

                clean = preprocess_ticker(raw, macro=macro, ticker=ticker)

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

                featured = add_target(
                    featured,
                    horizon=cfg["target"]["horizon"],
                    target_col=cfg["target"]["column"],
                    direction_col=cfg["target"]["direction_column"],
                )
                featured = drop_target_nans(featured, cfg["target"]["column"])

                proc_path = Path(data_cfg["processed_dir"]) / f"{ticker}_features.csv"
                featured.to_csv(proc_path)
                logger.info(f"Features saved -> {proc_path}")

                target_col = cfg["target"]["column"]
                available_features = [c for c in FEATURE_COLS if c in featured.columns]
                logger.info(f"Using {len(available_features)} feature columns.")

                train_df, val_df, test_df = train_val_test_split(
                    featured,
                    train_ratio=data_cfg["train_ratio"],
                    val_ratio=data_cfg["val_ratio"],
                )

                X_train = train_df[available_features]
                y_train = train_df[target_col]
                X_val = val_df[available_features]
                y_val = val_df[target_col]
                X_test = test_df[available_features]
                y_test = test_df[target_col]

                logger.info(
                    f"Split sizes - train={len(X_train)}, "
                    f"val={len(X_val)}, test={len(X_test)}"
                )

                ticker_model_dir = f"{train_cfg['model_dir']}/{ticker}"

                logger.info("Training Random Forest...")
                rf_model = train_random_forest(
                    X_train, y_train,
                    X_val=X_val, y_val=y_val,
                    params=cfg["models"]["random_forest"],
                    model_dir=ticker_model_dir,
                    cv_folds=train_cfg["cv_folds"],
                )
                rf_preds_test = rf_model.predict(X_test)
                rf_metrics = compute_regression_metrics(y_test.values, rf_preds_test)
                print_metrics(rf_metrics, f"{ticker} - Random Forest (Test)")

                logger.info("Training LightGBM...")
                lgbm_model = train_lightgbm(
                    X_train, y_train,
                    X_val=X_val, y_val=y_val,
                    params=cfg["models"]["lightgbm"],
                    use_optuna=use_optuna and train_cfg["use_optuna"],
                    n_trials=cfg["models"]["optuna"]["n_trials"],
                    cv_folds=train_cfg["cv_folds"],
                    model_dir=ticker_model_dir,
                )
                lgbm_preds_test = lgbm_model.predict(X_test)
                lgbm_metrics = compute_regression_metrics(y_test.values, lgbm_preds_test)
                print_metrics(lgbm_metrics, f"{ticker} - LightGBM (Test)")

                lstm_preds_test = None
                if use_lstm:
                    logger.info("Training LSTM...")
                    lstm_cfg = cfg["models"]["lstm"]

                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    X_test_scaled = scaler.transform(X_test)

                    scaler_dir = Path(train_cfg["model_dir"]) / ticker
                    scaler_dir.mkdir(parents=True, exist_ok=True)
                    joblib.dump(scaler, scaler_dir / "scaler.joblib")

                    lstm_model = train_lstm(
                        X_train_scaled, y_train.values,
                        X_val_scaled, y_val.values,
                        seq_len=lstm_cfg["sequence_length"],
                        hidden_size=lstm_cfg["hidden_size"],
                        num_layers=lstm_cfg["num_layers"],
                        dropout=lstm_cfg["dropout"],
                        batch_size=lstm_cfg["batch_size"],
                        epochs=lstm_cfg["epochs"],
                        lr=lstm_cfg["learning_rate"],
                        weight_decay=lstm_cfg["weight_decay"],
                        patience=lstm_cfg["patience"],
                        gradient_clip=lstm_cfg["gradient_clip"],
                        model_dir=str(scaler_dir),
                    )

                    lstm_preds_test = predict_lstm(
                        lstm_model, X_test_scaled,
                        seq_len=lstm_cfg["sequence_length"],
                    )
                    n_lstm = len(lstm_preds_test)
                    y_test_lstm = y_test.values[-n_lstm:]
                    lstm_metrics = compute_regression_metrics(y_test_lstm, lstm_preds_test)
                    print_metrics(lstm_metrics, f"{ticker} - LSTM (Test)")

                if use_shap:
                    logger.info("Running SHAP analysis...")
                    generate_shap_analysis(
                        lgbm_model,
                        X_test,
                        save_dir=f"reports/figures/{ticker}",
                    )

                logger.info("Running backtest...")
                backtest_df = long_short_backtest(
                    returns=test_df["Return_1d"],
                    signals=pd.Series(lgbm_preds_test, index=test_df.index),
                    transaction_cost=eval_cfg["transaction_cost"],
                )
                strategy_metrics = compute_strategy_metrics(
                    backtest_df,
                    risk_free_rate=eval_cfg["risk_free_rate"],
                    trading_days=eval_cfg["trading_days"],
                )
                print_metrics(strategy_metrics, f"{ticker} - Strategy Metrics")
                plot_backtest(
                    backtest_df,
                    title=f"{ticker} - LightGBM Strategy vs Buy-and-Hold",
                    save_path=f"reports/figures/{ticker}/backtest.png",
                )

                logger.info("Running Monte Carlo simulation...")
                mc_results = monte_carlo_backtest(
                    strategy_returns=backtest_df["strategy_return"].dropna(),
                    n_simulations=mc_cfg["n_simulations"],
                    n_trading_days=mc_cfg["n_trading_days"],
                )
                plot_monte_carlo(
                    mc_results,
                    title=f"{ticker} - Monte Carlo Strategy Simulation",
                    save_path=f"reports/figures/{ticker}/monte_carlo.png",
                )

                all_results[ticker] = {
                    "rf_metrics": rf_metrics,
                    "lgbm_metrics": lgbm_metrics,
                    "strategy_metrics": strategy_metrics,
                    "mc_prob_profit": mc_results["prob_profit"],
                }

        except Exception as exc:
            logger.error(f"Pipeline failed for {ticker}: {exc}", exc_info=True)

    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE SUMMARY")
    logger.info("=" * 60)
    for ticker, res in all_results.items():
        logger.info(
            f"  {ticker}: "
            f"LGB DA={res['lgbm_metrics']['Directional_Accuracy']:.3f} | "
            f"Sharpe={res['strategy_metrics']['Strategy_Sharpe']:.3f} | "
            f"MC Prob Profit={res['mc_prob_profit']:.1%}"
        )
    logger.info("Training pipeline complete.")


if __name__ == "__main__":
    main()
