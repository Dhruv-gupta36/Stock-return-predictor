import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

FEATURE_COLS = [
    "MA_5_ratio",
    "MA_10_ratio",
    "MA_20_ratio",
    "MA_50_ratio",
    "MA_200_ratio",
    "EMA_12",
    "EMA_26",
    "MACD",
    "MACD_signal",
    "MACD_histogram",
    "RSI_14",
    "Stoch_K",
    "Stoch_D",
    "Return_1d",
    "Return_2d",
    "Return_5d",
    "Return_10d",
    "Return_20d",
    "Log_Return_1d",
    "Lag_Return_1d",
    "Lag_Return_2d",
    "Lag_Return_3d",
    "Lag_Return_5d",
    "Lag_Return_10d",
    "RollingVol_5d",
    "RollingVol_10d",
    "RollingVol_20d",
    "RollingVol_60d",
    "ATR_14",
    "Volume_Ratio",
    "OBV_Ratio",
    "BB_width",
    "BB_position",
    "DayOfWeek",
    "Month",
    "IsMonday",
    "IsFriday",
]


def add_technical_features(
    df: pd.DataFrame,
    ma_windows: list = None,
    ema_spans: list = None,
    rsi_period: int = 14,
    atr_period: int = 14,
    bb_window: int = 20,
    bb_std: float = 2.0,
    vol_windows: list = None,
    lag_windows: list = None,
    return_windows: list = None,
    volume_ma_window: int = 20,
    stoch_period: int = 14,
    stoch_smooth: int = 3,
) -> pd.DataFrame:
    if ma_windows is None:
        ma_windows = [5, 10, 20, 50, 200]
    if ema_spans is None:
        ema_spans = [12, 26]
    if vol_windows is None:
        vol_windows = [5, 10, 20, 60]
    if lag_windows is None:
        lag_windows = [1, 2, 3, 5, 10]
    if return_windows is None:
        return_windows = [1, 2, 5, 10, 20]

    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    for w in ma_windows:
        ma_col = f"MA_{w}"
        df[ma_col] = close.rolling(w).mean()
        df[f"MA_{w}_ratio"] = close / (df[ma_col] + 1e-10)

    for span in ema_spans:
        df[f"EMA_{span}"] = close.ewm(span=span, adjust=False).mean()

    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_histogram"] = df["MACD"] - df["MACD_signal"]

    for w in return_windows:
        df[f"Return_{w}d"] = close.pct_change(w)
    df["Log_Return_1d"] = np.log(close / close.shift(1))

    for lag in lag_windows:
        df[f"Lag_Return_{lag}d"] = df["Return_1d"].shift(lag)

    for w in vol_windows:
        df[f"RollingVol_{w}d"] = (
            df["Return_1d"].rolling(w).std() * np.sqrt(252)
        )

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["TR"] = tr
    df[f"ATR_{atr_period}"] = tr.rolling(atr_period).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / (loss + 1e-10)
    df[f"RSI_{rsi_period}"] = 100.0 - (100.0 / (1.0 + rs))

    lowest_low = low.rolling(stoch_period).min()
    highest_high = high.rolling(stoch_period).max()
    df["Stoch_K"] = (
        100.0 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    )
    df["Stoch_D"] = df["Stoch_K"].rolling(stoch_smooth).mean()

    df["Volume_MA_20"] = volume.rolling(volume_ma_window).mean()
    df["Volume_Ratio"] = volume / (df["Volume_MA_20"] + 1e-10)

    obv = (np.sign(df["Return_1d"]) * volume).cumsum()
    df["OBV"] = obv
    df["OBV_MA_20"] = obv.rolling(volume_ma_window).mean()
    df["OBV_Ratio"] = obv / (df["OBV_MA_20"].abs() + 1e-10)

    df["BB_mid"] = close.rolling(bb_window).mean()
    df["BB_std"] = close.rolling(bb_window).std()
    df["BB_upper"] = df["BB_mid"] + bb_std * df["BB_std"]
    df["BB_lower"] = df["BB_mid"] - bb_std * df["BB_std"]
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / (df["BB_mid"] + 1e-10)
    df["BB_position"] = (close - df["BB_lower"]) / (
        df["BB_upper"] - df["BB_lower"] + 1e-10
    )

    df["DayOfWeek"] = df.index.dayofweek
    df["Month"] = df.index.month
    df["Quarter"] = df.index.quarter
    df["IsMonday"] = (df.index.dayofweek == 0).astype(int)
    df["IsFriday"] = (df.index.dayofweek == 4).astype(int)

    logger.info(f"Technical features computed. Total columns: {df.shape[1]}")
    return df
