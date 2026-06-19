from typing import Optional

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def preprocess_ticker(
    df: pd.DataFrame,
    macro: Optional[pd.DataFrame] = None,
    ticker: str = "UNKNOWN",
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_index()

    n_before = len(df)
    df = df[~df.index.duplicated(keep="last")]
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.warning(f"{ticker}: removed {n_removed} duplicate date rows.")

    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df[price_cols] = df[price_cols].ffill().bfill()

    n_before = len(df)
    df = df.dropna(subset=["Close"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning(f"{ticker}: dropped {n_dropped} rows with NaN Close.")

    if macro is not None and not macro.empty:
        macro_reindexed = macro.reindex(df.index, method="ffill")
        df = pd.concat([df, macro_reindexed], axis=1)
        logger.debug(f"{ticker}: merged macro columns {macro.columns.tolist()}")

    logger.info(f"{ticker}: preprocessing complete. Shape = {df.shape}")
    return df


def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple:
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    logger.info(
        f"Split -> train={len(train)} | val={len(val)} | test={len(test)} "
        f"(total={n})"
    )
    return train, val, test
