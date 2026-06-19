import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf

from src.utils.logging_config import get_logger
from src.utils.helpers import ensure_dirs, save_dataframe

logger = get_logger(__name__)

_FRED_SERIES = {
    "VIX": "VIXCLS",
    "FedFunds": "DFF",
    "CPI": "CPIAUCSL",
    "T10Y2Y": "T10Y2Y",
}


def download_stock_data(
    tickers: List[str],
    start: str,
    end: str,
    raw_dir: str = "data/raw",
    delay: float = 1.0,
) -> Dict[str, pd.DataFrame]:
    ensure_dirs(raw_dir)
    all_data: Dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        try:
            logger.info(f"Downloading {ticker} [{start} to {end}]")
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
            )
            if df.empty:
                logger.warning(f"No data returned for {ticker} - skipping.")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            df.index.name = "Date"
            df["Ticker"] = ticker
            all_data[ticker] = df

            out_path = Path(raw_dir) / f"{ticker}.csv"
            save_dataframe(df, out_path)
            logger.info(f"  {ticker}: {len(df)} rows -> {out_path}")
            time.sleep(delay)

        except Exception as exc:
            logger.error(f"Failed to download {ticker}: {exc}")

    logger.info(f"Download complete. {len(all_data)}/{len(tickers)} tickers succeeded.")
    return all_data


def download_macro_data(
    start: str,
    end: str,
    external_dir: str = "data/external",
) -> pd.DataFrame:
    ensure_dirs(external_dir)
    frames: Dict[str, pd.Series] = {}

    try:
        import pandas_datareader as pdr

        for col_name, series_id in _FRED_SERIES.items():
            try:
                logger.info(f"Fetching FRED series {series_id} ({col_name})")
                s = pdr.get_data_fred(series_id, start=start, end=end).squeeze()
                frames[col_name] = s
            except Exception as exc:
                logger.warning(f"FRED series {series_id} failed: {exc}")

    except ImportError:
        logger.warning("pandas_datareader not available - macro data will be empty.")

    if frames:
        macro = pd.DataFrame(frames)
        macro.index.name = "Date"
        macro = macro.ffill().bfill()
    else:
        logger.warning("No macro data retrieved. Creating empty placeholder.")
        macro = pd.DataFrame(columns=list(_FRED_SERIES.keys()))

    out_path = Path(external_dir) / "macro.csv"
    macro.to_csv(out_path)
    logger.info(f"Macro data saved -> {out_path}  shape={macro.shape}")
    return macro


def load_raw_ticker(ticker: str, raw_dir: str = "data/raw") -> pd.DataFrame:
    path = Path(raw_dir) / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data for {ticker} not found at {path}. Run download first."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    logger.debug(f"Loaded raw {ticker}: {df.shape}")
    return df


def load_macro(external_dir: str = "data/external") -> pd.DataFrame:
    path = Path(external_dir) / "macro.csv"
    if not path.exists():
        logger.warning(f"Macro data not found at {path}. Returning empty DataFrame.")
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    logger.debug(f"Loaded macro data: {df.shape}")
    return df
