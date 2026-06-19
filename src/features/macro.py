import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def add_macro_features(
    df: pd.DataFrame,
    macro: pd.DataFrame,
) -> pd.DataFrame:
    if macro.empty:
        logger.warning("Macro DataFrame is empty - skipping macro merge.")
        return df

    macro_aligned = macro.reindex(df.index, method="ffill")

    if "VIX" in macro_aligned.columns:
        macro_aligned["VIX_MA_10"] = macro_aligned["VIX"].rolling(10).mean()
        macro_aligned["VIX_Ratio"] = macro_aligned["VIX"] / (
            macro_aligned["VIX_MA_10"] + 1e-10
        )
        macro_aligned["VIX_Return_1d"] = macro_aligned["VIX"].pct_change(1)

    if "FedFunds" in macro_aligned.columns:
        macro_aligned["FedFunds_Change"] = macro_aligned["FedFunds"].diff(1)

    df = pd.concat([df, macro_aligned], axis=1)
    logger.info(f"Macro features merged. New columns: {macro_aligned.columns.tolist()}")
    return df
