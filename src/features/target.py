import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def add_target(
    df: pd.DataFrame,
    horizon: int = 1,
    target_col: str = "Target_Return",
    direction_col: str = "Target_Direction",
) -> pd.DataFrame:
    df = df.copy()
    df[target_col] = df["Return_1d"].shift(-horizon)
    df[direction_col] = (df[target_col] > 0).astype(float)
    logger.info(
        f"Target columns added: '{target_col}' and '{direction_col}' "
        f"(horizon={horizon}d)"
    )
    return df


def drop_target_nans(
    df: pd.DataFrame,
    target_col: str = "Target_Return",
) -> pd.DataFrame:
    n_before = len(df)
    df = df.dropna(subset=[target_col])
    logger.info(
        f"Dropped {n_before - len(df)} NaN target rows. "
        f"Remaining: {len(df)}"
    )
    return df
