from typing import Dict, List

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}
_MAX_MISSING_PCT = 0.05
_MAX_GAP_DAYS = 5


class ValidationResult:
    def __init__(self) -> None:
        self.passed: bool = True
        self.issues: List[str] = []
        self.warnings: List[str] = []

    def fail(self, msg: str) -> None:
        self.passed = False
        self.issues.append(msg)
        logger.error(f"[VALIDATION FAIL] {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning(f"[VALIDATION WARN] {msg}")

    def __repr__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"ValidationResult({status}, "
            f"issues={len(self.issues)}, "
            f"warnings={len(self.warnings)})"
        )


def validate_dataframe(
    df: pd.DataFrame,
    ticker: str = "UNKNOWN",
    min_rows: int = 250,
) -> ValidationResult:
    result = ValidationResult()

    if len(df) < min_rows:
        result.fail(f"{ticker}: only {len(df)} rows - need at least {min_rows}.")

    missing_cols = _REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        result.fail(f"{ticker}: missing columns {missing_cols}.")
        return result

    if not isinstance(df.index, pd.DatetimeIndex):
        result.fail(f"{ticker}: index is not DatetimeIndex.")

    n_dupes = df.index.duplicated().sum()
    if n_dupes > 0:
        result.fail(f"{ticker}: {n_dupes} duplicate date entries.")

    if not df.index.is_monotonic_increasing:
        result.warn(f"{ticker}: index is not sorted ascending - will be sorted.")

    for col in _REQUIRED_COLUMNS & set(df.columns):
        miss_pct = df[col].isna().mean()
        if miss_pct > _MAX_MISSING_PCT:
            result.warn(
                f"{ticker}.{col}: {miss_pct:.1%} missing values "
                f"(threshold {_MAX_MISSING_PCT:.0%})."
            )

    for col in ["Close"]:
        null_mask = df[col].isna()
        if null_mask.any():
            max_gap = _max_consecutive_true(null_mask)
            if max_gap > _MAX_GAP_DAYS:
                result.warn(
                    f"{ticker}.{col}: consecutive NaN gap of {max_gap} days."
                )

    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            n_non_pos = (df[col] <= 0).sum()
            if n_non_pos > 0:
                result.fail(f"{ticker}.{col}: {n_non_pos} non-positive price values.")

    if {"High", "Low", "Close", "Open"}.issubset(df.columns):
        violations = (df["High"] < df["Low"]).sum()
        if violations > 0:
            result.fail(f"{ticker}: {violations} rows where High < Low.")

    if "Volume" in df.columns:
        n_neg_vol = (df["Volume"] < 0).sum()
        if n_neg_vol > 0:
            result.fail(f"{ticker}: {n_neg_vol} negative volume entries.")

    logger.info(f"Validation for {ticker}: {result}")
    return result


def _max_consecutive_true(mask: pd.Series) -> int:
    if not mask.any():
        return 0
    groups = mask.ne(mask.shift()).cumsum()
    return int(mask.groupby(groups).sum().max())


def summarize_validation(results: Dict[str, ValidationResult]) -> None:
    print("\n" + "=" * 60)
    print(f"{'Ticker':<12} {'Status':<10} {'Issues':<8} {'Warnings'}")
    print("=" * 60)
    for ticker, res in results.items():
        status = "PASS" if res.passed else "FAIL"
        print(f"{ticker:<12} {status:<10} {len(res.issues):<8} {len(res.warnings)}")
    print("=" * 60 + "\n")
