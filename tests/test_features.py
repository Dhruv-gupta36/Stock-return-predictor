import numpy as np
import pandas as pd
import pytest

from src.features.technical import add_technical_features, FEATURE_COLS
from src.features.target import add_target, drop_target_nans


def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    np.random.seed(0)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    close = 100.0 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = close * (1 + np.random.normal(0, 0.003, n))
    volume = np.random.randint(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


class TestTechnicalFeatures:

    def test_output_has_expected_feature_columns(self):
        df = _make_ohlcv(300)
        out = add_technical_features(df)
        missing = [c for c in FEATURE_COLS if c not in out.columns]
        assert missing == [], f"Missing feature columns: {missing}"

    def test_output_preserves_ohlcv_columns(self):
        df = _make_ohlcv(300)
        out = add_technical_features(df)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in out.columns

    def test_rsi_bounds(self):
        df = _make_ohlcv(300)
        out = add_technical_features(df)
        rsi = out["RSI_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_bb_position_mostly_in_range(self):
        df = _make_ohlcv(300)
        out = add_technical_features(df)
        bb_pos = out["BB_position"].dropna()
        pct_in_range = ((bb_pos >= 0) & (bb_pos <= 1)).mean()
        assert pct_in_range > 0.90

    def test_no_future_leakage_in_lag_returns(self):
        df = _make_ohlcv(300)
        out = add_technical_features(df).dropna()
        lag = out["Lag_Return_1d"]
        ret = out["Return_1d"]
        expected = ret.shift(1).dropna()
        actual = lag.loc[expected.index]
        pd.testing.assert_series_equal(actual, expected, check_names=False, rtol=1e-6)

    def test_volume_ratio_non_negative(self):
        df = _make_ohlcv(300)
        out = add_technical_features(df)
        assert (out["Volume_Ratio"].dropna() >= 0).all()


class TestTargetConstruction:

    def test_target_direction_is_binary(self):
        df = _make_ohlcv(200)
        out = add_technical_features(df)
        out = add_target(out)
        out = drop_target_nans(out)
        unique_vals = set(out["Target_Direction"].dropna().unique())
        assert unique_vals.issubset({0.0, 1.0})

    def test_drop_target_nans_removes_rows(self):
        df = _make_ohlcv(200)
        out = add_technical_features(df)
        out = add_target(out)
        n_before = len(out)
        out = drop_target_nans(out)
        assert len(out) < n_before
