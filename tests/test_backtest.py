import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import long_short_backtest, compute_strategy_metrics
from src.evaluation.monte_carlo import monte_carlo_backtest


def _make_returns(n: int = 252, seed: int = 0) -> pd.Series:
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(np.random.normal(0.0005, 0.01, n), index=dates)


def _make_signals(returns: pd.Series, noise: float = 0.3) -> pd.Series:
    np.random.seed(42)
    noise_series = pd.Series(np.random.normal(0, noise, len(returns)), index=returns.index)
    return returns + noise_series


class TestBacktest:

    def test_output_columns_present(self):
        r = _make_returns()
        s = _make_signals(r)
        df = long_short_backtest(r, s)
        for col in ["strategy_return", "buyhold_return", "strategy_cum", "buyhold_cum"]:
            assert col in df.columns

    def test_buyhold_return_equals_actual(self):
        r = _make_returns()
        s = _make_signals(r)
        df = long_short_backtest(r, s, transaction_cost=0.0)
        pd.testing.assert_series_equal(
            df["buyhold_return"], df["actual_return"], check_names=False
        )

    def test_strategy_cum_starts_near_one(self):
        r = _make_returns()
        s = _make_signals(r)
        df = long_short_backtest(r, s)
        assert abs(df["strategy_cum"].iloc[0] - 1.0) < 0.05

    def test_compute_metrics_returns_all_keys(self):
        r = _make_returns()
        s = _make_signals(r)
        df = long_short_backtest(r, s)
        metrics = compute_strategy_metrics(df)
        for key in [
            "Strategy_Sharpe", "BuyHold_Sharpe",
            "Strategy_Annual_Return", "BuyHold_Annual_Return",
            "Strategy_MaxDrawdown", "BuyHold_MaxDrawdown",
        ]:
            assert key in metrics

    def test_max_drawdown_non_positive(self):
        r = _make_returns()
        s = _make_signals(r)
        df = long_short_backtest(r, s)
        metrics = compute_strategy_metrics(df)
        assert metrics["Strategy_MaxDrawdown"] <= 0
        assert metrics["BuyHold_MaxDrawdown"] <= 0


class TestMonteCarlo:

    def test_output_shape(self):
        r = _make_returns()
        s = _make_signals(r)
        df = long_short_backtest(r, s)
        results = monte_carlo_backtest(
            df["strategy_return"].dropna(),
            n_simulations=50,
            n_trading_days=100,
        )
        assert results["simulated_curves"].shape == (50, 100)

    def test_prob_profit_in_range(self):
        r = _make_returns()
        s = _make_signals(r)
        df = long_short_backtest(r, s)
        results = monte_carlo_backtest(df["strategy_return"].dropna(), n_simulations=100)
        assert 0.0 <= results["prob_profit"] <= 1.0

    def test_percentile_ordering(self):
        r = _make_returns()
        s = _make_signals(r)
        df = long_short_backtest(r, s)
        results = monte_carlo_backtest(df["strategy_return"].dropna(), n_simulations=100)
        assert results["p5_final_value"] <= results["median_final_value"]
        assert results["median_final_value"] <= results["p95_final_value"]

    def test_empty_returns_raises(self):
        with pytest.raises(ValueError):
            monte_carlo_backtest(pd.Series([], dtype=float), n_simulations=10)
