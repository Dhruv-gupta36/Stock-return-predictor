from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def long_short_backtest(
    returns: pd.Series,
    signals: pd.Series,
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    df = pd.DataFrame({
        "actual_return": returns,
        "signal": signals,
    }).dropna()

    df["position"] = np.sign(df["signal"])
    df["position_change"] = df["position"].diff().abs().fillna(0.0)
    df["transaction_cost_paid"] = df["position_change"] * transaction_cost
    df["strategy_return"] = (
        df["position"].shift(1).fillna(0.0) * df["actual_return"]
        - df["transaction_cost_paid"]
    )
    df["buyhold_return"] = df["actual_return"]
    df["strategy_cum"] = (1.0 + df["strategy_return"]).cumprod()
    df["buyhold_cum"] = (1.0 + df["buyhold_return"]).cumprod()

    logger.info(
        f"Backtest complete. "
        f"Strategy final: ${df['strategy_cum'].iloc[-1]:.3f} | "
        f"BuyHold final: ${df['buyhold_cum'].iloc[-1]:.3f}"
    )
    return df


def compute_strategy_metrics(
    df: pd.DataFrame,
    risk_free_rate: float = 0.05,
    trading_days: int = 252,
) -> Dict[str, float]:
    rf_daily = risk_free_rate / trading_days
    strat_r = df["strategy_return"].dropna()
    bh_r = df["buyhold_return"].dropna()

    def sharpe(returns):
        excess = returns - rf_daily
        if excess.std() < 1e-10:
            return 0.0
        return float((excess.mean() / excess.std()) * np.sqrt(trading_days))

    def max_drawdown(cum):
        roll_max = cum.cummax()
        dd = (cum - roll_max) / (roll_max + 1e-10)
        return float(dd.min())

    def annualized_return(cum, n_days):
        final = cum.iloc[-1]
        if final <= 0 or n_days <= 0:
            return 0.0
        return float(final ** (trading_days / n_days) - 1.0)

    n = len(strat_r)
    strat_ann_ret = annualized_return(df["strategy_cum"].dropna(), n)
    bh_ann_ret = annualized_return(df["buyhold_cum"].dropna(), n)
    strat_mdd = max_drawdown(df["strategy_cum"].dropna())
    bh_mdd = max_drawdown(df["buyhold_cum"].dropna())

    metrics = {
        "Strategy_Sharpe": sharpe(strat_r),
        "BuyHold_Sharpe": sharpe(bh_r),
        "Strategy_Annual_Return": strat_ann_ret,
        "BuyHold_Annual_Return": bh_ann_ret,
        "Strategy_MaxDrawdown": strat_mdd,
        "BuyHold_MaxDrawdown": bh_mdd,
        "Strategy_Calmar": (
            strat_ann_ret / (abs(strat_mdd) + 1e-10) if strat_mdd < 0 else 0.0
        ),
        "Total_Trades": int(df["position_change"].sum()),
        "Win_Rate": float((strat_r > 0).mean()),
    }
    logger.info(f"Strategy metrics: {metrics}")
    return metrics


def plot_backtest(
    df: pd.DataFrame,
    title: str = "Strategy vs Buy-and-Hold",
    save_path: str = "reports/figures/backtest.png",
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(df.index, df["strategy_cum"], label="Long/Short Strategy", linewidth=2)
    axes[0].plot(df.index, df["buyhold_cum"], label="Buy & Hold", linewidth=2, alpha=0.8)
    axes[0].axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[0].set_ylabel("Portfolio Value ($1 invested)")
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(df.index, df["strategy_return"], alpha=0.6, width=1, label="Daily Return")
    axes[1].axhline(y=0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Daily Return")
    axes[1].set_xlabel("Date")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Backtest plot saved -> {save_path}")
