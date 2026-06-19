from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def monte_carlo_backtest(
    strategy_returns: pd.Series,
    n_simulations: int = 1000,
    n_trading_days: int = 252,
    random_seed: int = 42,
) -> Dict:
    np.random.seed(random_seed)
    returns_array = strategy_returns.dropna().values

    if len(returns_array) == 0:
        raise ValueError("strategy_returns is empty after dropping NaN values.")

    logger.info(
        f"Running Monte Carlo: {n_simulations} simulations x "
        f"{n_trading_days} trading days"
    )

    simulated_curves = np.zeros((n_simulations, n_trading_days))
    for i in range(n_simulations):
        sampled = np.random.choice(returns_array, size=n_trading_days, replace=True)
        simulated_curves[i] = np.cumprod(1.0 + sampled)

    final_values = simulated_curves[:, -1]

    results = {
        "simulated_curves": simulated_curves,
        "mean_final_value": float(np.mean(final_values)),
        "median_final_value": float(np.median(final_values)),
        "p5_final_value": float(np.percentile(final_values, 5)),
        "p95_final_value": float(np.percentile(final_values, 95)),
        "prob_profit": float((final_values > 1.0).mean()),
        "prob_loss_20pct": float((final_values < 0.80).mean()),
        "expected_sharpe": float(
            np.mean(returns_array) / (np.std(returns_array) + 1e-10) * np.sqrt(252)
        ),
    }

    logger.info(
        f"Monte Carlo complete. "
        f"Prob(profit)={results['prob_profit']:.1%} | "
        f"Median={results['median_final_value']:.3f}"
    )
    return results


def plot_monte_carlo(
    results: Dict,
    title: str = "Monte Carlo Simulation - Strategy Returns",
    save_path: str = "reports/figures/monte_carlo.png",
    max_curves_to_plot: int = 200,
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    curves = results["simulated_curves"]
    n_days = curves.shape[1]
    x_axis = np.arange(1, n_days + 1)

    fig, ax = plt.subplots(figsize=(14, 7))

    n_plot = min(max_curves_to_plot, len(curves))
    for i in range(n_plot):
        ax.plot(x_axis, curves[i], alpha=0.04, color="royalblue", linewidth=0.5)

    p5 = np.percentile(curves, 5, axis=0)
    p25 = np.percentile(curves, 25, axis=0)
    p50 = np.percentile(curves, 50, axis=0)
    p75 = np.percentile(curves, 75, axis=0)
    p95 = np.percentile(curves, 95, axis=0)

    ax.fill_between(x_axis, p5, p95, alpha=0.15, color="royalblue", label="5th-95th Pct")
    ax.fill_between(x_axis, p25, p75, alpha=0.25, color="royalblue", label="25th-75th Pct")
    ax.plot(x_axis, p50, color="red", linewidth=2.5, label="Median Path", zorder=5)
    ax.plot(x_axis, p5, color="orange", linewidth=1.5, linestyle="--", label="5th Pct")
    ax.plot(x_axis, p95, color="green", linewidth=1.5, linestyle="--", label="95th Pct")
    ax.axhline(y=1.0, color="black", linestyle="-", linewidth=1.0, alpha=0.6, label="Break Even")

    annotation = (
        f"Prob. Profit:   {results['prob_profit']:.1%}\n"
        f"Prob. Loss>20%: {results['prob_loss_20pct']:.1%}\n"
        f"Median Final:   {results['median_final_value']:.2f}x\n"
        f"5th Pct Final:  {results['p5_final_value']:.2f}x\n"
        f"95th Pct Final: {results['p95_final_value']:.2f}x\n"
        f"Expected Sharpe:{results['expected_sharpe']:.2f}"
    )
    ax.text(
        0.02, 0.97, annotation,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        fontsize=9,
        fontfamily="monospace",
    )

    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Portfolio Value ($1 invested)")
    ax.set_title(f"{title}\n({len(curves):,} Bootstrap Simulations)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Monte Carlo plot saved -> {save_path}")
