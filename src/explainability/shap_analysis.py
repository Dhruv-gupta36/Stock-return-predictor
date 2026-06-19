from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def generate_shap_analysis(
    model,
    X_test: pd.DataFrame,
    save_dir: str = "reports/figures",
    n_top_features: int = 10,
    max_display: int = 20,
) -> Tuple[np.ndarray, object]:
    try:
        import shap
    except ImportError:
        logger.error("shap package not installed. Run: pip install shap")
        raise

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Computing SHAP values for {len(X_test)} test samples...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, max_display=max_display, show=False)
    plt.title("SHAP Summary - Feature Impact on Predictions")
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=max_display, show=False)
    plt.title("Feature Importance - Mean |SHAP Value|")
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    try:
        expected_value = (
            explainer.expected_value
            if not isinstance(explainer.expected_value, list)
            else explainer.expected_value[0]
        )
        shap_exp = shap.Explanation(
            values=shap_values[0],
            base_values=expected_value,
            data=X_test.iloc[0].values,
            feature_names=X_test.columns.tolist(),
        )
        plt.figure(figsize=(12, 6))
        shap.waterfall_plot(shap_exp, max_display=15, show=False)
        plt.title("SHAP Waterfall - Single Prediction Breakdown")
        plt.tight_layout()
        plt.savefig(Path(save_dir) / "shap_waterfall.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        logger.warning(f"Waterfall plot failed: {exc}")

    importance_df = pd.DataFrame({
        "Feature": X_test.columns,
        "MeanAbsSHAP": np.abs(shap_values).mean(axis=0),
    }).sort_values("MeanAbsSHAP", ascending=False)

    for feat in importance_df["Feature"].head(3).tolist():
        try:
            plt.figure(figsize=(8, 5))
            shap.dependence_plot(feat, shap_values, X_test, show=False)
            plt.title(f"SHAP Dependence Plot - {feat}")
            plt.tight_layout()
            safe_name = feat.replace("/", "_").replace(" ", "_")
            plt.savefig(
                Path(save_dir) / f"shap_dep_{safe_name}.png",
                dpi=150, bbox_inches="tight",
            )
            plt.close()
        except Exception as exc:
            logger.warning(f"Dependence plot for {feat} failed: {exc}")

    logger.info("=== TOP SHAP FEATURES ===")
    for _, row in importance_df.head(n_top_features).iterrows():
        logger.info(f"  {row['Feature']:<35s} {row['MeanAbsSHAP']:.6f}")

    logger.info(f"SHAP analysis complete. Figures saved to {save_dir}")
    return shap_values, explainer
