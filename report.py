"""Console reporting / CLI entry point for the Layer 2 ML forecasting package.

Holds the dependency-free table formatters and the :func:`main` pipeline that
runs ``CryptoMLPredictor`` end to end and prints a readable report. Kept
separate from the model logic so the library stays presentation-free.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from mlbase import DEFAULT_TICKERS
from model import CryptoMLPredictor


def _format_metrics(metrics: pd.DataFrame) -> str:
    """Format metrics as a dependency-free plain text table."""
    if metrics.empty:
        return "No metrics available."

    formatted = metrics.copy()
    for column in ("mae", "r2"):
        formatted[column] = formatted[column].map(
            lambda value: f"{value:.6f}" if pd.notna(value) else ""
        )
    return formatted.to_string(index=False)


def _format_financial(metrics: pd.DataFrame) -> str:
    """Format the financial metrics table with percentages and ratios."""
    if metrics.empty:
        return "No financial metrics available."

    percent_cols = (
        "total_return",
        "ann_volatility",
        "max_drawdown",
        "var_95",
        "cvar_95",
        "win_rate",
        "dir_acc",
    )
    ratio_cols = ("sharpe", "sortino")

    formatted = metrics.copy()
    for column in percent_cols:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: f"{value:.2%}" if pd.notna(value) else ""
            )
    for column in ratio_cols:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: f"{value:.2f}" if pd.notna(value) else ""
            )
    return formatted.to_string(index=False)


def _format_threshold_table(table: pd.DataFrame) -> str:
    """Format the optimised-threshold summary with readable units."""
    if table.empty:
        return "No threshold optimisation available."

    percent_cols = (
        "threshold",
        "total_return",
        "ann_volatility",
        "max_drawdown",
        "var_95",
        "cvar_95",
        "win_rate",
        "time_in_market",
    )
    ratio_cols = ("sharpe", "sortino")

    formatted = table.copy()
    column_order = [
        "ticker",
        "threshold",
        "sortino",
        "sharpe",
        "total_return",
        "max_drawdown",
        "time_in_market",
        "n_trades",
    ]
    available = [c for c in column_order if c in formatted.columns]
    formatted = formatted[available]

    for column in percent_cols:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: f"{value:.3%}" if pd.notna(value) else ""
            )
    for column in ratio_cols:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: f"{value:.2f}" if pd.notna(value) else ""
            )
    if "n_trades" in formatted.columns:
        formatted["n_trades"] = formatted["n_trades"].map(
            lambda value: f"{value:.0f}" if pd.notna(value) else ""
        )
    return formatted.to_string(index=False)


def _walk_forward_table(wf_results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build a strategy-vs-benchmark table from walk-forward results."""
    rows: list[dict[str, Any]] = []
    for ticker, result in wf_results.items():
        for kind in ("strategy", "benchmark"):
            row: dict[str, Any] = {"ticker": ticker, "type": kind}
            row.update(result[kind])
            row["dir_acc"] = (
                result["directional_accuracy"] if kind == "strategy" else np.nan
            )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    """Run the full Layer 2 ML forecasting pipeline."""
    predictor = CryptoMLPredictor(
        tickers=DEFAULT_TICKERS,
        start="2020-01-01",
        end="2026-06-01",
        interval="1d",
        n_splits=5,
        cache_dir="data_cache",
    )
    predictor.train_all()

    print("=" * 72)
    print("Layer 2 ML Forecasting Metrics")
    print("=" * 72)
    print(_format_metrics(predictor.metrics_table()))
    print()

    print("=" * 72)
    print("Feature Importances")
    print("=" * 72)
    for ticker in predictor.tickers:
        print(f"\n{ticker}")
        print(predictor.feature_importances(ticker).to_string())

    print("\n" + "=" * 72)
    print("Financial Metrics: Long/Flat Strategy vs Buy & Hold (out-of-sample)")
    print("=" * 72)
    print(_format_financial(predictor.financial_summary_table()))
    print()

    print("=" * 72)
    print("Investment Verdict")
    print("=" * 72)
    for ticker in predictor.tickers:
        evaluation = predictor.evaluate_financial(ticker)
        print(f"\n{ticker}")
        print(evaluation["verdict"])

    print("\n" + "=" * 72)
    print("Calibrated Long/Flat Threshold (max Sortino, OOS)")
    print("=" * 72)
    optimized = predictor.optimize_all_thresholds(objective="sortino")
    print(_format_threshold_table(optimized))
    print(
        "\nNote: threshold is calibrated on the same out-of-sample predictions "
        "used for evaluation, so treat it as an optimistic upper bound."
    )

    print("\n" + "=" * 72)
    print("Walk-Forward Validated Strategy vs Buy & Hold (honest OOS)")
    print("=" * 72)
    wf_results = {
        ticker: predictor.walk_forward_threshold(ticker)
        for ticker in predictor.tickers
    }
    print(_format_financial(_walk_forward_table(wf_results)))
    print(
        "\nThreshold is calibrated only on data strictly preceding each test "
        "window, so these metrics are a leakage-free estimate of real edge."
    )
    for ticker, result in wf_results.items():
        print(f"\n{ticker} per-window thresholds:")
        print(result["thresholds"].to_string(index=False))
        print(f"{ticker} verdict:")
        print(result["verdict"])

    print("\n" + "=" * 72)
    print("Next-Day Expected Log Return Forecasts")
    print("=" * 72)
    for ticker in predictor.tickers:
        mu = predictor.predict_next_day_return(ticker)
        print(f"{ticker}: {mu:.6f}")

    try:
        predictor.save_models("models_artifacts")
        print("\nSaved trained models to 'models_artifacts/'.")
    except Exception as exc:  # pragma: no cover - persistence is best-effort
        warnings.warn(
            f"Model persistence demo failed ({exc!r}); pipeline unaffected.",
            RuntimeWarning,
            stacklevel=2,
        )

    print("\n" + "=" * 72)
    print("Optional Hyperparameter Tuning")
    print("=" * 72)
    print(
        "Tip: construct CryptoMLPredictor(tune=True) to enable "
        "time-series-aware hyperparameter search."
    )
    if False:  # Disabled by default: tuning is slow. Flip to True to demo.
        try:
            tuned_predictor = CryptoMLPredictor(tune=True, n_iter_search=20)
            tuned_predictor.train_all()
            print("Tuned hyperparameters per ticker:")
            for ticker, params in tuned_predictor.best_params.items():
                print(f"{ticker}: {params}")
        except Exception as exc:  # pragma: no cover - demonstration only
            warnings.warn(
                f"Hyperparameter tuning demo failed ({exc!r}); "
                "pipeline unaffected.",
                RuntimeWarning,
                stacklevel=2,
            )


if __name__ == "__main__":
    main()
