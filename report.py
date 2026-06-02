"""Full-stack CLI pipeline for the hybrid crypto model (Layers 0–3).

Runs data ingestion, statistical analysis (ADF + GARCH plots), ML forecasting,
financial backtesting, Monte Carlo risk simulation (with plots), and optional
model persistence. Console tables plus matplotlib figures are produced so the
project demonstrates every implemented layer in one command.
"""

from __future__ import annotations

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlbase import DEFAULT_TICKERS
from model import CryptoMLPredictor
from risk_management import CryptoMonteCarloSimulator, SimulationResult, suggest_allocation
from statistical_analysis import CryptoVolatilityModel


def _ticker_slug(ticker: str) -> str:
    """Return a filesystem-safe ticker symbol for plot filenames."""
    return ticker.replace("-", "_")


def _run_data_layer(predictor: CryptoMLPredictor) -> None:
    """Summarise Layer 0: downloaded market data and engineered series."""
    print("\n" + "=" * 72)
    print("Layer 0 — Data (CryptoDataFetcher)")
    print("=" * 72)
    print(f"  Tickers   : {', '.join(predictor.tickers)}")
    print(f"  Interval  : {predictor.interval}")
    print(f"  Range     : {predictor.start} → {predictor.end}")
    print(f"  Cache     : {predictor.cache_dir or 'disabled'}")

    if predictor.log_returns is None or predictor.close_prices is None:
        print("  Status    : data not loaded.")
        return

    n_rows = len(predictor.log_returns)
    idx = predictor.log_returns.index
    print(f"  Rows      : {n_rows} aligned observations")
    print(f"  Dates     : {idx.min()} → {idx.max()}")
    print(
        f"  Last close: "
        + ", ".join(
            f"{t}={predictor.close_prices[t].dropna().iloc[-1]:,.2f}"
            for t in predictor.tickers
            if t in predictor.close_prices.columns
        )
    )


def _run_statistical_layer(
    predictor: CryptoMLPredictor,
    *,
    show_plots: bool = True,
    save_plots: bool = True,
) -> None:
    """Run Layer 1: ADF stationarity checks and GARCH volatility charts."""
    print("\n" + "=" * 72)
    print("Layer 1 — Statistical Analysis (ADF + GARCH)")
    print("=" * 72)

    if predictor.log_returns is None:
        warnings.warn(
            "Log returns unavailable; skipping Layer 1.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    vol_model = CryptoVolatilityModel(
        predictor.log_returns,
        trading_days=predictor.periods_per_year,
    )

    for ticker in predictor.tickers:
        if ticker not in predictor.log_returns.columns:
            continue

        print(f"\n--- {ticker} ---")
        try:
            vol_model.check_stationarity(predictor.log_returns[ticker])
            garch_result = vol_model.fit_garch(
                predictor.log_returns[ticker], ticker=ticker
            )
            status = "converged" if garch_result.converged else "NOT converged"
            print(f"  GARCH(1,1) fit: {status}")
            latest_vol = garch_result.annualized_volatility.dropna().iloc[-1]
            print(f"  Latest annualised conditional vol: {latest_vol:.2%}")

            savepath = (
                f"garch_{_ticker_slug(ticker)}.png" if save_plots else None
            )
            fig = vol_model.plot_volatility(
                ticker,
                show=show_plots,
                savepath=savepath,
            )
            if not show_plots:
                plt.close(fig)
            if savepath:
                print(f"  Saved chart: {savepath}")
        except Exception as exc:  # pragma: no cover - plotting is best-effort
            warnings.warn(
                f"Layer 1 failed for {ticker} ({exc!r}); continuing.",
                RuntimeWarning,
                stacklevel=2,
            )


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


def _run_risk_layer(
    predictor: CryptoMLPredictor,
    horizon_days: int = 30,
    num_simulations: int = 10_000,
    *,
    show_plots: bool = False,
    save_plots: bool = True,
) -> None:
    """Drive the Layer 3 Monte Carlo risk engine from live model outputs.

    For each ticker the drift comes from the ML next-day forecast, the
    diffusion from the latest annualised GARCH conditional volatility, and the
    start price from the most recent close. Prints a VaR/CVaR report and a
    Sharpe/Kelly allocation suggestion.
    """
    print("\n" + "=" * 72)
    print(
        f"Layer 3 Monte Carlo Risk Simulation "
        f"({horizon_days}-day horizon, {num_simulations:,} sims)"
    )
    print("=" * 72)

    results: dict[str, SimulationResult] = {}
    for ticker in predictor.tickers:
        try:
            mu = predictor.predict_next_day_return(ticker)
            price = float(predictor.close_prices[ticker].dropna().iloc[-1])
            garch_column = f"{ticker}_GARCH_Vol"
            sigma = float(
                predictor.garch_features[garch_column].dropna().iloc[-1]
            )
        except (KeyError, IndexError, AttributeError) as exc:
            warnings.warn(
                f"Could not assemble risk inputs for {ticker} ({exc!r}); skipped.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        simulator = CryptoMonteCarloSimulator(
            current_price=price,
            expected_return=mu,
            volatility=sigma,
            ticker=ticker,
            random_state=42,
        )
        result = simulator.run_simulation(
            horizon_days=horizon_days, num_simulations=num_simulations
        )
        results[ticker] = result
        metrics = simulator.calculate_risk_metrics(result)
        print(
            f"\n{metrics.report()}\n"
            f"  Inputs: daily mu={mu:+.5f}, annual sigma={sigma:.2%}"
        )

        plot_path = (
            f"montecarlo_{_ticker_slug(ticker)}.png" if save_plots else None
        )
        try:
            fig = simulator.plot_simulation(
                result,
                metrics=metrics,
                show=show_plots,
                savepath=plot_path,
            )
            if not show_plots:
                plt.close(fig)
            if plot_path:
                print(f"  Saved chart: {plot_path}")
        except Exception as exc:  # pragma: no cover - plotting is best-effort
            warnings.warn(
                f"Monte Carlo plot failed for {ticker} ({exc!r}); "
                "metrics report unaffected.",
                RuntimeWarning,
                stacklevel=2,
            )

    if len(results) >= 2:
        print("\nSuggested allocation from simulated outcomes:")
        for method in ("sharpe", "kelly"):
            allocation = suggest_allocation(results, method=method)  # type: ignore[arg-type]
            split = ", ".join(f"{t}: {w:.1%}" for t, w in allocation.items())
            print(f"  {method.capitalize():7s}: {split}")


def main(*, show_plots: bool = False) -> None:
    """Run the complete hybrid model pipeline (Layers 0–3) with visualisations.

    Parameters
    ----------
    show_plots:
        When ``True``, open blocking matplotlib windows after each chart.
        Default is ``False``: charts are saved to PNG only so the full
        console report runs without waiting for window closes.
    """
    print("=" * 72)
    print("Crypto Hybrid Model — Full Pipeline")
    print("=" * 72)
    if show_plots:
        print("(Interactive charts enabled — close each window to continue.)")
    else:
        print("(Charts saved to PNG; use --show-plots to open interactive windows.)")

    predictor = CryptoMLPredictor(
        tickers=DEFAULT_TICKERS,
        start="2020-01-01",
        end="2026-06-01",
        interval="1d",
        n_splits=5,
        cache_dir="data_cache",
    )
    predictor.train_all()

    _run_data_layer(predictor)
    _run_statistical_layer(
        predictor, show_plots=show_plots, save_plots=True
    )

    print("\n" + "=" * 72)
    print("Layer 2 — Machine Learning (XGBoost / TimeSeriesSplit)")
    print("=" * 72)
    print()
    print("ML Forecasting Metrics")
    print("-" * 72)
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

    _run_risk_layer(predictor, show_plots=show_plots, save_plots=True)

    print("\n" + "=" * 72)
    print("Pipeline complete")
    print("=" * 72)
    print("  Charts: garch_BTC_USD.png, garch_ETH_USD.png,")
    print("          montecarlo_BTC_USD.png, montecarlo_ETH_USD.png")

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
    import argparse

    parser = argparse.ArgumentParser(description="Crypto hybrid model full pipeline")
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Open matplotlib windows (blocks until each chart is closed).",
    )
    args = parser.parse_args()
    main(show_plots=args.show_plots)
