"""Financial backtesting for the Layer 2 ML forecasting package.

Provides :class:`BacktestMixin`, the part of ``CryptoMLPredictor`` that turns
out-of-sample model predictions into Long/Flat trading decisions and evaluates
them: fixed-threshold metrics, in-sample threshold calibration and a
leakage-free walk-forward validation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import finance_metrics as fin


class BacktestMixin:
    """Strategy evaluation methods for ``CryptoMLPredictor``.

    Relies on attributes/methods of the host class (``oos_predictions``,
    ``tickers``, ``signal_threshold``, ``cost_bps``, ``periods_per_year`` and
    ``train``).
    """

    def evaluate_financial(self, ticker: str) -> dict[str, Any]:
        """Evaluate a Long/Flat strategy on out-of-sample predictions.

        Builds positions from the model's OOS predictions, applies transaction
        costs, and computes risk/return metrics for the strategy and a
        buy-and-hold benchmark over the same window, plus a textual verdict.

        Returns
        -------
        dict
            Keys: ``strategy`` and ``benchmark`` (metric dicts),
            ``directional_accuracy`` (float) and ``verdict`` (str).
        """
        if ticker not in self.oos_predictions:
            self.train(ticker)

        oos = self.oos_predictions[ticker]
        y_true = oos["y_true"]
        y_pred = oos["y_pred"]

        positions = fin.long_flat_positions(y_pred, self.signal_threshold)
        strat_returns = fin.strategy_returns(positions, y_true, self.cost_bps)

        # Buy-and-hold benchmark: always invested over the same OOS window.
        benchmark_returns = np.expm1(y_true.astype(float))

        strategy_summary = fin.summarize(strat_returns, self.periods_per_year)
        benchmark_summary = fin.summarize(benchmark_returns, self.periods_per_year)
        directional = fin.directional_accuracy(y_true, y_pred)

        verdict = fin.investment_verdict(
            strategy_summary, benchmark_summary, directional
        )
        return {
            "strategy": strategy_summary,
            "benchmark": benchmark_summary,
            "directional_accuracy": directional,
            "verdict": verdict,
        }

    def financial_summary_table(self) -> pd.DataFrame:
        """Tabulate strategy vs benchmark metrics for all trained tickers."""
        rows: list[dict[str, Any]] = []
        for ticker in self.tickers:
            if ticker not in self.oos_predictions:
                continue
            evaluation = self.evaluate_financial(ticker)
            for kind in ("strategy", "benchmark"):
                row: dict[str, Any] = {"ticker": ticker, "type": kind}
                row.update(evaluation[kind])
                row["dir_acc"] = (
                    evaluation["directional_accuracy"] if kind == "strategy" else np.nan
                )
                rows.append(row)
        return pd.DataFrame(rows)

    def optimize_threshold(
        self,
        ticker: str,
        objective: str = "sortino",
        max_drawdown_limit: float | None = None,
        n_grid: int = 25,
    ) -> dict[str, Any]:
        """Grid-search the Long/Flat entry threshold on out-of-sample data.

        Candidate thresholds are drawn from the quantiles of the strictly
        positive predicted returns (plus 0.0), so the search spans "trade on
        any positive signal" through "trade only on the strongest signals".

        Parameters
        ----------
        objective:
            Metric to maximise (``"sortino"`` by default; ``"sharpe"`` or
            ``"total_return"`` also valid). Sortino is preferred because it
            targets downside risk.
        max_drawdown_limit:
            Optional floor on max drawdown (a negative number, e.g. ``-0.5``).
            Thresholds breaching it are excluded unless none qualify.
        n_grid:
            Number of quantile points used to build the threshold grid.

        Returns
        -------
        dict
            ``best`` (metric dict for the chosen threshold), ``table`` (all
            candidates) and ``objective``.

        Notes
        -----
        Calibrating on the same OOS predictions used for evaluation is a mild
        form of in-sample tuning; a fully clean estimate would reserve a
        further holdout. Treat the result as an upper bound on achievable edge.
        """
        if ticker not in self.oos_predictions:
            self.train(ticker)

        oos = self.oos_predictions[ticker]
        y_true = oos["y_true"]
        y_pred = oos["y_pred"]

        candidates = fin.candidate_thresholds(y_pred, n_grid=n_grid)

        rows = [
            fin.evaluate_strategy(
                y_true, y_pred, threshold, self.cost_bps, self.periods_per_year
            )
            for threshold in candidates
        ]
        table = pd.DataFrame(rows)

        feasible = table
        if max_drawdown_limit is not None:
            within = table[table["max_drawdown"] >= max_drawdown_limit]
            if not within.empty:
                feasible = within

        feasible = feasible.dropna(subset=[objective])
        if feasible.empty:
            best = table.iloc[0]
        else:
            best = feasible.loc[feasible[objective].idxmax()]

        return {"best": best, "table": table, "objective": objective}

    def optimize_all_thresholds(
        self,
        objective: str = "sortino",
        max_drawdown_limit: float | None = None,
    ) -> pd.DataFrame:
        """Optimise the threshold for every ticker and tabulate the winners."""
        rows: list[dict[str, Any]] = []
        for ticker in self.tickers:
            result = self.optimize_threshold(
                ticker,
                objective=objective,
                max_drawdown_limit=max_drawdown_limit,
            )
            row: dict[str, Any] = {"ticker": ticker}
            row.update(result["best"].to_dict())
            rows.append(row)
        return pd.DataFrame(rows)

    def walk_forward_threshold(
        self,
        ticker: str,
        n_windows: int = 5,
        objective: str = "sortino",
        min_calibration: int = 252,
        max_drawdown_limit: float | None = None,
    ) -> dict[str, Any]:
        """Honest, leakage-free evaluation of threshold-based trading.

        Uses nested walk-forward validation on the out-of-sample predictions:
        the test span (everything after ``min_calibration`` observations) is
        divided into ``n_windows`` contiguous blocks. For each block the entry
        threshold is calibrated on an *expanding window of strictly earlier
        data only*, then applied to the block. Concatenating the per-block
        strategy returns yields an equity curve in which neither the model nor
        the threshold ever saw the evaluation data -- the scientifically sound
        way to test whether the edge survives out of sample.

        Returns
        -------
        dict
            ``strategy`` and ``benchmark`` metric dicts over the combined test
            span, ``thresholds`` (per-window choices), ``directional_accuracy``,
            ``verdict`` and the raw ``returns`` series.
        """
        if ticker not in self.oos_predictions:
            self.train(ticker)

        oos = self.oos_predictions[ticker]
        y_true_all = oos["y_true"]
        y_pred_all = oos["y_pred"]
        n_obs = len(oos)

        if n_obs < min_calibration + n_windows:
            raise ValueError(
                f"[{ticker}] Not enough OOS observations ({n_obs}) for "
                f"walk-forward with min_calibration={min_calibration} and "
                f"n_windows={n_windows}."
            )

        test_start = max(min_calibration, n_obs // (n_windows + 1))
        edges = np.linspace(test_start, n_obs, n_windows + 1).astype(int)

        net_parts: list[pd.Series] = []
        window_rows: list[dict[str, Any]] = []
        for window, (start, stop) in enumerate(zip(edges[:-1], edges[1:]), start=1):
            if stop <= start:
                continue

            # Calibrate ONLY on data strictly before the test block.
            calib_true = y_true_all.iloc[:start]
            calib_pred = y_pred_all.iloc[:start]
            threshold = fin.best_threshold(
                calib_true,
                calib_pred,
                self.cost_bps,
                self.periods_per_year,
                objective=objective,
                max_drawdown_limit=max_drawdown_limit,
            )

            test_true = y_true_all.iloc[start:stop]
            test_pred = y_pred_all.iloc[start:stop]
            positions = fin.long_flat_positions(test_pred, threshold)
            net = fin.strategy_returns(positions, test_true, self.cost_bps)
            net_parts.append(net)

            window_rows.append(
                {
                    "window": window,
                    "threshold": threshold,
                    "calib_obs": int(start),
                    "test_obs": int(stop - start),
                    "time_in_market": float((positions > 0).mean()),
                }
            )

        honest_returns = pd.concat(net_parts).sort_index()
        test_true_span = y_true_all.loc[honest_returns.index]
        test_pred_span = y_pred_all.loc[honest_returns.index]
        benchmark_returns = np.expm1(test_true_span.astype(float))

        strategy_summary = fin.summarize(honest_returns, self.periods_per_year)
        benchmark_summary = fin.summarize(benchmark_returns, self.periods_per_year)
        directional = fin.directional_accuracy(test_true_span, test_pred_span)
        verdict = fin.investment_verdict(
            strategy_summary, benchmark_summary, directional
        )

        return {
            "strategy": strategy_summary,
            "benchmark": benchmark_summary,
            "thresholds": pd.DataFrame(window_rows),
            "directional_accuracy": directional,
            "verdict": verdict,
            "returns": honest_returns,
        }
