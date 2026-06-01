"""Financial evaluation metrics for the crypto forecasting model (Layer 2.5).

These are pure, dependency-light functions that turn model predictions and
realised returns into risk/return analytics for a Long/Flat trading strategy:

* Directional accuracy (does the model get the sign right?).
* Long/Flat position sizing with a configurable entry threshold.
* Net strategy returns including simple transaction costs (in basis points).
* Risk-adjusted performance: Sharpe and Sortino ratios.
* Downside risk: maximum drawdown, historical VaR and CVaR.
* A rule-based textual investment verdict comparing the strategy against a
  buy-and-hold benchmark.

All return inputs are expected to be *log* returns (as produced by the data
layer). They are converted to simple returns internally for compounding and
cost accounting. Nothing here constitutes financial advice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Convention: returns are expressed per period (e.g. daily). ``periods_per_year``
# annualises ratios; for 24/7 crypto daily data this is 365.


def _to_series(values: pd.Series | np.ndarray, index: pd.Index | None = None) -> pd.Series:
    """Coerce an array-like to a float Series, optionally with a given index."""
    if isinstance(values, pd.Series):
        return values.astype(float)
    array = np.asarray(values, dtype=float)
    return pd.Series(array, index=index)


def directional_accuracy(
    y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray
) -> float:
    """Fraction of observations where the predicted sign matches the realised
    sign. Periods with a zero realised return are ignored.

    Returns ``nan`` when there are no non-zero observations.
    """
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = true != 0.0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(pred[mask]) == np.sign(true[mask])))


def long_flat_positions(
    predicted_returns: pd.Series | np.ndarray, threshold: float = 0.0
) -> pd.Series:
    """Long/Flat position sizing: hold the asset (1.0) when the predicted
    return exceeds ``threshold``, otherwise stay in cash (0.0).

    No short selling, which bounds downside to the asset's own drawdown while
    invested and removes exposure during predicted declines.
    """
    series = _to_series(predicted_returns)
    return (series > threshold).astype(float)


def strategy_returns(
    positions: pd.Series,
    realised_log_returns: pd.Series,
    cost_bps: float = 10.0,
) -> pd.Series:
    """Compute net per-period simple returns of a Long/Flat strategy.

    Parameters
    ----------
    positions:
        Position taken for each period (``1.0`` long, ``0.0`` flat), aligned
        with ``realised_log_returns``.
    realised_log_returns:
        The realised next-period log return earned while holding the position.
    cost_bps:
        Round-trip-agnostic transaction cost in basis points, charged on the
        change in position (turnover). ``10`` bps = ``0.10%``.

    Returns
    -------
    pd.Series
        Net simple returns per period (gross position return minus costs).
    """
    pos = positions.astype(float)
    simple = np.expm1(realised_log_returns.astype(float))
    gross = pos * simple

    turnover = pos.diff().abs()
    if len(turnover) > 0:
        turnover.iloc[0] = abs(pos.iloc[0])  # cost of entering initial position
    cost = (cost_bps / 10_000.0) * turnover

    net = gross - cost
    net.name = "strategy_return"
    return net


def equity_curve(returns: pd.Series | np.ndarray) -> pd.Series:
    """Cumulative growth of 1 unit of capital from per-period simple returns."""
    series = _to_series(returns)
    return (1.0 + series).cumprod()


def cumulative_return(returns: pd.Series | np.ndarray) -> float:
    """Total compounded return over the whole period."""
    series = _to_series(returns)
    if series.empty:
        return float("nan")
    return float((1.0 + series).prod() - 1.0)


def annualized_volatility(
    returns: pd.Series | np.ndarray, periods_per_year: float
) -> float:
    """Annualised standard deviation of per-period returns."""
    series = _to_series(returns)
    if series.shape[0] < 2:
        return float("nan")
    return float(series.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series | np.ndarray,
    periods_per_year: float,
    risk_free: float = 0.0,
) -> float:
    """Annualised Sharpe ratio. ``risk_free`` is a per-period rate."""
    series = _to_series(returns) - risk_free
    std = series.std(ddof=1)
    if not np.isfinite(std) or std == 0.0:
        return float("nan")
    return float(series.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series | np.ndarray,
    periods_per_year: float,
    target: float = 0.0,
) -> float:
    """Annualised Sortino ratio (penalises only downside deviation)."""
    series = _to_series(returns) - target
    downside = series.clip(upper=0.0)
    downside_std = np.sqrt(np.mean(np.square(downside)))
    if not np.isfinite(downside_std) or downside_std == 0.0:
        return float("nan")
    return float(series.mean() / downside_std * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series | np.ndarray) -> float:
    """Maximum peak-to-trough decline of the equity curve (negative number)."""
    series = _to_series(returns)
    if series.empty:
        return float("nan")
    equity = (1.0 + series).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def value_at_risk(returns: pd.Series | np.ndarray, level: float = 0.05) -> float:
    """Historical Value at Risk: the per-period return at the ``level``
    quantile (typically negative, i.e. a loss)."""
    series = _to_series(returns)
    if series.empty:
        return float("nan")
    return float(np.quantile(series, level))


def conditional_var(returns: pd.Series | np.ndarray, level: float = 0.05) -> float:
    """Conditional VaR (Expected Shortfall): mean return in the worst
    ``level`` tail."""
    series = _to_series(returns)
    if series.empty:
        return float("nan")
    var = np.quantile(series, level)
    tail = series[series <= var]
    if tail.empty:
        return float(var)
    return float(tail.mean())


def win_rate(returns: pd.Series | np.ndarray) -> float:
    """Fraction of periods with a strictly positive return."""
    series = _to_series(returns)
    if series.empty:
        return float("nan")
    return float((series > 0.0).mean())


def summarize(returns: pd.Series | np.ndarray, periods_per_year: float) -> dict[str, float]:
    """Bundle the key risk/return metrics for a per-period return series."""
    return {
        "total_return": cumulative_return(returns),
        "ann_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "sortino": sortino_ratio(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "var_95": value_at_risk(returns, 0.05),
        "cvar_95": conditional_var(returns, 0.05),
        "win_rate": win_rate(returns),
    }


def candidate_thresholds(
    predicted_returns: pd.Series | np.ndarray, n_grid: int = 25
) -> list[float]:
    """Build a Long/Flat threshold grid from the quantiles of the strictly
    positive predicted returns (always including ``0.0``).

    Spans "enter on any positive signal" through "enter only on the strongest
    signals". Returns ``[0.0]`` when there are no positive predictions.
    """
    pred = _to_series(predicted_returns)
    positive = pred[pred > 0.0]
    if positive.empty:
        return [0.0]
    quantiles = np.linspace(0.0, 0.9, n_grid)
    return sorted({0.0, *np.quantile(positive, quantiles).tolist()})


def best_threshold(
    realised_log_returns: pd.Series,
    predicted_returns: pd.Series,
    cost_bps: float,
    periods_per_year: float,
    objective: str = "sortino",
    n_grid: int = 25,
    max_drawdown_limit: float | None = None,
) -> float:
    """Select the threshold maximising ``objective`` over a candidate grid.

    Used both for one-shot calibration and, crucially, inside walk-forward
    validation where it is only ever fed *past* data relative to the test
    window. Falls back to ``0.0`` when no candidate yields a finite objective.
    """
    candidates = candidate_thresholds(predicted_returns, n_grid)
    rows = [
        evaluate_strategy(
            realised_log_returns, predicted_returns, threshold,
            cost_bps, periods_per_year,
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
        return 0.0
    return float(feasible.loc[feasible[objective].idxmax(), "threshold"])


def evaluate_strategy(
    realised_log_returns: pd.Series,
    predicted_returns: pd.Series,
    threshold: float,
    cost_bps: float,
    periods_per_year: float,
) -> dict[str, float]:
    """Summarise a Long/Flat strategy for one entry ``threshold``.

    Combines position sizing, net returns (with costs) and the risk/return
    bundle, and adds activity diagnostics (``threshold``, ``n_trades`` and
    ``time_in_market``) useful for threshold calibration.
    """
    positions = long_flat_positions(predicted_returns, threshold)
    net = strategy_returns(positions, realised_log_returns, cost_bps)

    summary = summarize(net, periods_per_year)

    turnover = positions.diff().abs()
    if len(turnover) > 0:
        turnover.iloc[0] = abs(positions.iloc[0])
    summary["threshold"] = float(threshold)
    summary["n_trades"] = float(turnover.sum())
    summary["time_in_market"] = float((positions > 0).mean())
    return summary


def investment_verdict(
    strategy: dict[str, float],
    benchmark: dict[str, float],
    directional: float,
) -> str:
    """Produce a rule-based, human-readable verdict on whether the model-driven
    strategy looks worth pursuing versus buy-and-hold.

    The classification distinguishes three cases by comparing the strategy
    against the benchmark on total return, risk-adjusted ratios (Sharpe and
    Sortino) and maximum drawdown:

    * ``PELNA PRZEWAGA`` -- the strategy beats the benchmark on total return
      AND on at least one risk-adjusted ratio AND its drawdown is not worse.
    * ``DEFENSYWNA PRZEWAGA`` -- the strategy improves risk-adjusted ratios
      and/or drawdown but earns a lower total return (safer, less profit).
    * ``BRAK PRZEWAGI`` -- the strategy neither improves risk-adjusted metrics
      nor beats the benchmark return.

    The reasons list always reports the concrete numbers. Caveats about the
    lack of a directional edge (when accuracy is near 50%) and regime
    dependence are appended, along with the standard disclaimer. This is a
    heuristic summary, NOT financial advice.
    """
    reasons: list[str] = []

    def _finite(value: float) -> bool:
        return isinstance(value, (int, float)) and np.isfinite(value)

    def _strictly_better(key: str, higher_is_better: bool = True) -> bool:
        """Return ``True`` only when both values are finite and the strategy
        beats the benchmark in the desired direction."""
        s_val = strategy.get(key, float("nan"))
        b_val = benchmark.get(key, float("nan"))
        if not (_finite(s_val) and _finite(b_val)):
            return False
        return s_val > b_val if higher_is_better else s_val < b_val

    def _not_worse(key: str, higher_is_better: bool = True) -> bool:
        """Return ``True`` when the strategy is at least as good as the
        benchmark (or when a value is missing, so it is not held against it)."""
        s_val = strategy.get(key, float("nan"))
        b_val = benchmark.get(key, float("nan"))
        if not (_finite(s_val) and _finite(b_val)):
            return True
        return s_val >= b_val if higher_is_better else s_val <= b_val

    # Directional accuracy (informational; the empirical edge is ~50%).
    if _finite(directional):
        if directional > 0.52:
            reasons.append(f"trafnosc kierunkowa {directional:.1%} powyzej 50%")
        elif directional < 0.48:
            reasons.append(f"trafnosc kierunkowa {directional:.1%} ponizej 50%")
        else:
            reasons.append(
                f"trafnosc kierunkowa {directional:.1%} bliska losowej (50%)"
            )

    sharpe_s = strategy.get("sharpe", float("nan"))
    sharpe_b = benchmark.get("sharpe", float("nan"))
    reasons.append(
        f"Sharpe strategii {sharpe_s:.2f} vs buy&hold {sharpe_b:.2f}"
    )
    sortino_s = strategy.get("sortino", float("nan"))
    sortino_b = benchmark.get("sortino", float("nan"))
    reasons.append(
        f"Sortino strategii {sortino_s:.2f} vs buy&hold {sortino_b:.2f}"
    )

    # Max drawdown is negative; a larger (less negative) value is better.
    dd_strategy = strategy.get("max_drawdown", float("nan"))
    dd_benchmark = benchmark.get("max_drawdown", float("nan"))
    if _finite(dd_strategy) and _finite(dd_benchmark):
        if dd_strategy > dd_benchmark:
            reasons.append(
                f"plytszy max drawdown ({dd_strategy:.1%} vs {dd_benchmark:.1%})"
            )
        else:
            reasons.append(
                f"glebszy max drawdown ({dd_strategy:.1%} vs {dd_benchmark:.1%})"
            )

    ret_strategy = strategy.get("total_return", float("nan"))
    ret_benchmark = benchmark.get("total_return", float("nan"))
    if _finite(ret_strategy) and _finite(ret_benchmark):
        if ret_strategy > ret_benchmark:
            reasons.append(
                f"wyzszy zwrot netto ({ret_strategy:.1%} vs {ret_benchmark:.1%})"
            )
        else:
            reasons.append(
                f"nizszy zwrot netto ({ret_strategy:.1%} vs {ret_benchmark:.1%})"
            )

    # Classification.
    better_return = _strictly_better("total_return")
    better_risk_adjusted = _strictly_better("sharpe") or _strictly_better("sortino")
    drawdown_not_worse = _not_worse("max_drawdown")
    shallower_drawdown = _strictly_better("max_drawdown")
    defensive_edge = better_risk_adjusted or shallower_drawdown

    if better_return and better_risk_adjusted and drawdown_not_worse:
        headline = (
            "PELNA PRZEWAGA: strategia bije buy&hold zarowno na zwrocie, jak i "
            "na metrykach ryzyka."
        )
    elif defensive_edge and not better_return:
        headline = (
            "DEFENSYWNA PRZEWAGA: lepsze metryki ryzyka/plytszy drawdown, ale "
            "nizszy zwrot niz buy&hold (bezpieczniej, mniej zysku)."
        )
    else:
        headline = (
            "BRAK PRZEWAGI: brak poprawy metryk ryzyka i brak wyzszego zwrotu "
            "niz buy&hold."
        )

    detail = "; ".join(reasons) if reasons else "brak wystarczajacych danych"

    notes: list[str] = []
    if _finite(directional) and 0.48 <= directional <= 0.52:
        notes.append(
            "Trafnosc kierunkowa ~50% oznacza brak przewagi kierunkowej; "
            "ewentualna korzysc wynika z unikania ryzyka, nie z prognozy."
        )
    notes.append(
        "Wynik zalezny od rezimu rynku: w trwalej hossie strategia 'long/flat' "
        "z gotowka przez wiekszosc czasu moze mocno przegrac z buy&hold."
    )
    disclaimer = (
        "Uwaga: ocena heurystyczna na danych historycznych out-of-sample, "
        "nie stanowi porady inwestycyjnej."
    )
    notes.append(disclaimer)

    notes_text = "\n  ".join(notes)
    return f"{headline}\n  Przeslanki: {detail}.\n  {notes_text}"
