"""Statistical Layer (Layer 1) of the hybrid crypto financial model.

This module implements :class:`CryptoVolatilityModel`, which performs advanced
statistical analysis on log-returns produced by the data layer:

* Augmented Dickey-Fuller (ADF) stationarity testing.
* GARCH(1,1) volatility modelling via the ``arch`` library.
* Extraction of annualised conditional volatility (crypto trades 24/7/365,
  so daily volatility is annualised with ``sqrt(365)``).
* Feature engineering: appending a ``{ticker}_GARCH_Vol`` column.
* Two-panel visualisation of returns and volatility clustering.

The model is deliberately decoupled from the data source: it consumes a
log-returns :class:`pandas.DataFrame`. The ``__main__`` block wires it to the
real data repository (``CryptoDataFetcher`` in :mod:`data.data`).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult
from matplotlib.ticker import PercentFormatter
from statsmodels.tsa.stattools import adfuller

# ``arch`` exports ConvergenceWarning on modern versions only.
try:
    from arch.utility.exceptions import ConvergenceWarning
except ImportError:  # pragma: no cover - depends on installed arch version
    ConvergenceWarning = UserWarning  # type: ignore[assignment, misc]


# Number of return periods per year for each yfinance interval. Crypto trades
# 24/7/365, so intraday factors use 24 hours x 365 days. Used to correctly
# annualise conditional volatility (annualised = daily/period vol x sqrt(N)).
PERIODS_PER_YEAR: dict[str, float] = {
    "1m": 365 * 24 * 60,
    "2m": 365 * 24 * 30,
    "5m": 365 * 24 * 12,
    "15m": 365 * 24 * 4,
    "30m": 365 * 24 * 2,
    "60m": 365 * 24,
    "1h": 365 * 24,
    "90m": 365 * 24 * 60 / 90,
    "1d": 365,
    "5d": 365 / 5,
    "1wk": 52,
    "1mo": 12,
    "3mo": 4,
}


def annualization_periods(interval: str) -> float:
    """Return the number of return periods per year for a yfinance interval.

    Parameters
    ----------
    interval:
        A yfinance data interval such as ``"1h"`` or ``"1d"``.

    Returns
    -------
    float
        Periods per year used to annualise volatility. Falls back to ``365``
        (daily) for unrecognised intervals, with a warning.
    """
    if interval not in PERIODS_PER_YEAR:
        warnings.warn(
            f"Unknown interval '{interval}'; defaulting annualisation to 365.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 365.0
    return float(PERIODS_PER_YEAR[interval])


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class StationarityResult:
    """Outcome of an Augmented Dickey-Fuller test."""

    adf_statistic: float
    p_value: float
    used_lag: int
    n_obs: int
    critical_values: dict[str, float]
    is_stationary: bool
    significance_level: float


@dataclass
class GARCHFitResult:
    """Container for a fitted GARCH(1,1) model and derived volatility."""

    ticker: str
    fit_result: ARCHModelResult
    daily_conditional_volatility: pd.Series
    annualized_volatility: pd.Series
    converged: bool


# ---------------------------------------------------------------------------
# Statistical volatility model
# ---------------------------------------------------------------------------


class CryptoVolatilityModel:
    """Performs stationarity testing and GARCH(1,1) volatility modelling on
    cryptocurrency log-returns.

    Parameters
    ----------
    log_returns:
        DataFrame of log-returns where each column is a ticker (e.g.
        ``BTC-USD``) and the index is a :class:`~pandas.DatetimeIndex`.
    significance_level:
        Significance level used by :meth:`check_stationarity`. Defaults to
        ``0.05`` (5%).
    trading_days:
        Number of return periods per year used for annualisation. For daily
        crypto data this is ``365``; for hourly data it is ``365 * 24 = 8760``.
        Use :func:`annualization_periods` to derive it from a yfinance
        interval.
    fit_scale:
        Factor applied to returns before fitting. Scaling decimal returns to
        percentage units (``×100``) markedly improves GARCH numerical
        stability and convergence. The conditional volatility is rescaled back
        to decimal units before annualisation.
    max_iter:
        Maximum optimiser iterations for the initial fit attempt.
    """

    def __init__(
        self,
        log_returns: pd.DataFrame,
        significance_level: float = 0.05,
        trading_days: float = 365.0,
        fit_scale: float = 100.0,
        max_iter: int = 500,
    ) -> None:
        if not isinstance(log_returns, pd.DataFrame):
            raise TypeError(
                "log_returns must be a pandas DataFrame, got "
                f"{type(log_returns).__name__}."
            )
        if log_returns.empty:
            raise ValueError("log_returns is empty; nothing to model.")

        self.log_returns: pd.DataFrame = log_returns
        self.significance_level = significance_level
        self.trading_days = trading_days
        self.fit_scale = fit_scale
        self.max_iter = max_iter

        self._garch_results: dict[str, GARCHFitResult] = {}

    # ------------------------------------------------------------------
    # 2. Stationarity check
    # ------------------------------------------------------------------

    def check_stationarity(self, series: pd.Series) -> StationarityResult:
        """Run the Augmented Dickey-Fuller test on ``series``.

        Prints the ADF statistic, p-value and a verdict, then returns a
        :class:`StationarityResult`. The null hypothesis of the ADF test is
        that a unit root is present (i.e. the series is non-stationary);
        rejecting it (``p < significance_level``) implies stationarity.

        Parameters
        ----------
        series:
            One-dimensional return (or price) series. NaNs are dropped.

        Returns
        -------
        StationarityResult
            Structured test output.
        """
        clean = series.dropna()
        if clean.empty:
            raise ValueError("Series contains no valid observations for ADF.")

        adf_stat, p_value, used_lag, n_obs, crit_values, _ = adfuller(
            clean, autolag="AIC"
        )
        is_stationary = p_value < self.significance_level

        name = series.name if series.name is not None else "series"
        verdict = "STATIONARY" if is_stationary else "NON-STATIONARY"
        print(f"--- ADF Stationarity Test: {name} ---")
        print(f"ADF Statistic : {adf_stat:.6f}")
        print(f"p-value       : {p_value:.6f}")
        print("Critical values:")
        for level, value in crit_values.items():
            print(f"    {level:>3}: {value:.6f}")
        print(
            f"Verdict       : series is {verdict} at the "
            f"{self.significance_level:.0%} significance level.\n"
        )

        return StationarityResult(
            adf_statistic=float(adf_stat),
            p_value=float(p_value),
            used_lag=int(used_lag),
            n_obs=int(n_obs),
            critical_values={k: float(v) for k, v in crit_values.items()},
            is_stationary=bool(is_stationary),
            significance_level=self.significance_level,
        )

    # ------------------------------------------------------------------
    # 3 & 4. GARCH modelling + volatility extraction
    # ------------------------------------------------------------------

    def fit_garch(self, series: pd.Series, ticker: Optional[str] = None) -> GARCHFitResult:
        """Fit a GARCH(1,1) model with a constant mean to ``series``.

        Uses ``Mean='Constant'``, ``Vol='GARCH'``, ``p=1``, ``q=1``. The daily
        conditional volatility is extracted and annualised by multiplying by
        ``sqrt(trading_days)`` (``sqrt(365)`` for crypto).

        Robustness: if SLSQP (arch's only optimiser) fails to converge, the
        fit is retried with more iterations and a looser tolerance. A
        non-converged result is still returned but flagged via
        :attr:`GARCHFitResult.converged`.

        Parameters
        ----------
        series:
            Log-return series for a single ticker.
        ticker:
            Optional label; defaults to ``series.name``.

        Returns
        -------
        GARCHFitResult
            Fitted model plus daily and annualised conditional volatility.
        """
        label = ticker or (str(series.name) if series.name is not None else "asset")
        clean = series.dropna()
        if clean.shape[0] < 50:
            raise ValueError(
                f"[{label}] Need at least 50 observations to fit GARCH(1,1); "
                f"got {clean.shape[0]}."
            )

        scaled = clean * self.fit_scale
        model = arch_model(
            scaled,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="t",
            rescale=False,
        )

        fit_result: ARCHModelResult | None = None
        converged = False

        # --- Initial fit ---------------------------------------------------
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit_result = model.fit(
                    disp="off", options={"maxiter": self.max_iter}
                )
            converged = self._is_converged(fit_result, caught)
        except Exception as exc:  # noqa: BLE001 - report and retry
            warnings.warn(
                f"[{label}] Initial GARCH fit failed ({exc}); retrying.",
                RuntimeWarning,
                stacklevel=2,
            )

        # --- Retry with relaxed settings if needed ------------------------
        if fit_result is None or not converged:
            try:
                with warnings.catch_warnings(record=True) as caught2:
                    warnings.simplefilter("always")
                    fit_result = model.fit(
                        disp="off",
                        tol=1e-6,
                        options={"maxiter": self.max_iter * 5, "ftol": 1e-6},
                    )
                converged = self._is_converged(fit_result, caught2)
                if not converged:
                    warnings.warn(
                        f"[{label}] GARCH(1,1) did not fully converge; "
                        "volatility estimates may be unreliable.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            except Exception as exc2:  # noqa: BLE001
                raise RuntimeError(
                    f"[{label}] GARCH(1,1) estimation failed: {exc2}"
                ) from exc2

        if fit_result is None:
            raise RuntimeError(f"[{label}] GARCH(1,1) could not be fitted.")

        # conditional_volatility is in the fitted (scaled) units -> rescale.
        daily_vol = pd.Series(
            np.asarray(fit_result.conditional_volatility) / self.fit_scale,
            index=clean.index,
            name=f"{label}_daily_vol",
        )
        annualized_vol = (daily_vol * np.sqrt(self.trading_days)).rename(
            f"{label}_GARCH_Vol"
        )

        result = GARCHFitResult(
            ticker=label,
            fit_result=fit_result,
            daily_conditional_volatility=daily_vol,
            annualized_volatility=annualized_vol,
            converged=converged,
        )
        self._garch_results[label] = result
        return result

    # ------------------------------------------------------------------
    # 5. Feature engineering
    # ------------------------------------------------------------------

    def build_volatility_features(self) -> pd.DataFrame:
        """Fit GARCH(1,1) for every ticker and append annualised volatility.

        Returns a copy of :attr:`log_returns` with one extra column per
        ticker named ``{ticker}_GARCH_Vol``, aligned on the original index.

        Returns
        -------
        pd.DataFrame
            Original log-returns plus appended GARCH volatility features.
        """
        enriched = self.log_returns.copy()
        for ticker in self.log_returns.columns:
            result = self._garch_results.get(ticker) or self.fit_garch(
                self.log_returns[ticker], ticker=str(ticker)
            )
            enriched[f"{ticker}_GARCH_Vol"] = result.annualized_volatility
        return enriched

    # ------------------------------------------------------------------
    # 6. Visualisation
    # ------------------------------------------------------------------

    def plot_volatility(
        self, ticker: str, show: bool = True
    ) -> plt.Figure:
        """Render a two-panel chart for ``ticker``.

        Top panel: log-returns (showing volatility clustering visually).
        Bottom panel: annualised GARCH conditional volatility.

        Parameters
        ----------
        ticker:
            Column name to plot. GARCH is fitted on demand if needed.
        show:
            If ``True``, call ``plt.show()`` before returning.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        if ticker not in self.log_returns.columns:
            raise KeyError(
                f"'{ticker}' not in log_returns columns: "
                f"{list(self.log_returns.columns)}"
            )

        result = self._garch_results.get(ticker) or self.fit_garch(
            self.log_returns[ticker], ticker=ticker
        )
        returns = self.log_returns[ticker]
        vol = result.annualized_volatility

        # Pick a clean style if available, else fall back to default.
        style = next(
            (s for s in ("seaborn-v0_8-darkgrid", "seaborn-darkgrid", "ggplot")
             if s in plt.style.available),
            "default",
        )

        with plt.style.context(style):
            fig, (ax_ret, ax_vol) = plt.subplots(
                2,
                1,
                figsize=(14, 8),
                sharex=True,
                gridspec_kw={"hspace": 0.18},
            )

            # --- Top panel: log returns -----------------------------------
            ax_ret.plot(
                returns.index,
                returns.values,
                color="#2c7fb8",
                linewidth=0.6,
                alpha=0.8,
            )
            ax_ret.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
            ax_ret.set_title("Log Returns", fontsize=12, loc="left")
            ax_ret.set_ylabel("Return")
            ax_ret.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            ax_ret.margins(x=0.01)

            # --- Bottom panel: annualised conditional volatility ----------
            ax_vol.plot(
                vol.index,
                vol.values,
                color="#d7301f",
                linewidth=1.3,
            )
            ax_vol.fill_between(
                vol.index, vol.values, color="#d7301f", alpha=0.12
            )
            status = "" if result.converged else "  [NOT CONVERGED]"
            ax_vol.set_title(
                f"GARCH(1,1) Annualised Conditional Volatility{status}",
                fontsize=12,
                loc="left",
            )
            ax_vol.set_ylabel("Annualised volatility")
            ax_vol.set_xlabel("Date")
            ax_vol.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            ax_vol.margins(x=0.01)

            # --- Shared x-axis: concise, auto-spaced date ticks -----------
            locator = mdates.AutoDateLocator()
            ax_vol.xaxis.set_major_locator(locator)
            ax_vol.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(locator)
            )

            fig.suptitle(
                f"{ticker} - Volatility Clustering",
                fontsize=15,
                fontweight="bold",
            )
            fig.tight_layout(rect=(0, 0, 1, 0.97))

        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_converged(
        fit_result: ARCHModelResult, caught: list[warnings.WarningMessage]
    ) -> bool:
        """Infer convergence from scipy's optimiser flag, falling back to
        inspecting captured ``ConvergenceWarning`` messages."""
        opt = getattr(fit_result, "optimization_result", None)
        if opt is not None and hasattr(opt, "success"):
            return bool(opt.success)
        for warning in caught:
            message = str(warning.message).lower()
            if issubclass(warning.category, ConvergenceWarning) or (
                "convergence" in message
            ):
                return False
        return True


# ---------------------------------------------------------------------------
# 7. Execution pipeline
# ---------------------------------------------------------------------------


def _load_log_returns(
    tickers: list[str], start: str, end: str, interval: str = "1d"
) -> pd.DataFrame:
    """Fetch prices and compute log-returns using the data layer.

    Falls back to a clear error if the data repository cannot be imported.

    Parameters
    ----------
    tickers:
        Yahoo Finance ticker symbols.
    start, end:
        Date range in ``YYYY-MM-DD`` format.
    interval:
        Sampling frequency (e.g. ``"1h"``). Note yfinance limits intraday
        history (hourly data is only available for roughly the last 730 days).
    """
    try:
        # Real class lives in data/data.py; alias to the requested name.
        from data.data import CryptoDataFetcher as CryptoDataRepository
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Could not import the data repository from 'data.data'. Run this "
            "script from the project root so the 'data' package is importable."
        ) from exc

    repository = CryptoDataRepository(
        tickers=tickers, start=start, end=end, interval=interval
    )
    repository.fetch()
    return repository.log_returns()


def main() -> None:
    """Run the full statistical pipeline for BTC-USD and ETH-USD."""
    tickers = ["BTC-USD", "ETH-USD"]

    # Recent, higher-frequency data: ~1 year of hourly bars ending mid-2026.
    # Hourly history stays within yfinance's ~730-day intraday limit.
    interval = "1h"
    start = "2025-06-01"
    end = "2026-06-01"

    log_returns = _load_log_returns(
        tickers=tickers, start=start, end=end, interval=interval
    )

    # Annualise using the correct number of periods per year for the interval
    # (hourly crypto -> 365 * 24 = 8760), not a fixed 365.
    model = CryptoVolatilityModel(
        log_returns, trading_days=annualization_periods(interval)
    )

    for ticker in tickers:
        if ticker not in log_returns.columns:
            warnings.warn(f"No data for {ticker}; skipping.", RuntimeWarning)
            continue

        print("=" * 60)
        print(f"Statistical analysis for {ticker}")
        print("=" * 60)

        model.check_stationarity(log_returns[ticker])

        result = model.fit_garch(log_returns[ticker], ticker=ticker)
        print(result.fit_result.summary())
        print()

    features = model.build_volatility_features()
    print("Feature matrix with appended GARCH volatility (tail):")
    print(features.tail())

    for ticker in tickers:
        if ticker in log_returns.columns:
            model.plot_volatility(ticker, show=True)


if __name__ == "__main__":
    main()
