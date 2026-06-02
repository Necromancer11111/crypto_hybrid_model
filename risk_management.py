"""Risk Management and Market Simulation (Layer 3) of the hybrid crypto model.

This module implements :class:`CryptoMonteCarloSimulator`, which fuses the
outputs of the previous layers into a forward-looking risk engine:

* **Drift (mu)** comes from the ML layer (:class:`CryptoMLPredictor` in
  :mod:`ml_forecasting`) -- the expected next-day/horizon log return.
* **Diffusion (sigma)** comes from the statistical layer
  (:class:`CryptoVolatilityModel` in :mod:`statistical_analysis`) -- the latest
  annualised GARCH conditional volatility.

Those parameters drive vectorised Geometric Brownian Motion (GBM) simulations

    S_t = S_0 * exp( (mu - sigma^2 / 2) * dt + sigma * sqrt(dt) * Z )

from which we derive tail-risk analytics (VaR / Expected Shortfall), a
Kelly/Sharpe based portfolio allocation suggestion and a two-panel
visualisation. Crypto trades 24/7/365, so ``dt = 1 / 365``.

Nothing here constitutes financial advice.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

TRADING_DAYS_PER_YEAR: int = 365

AllocationMethod = Literal["sharpe", "kelly"]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class SimulationResult:
    """Container for the output of a single GBM simulation run.

    Attributes
    ----------
    ticker:
        Symbol the simulation was run for (informational).
    initial_price:
        The starting price ``S_0`` shared by every path.
    horizon_days:
        Number of daily steps simulated forward.
    num_simulations:
        Number of independent Monte Carlo paths.
    annual_drift, annual_volatility:
        The annualised GBM parameters actually used (after any conversion).
    price_paths:
        Array of shape ``(num_simulations, horizon_days + 1)`` including the
        shared ``S_0`` at column 0.
    final_prices:
        Array of shape ``(num_simulations,)`` with the terminal price of each
        path (equivalent to ``price_paths[:, -1]``).
    """

    ticker: str
    initial_price: float
    horizon_days: int
    num_simulations: int
    annual_drift: float
    annual_volatility: float
    price_paths: np.ndarray = field(repr=False)
    final_prices: np.ndarray = field(repr=False)

    @property
    def horizon_returns(self) -> np.ndarray:
        """Simple (not log) return of each path over the full horizon."""
        return self.final_prices / self.initial_price - 1.0

    @property
    def time_grid(self) -> np.ndarray:
        """Integer day axis ``[0, 1, ..., horizon_days]`` for plotting."""
        return np.arange(self.horizon_days + 1)


@dataclass
class RiskMetrics:
    """Tail-risk analytics derived from the terminal return distribution.

    VaR and CVaR follow the same sign convention as :mod:`finance_metrics`:
    they are expressed as *signed horizon returns*, so a loss is negative
    (e.g. ``var_95 = -0.18`` means a 18% loss at the 95% confidence level).
    """

    ticker: str
    horizon_days: int
    initial_price: float
    expected_final_price: float
    median_final_price: float
    expected_return: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    prob_loss: float
    var_95_price: float
    var_99_price: float

    def report(self) -> str:
        """Return a human-readable multi-line VaR/CVaR report."""
        return (
            f"Risk report - {self.ticker} ({self.horizon_days}-day horizon)\n"
            f"  Start price            : {self.initial_price:,.2f}\n"
            f"  Expected final price   : {self.expected_final_price:,.2f}\n"
            f"  Median final price     : {self.median_final_price:,.2f}\n"
            f"  Expected horizon return: {self.expected_return:+.2%}\n"
            f"  Probability of loss    : {self.prob_loss:.2%}\n"
            f"  VaR  95%               : {self.var_95:+.2%} "
            f"(price {self.var_95_price:,.2f})\n"
            f"  VaR  99%               : {self.var_99:+.2%} "
            f"(price {self.var_99_price:,.2f})\n"
            f"  CVaR 95% (ES)          : {self.cvar_95:+.2%}\n"
            f"  CVaR 99% (ES)          : {self.cvar_99:+.2%}"
        )


# ---------------------------------------------------------------------------
# Monte Carlo simulator
# ---------------------------------------------------------------------------


class CryptoMonteCarloSimulator:
    """Geometric Brownian Motion simulator driven by ML and GARCH outputs.

    Parameters
    ----------
    current_price:
        Latest spot price ``S_0`` (must be strictly positive).
    expected_return:
        Drift parameter ``mu``. By default this is interpreted as a *per-day*
        log return (as produced by ``CryptoMLPredictor.predict_next_day_return``)
        and is annualised internally. Set ``return_is_annualized=True`` to pass
        an already-annualised drift.
    volatility:
        Diffusion parameter ``sigma``. By default interpreted as an *annualised*
        volatility (as produced by ``CryptoVolatilityModel``). Set
        ``volatility_is_annualized=False`` to pass a per-day volatility.
    ticker:
        Symbol label used in results and plots.
    trading_days:
        Periods per year for time-step and annualisation conversions. Crypto
        trades continuously, so the default is 365.
    return_is_annualized, volatility_is_annualized:
        Flags controlling the unit interpretation of ``expected_return`` and
        ``volatility`` (see above).
    random_state:
        Optional seed for the NumPy random generator (reproducible paths).

    Raises
    ------
    ValueError
        If ``current_price`` is not positive, ``volatility`` is negative, or
        ``trading_days`` is not positive.
    """

    def __init__(
        self,
        current_price: float,
        expected_return: float,
        volatility: float,
        ticker: str = "ASSET",
        trading_days: int = TRADING_DAYS_PER_YEAR,
        return_is_annualized: bool = False,
        volatility_is_annualized: bool = True,
        random_state: int | None = None,
    ) -> None:
        if not np.isfinite(current_price) or current_price <= 0.0:
            raise ValueError(
                f"current_price must be a positive finite number, got {current_price!r}."
            )
        if not np.isfinite(volatility) or volatility < 0.0:
            raise ValueError(
                f"volatility must be a non-negative finite number, got {volatility!r}."
            )
        if trading_days <= 0:
            raise ValueError(f"trading_days must be positive, got {trading_days!r}.")
        if not np.isfinite(expected_return):
            raise ValueError(
                f"expected_return must be finite, got {expected_return!r}."
            )

        self.current_price = float(current_price)
        self.ticker = ticker
        self.trading_days = int(trading_days)

        # Normalise both parameters to an annualised basis for the GBM SDE.
        self.annual_drift = (
            float(expected_return)
            if return_is_annualized
            else float(expected_return) * self.trading_days
        )
        self.annual_volatility = (
            float(volatility)
            if volatility_is_annualized
            else float(volatility) * np.sqrt(self.trading_days)
        )

        self._rng = np.random.default_rng(random_state)
        self._result: SimulationResult | None = None

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def run_simulation(
        self,
        horizon_days: int = 30,
        num_simulations: int = 10_000,
    ) -> SimulationResult:
        """Generate future price paths via vectorised Geometric Brownian Motion.

        Each daily increment of the log price is

            (mu - sigma^2 / 2) * dt + sigma * sqrt(dt) * Z,   dt = 1 / trading_days

        with ``Z`` standard normal. All paths are drawn at once as a
        ``(num_simulations, horizon_days)`` matrix and accumulated with a single
        ``cumsum`` -- no Python-level loop over simulations or steps.

        Parameters
        ----------
        horizon_days:
            Number of trading days to project forward (must be positive).
        num_simulations:
            Number of independent paths (must be positive).

        Returns
        -------
        SimulationResult
            The simulated paths and their terminal prices. Also cached on the
            instance for :meth:`calculate_risk_metrics` and :meth:`plot_simulation`.

        Raises
        ------
        ValueError
            If ``horizon_days`` or ``num_simulations`` is not positive.
        """
        if horizon_days <= 0:
            raise ValueError(f"horizon_days must be positive, got {horizon_days!r}.")
        if num_simulations <= 0:
            raise ValueError(
                f"num_simulations must be positive, got {num_simulations!r}."
            )

        dt = 1.0 / self.trading_days
        drift = (self.annual_drift - 0.5 * self.annual_volatility**2) * dt
        diffusion_scale = self.annual_volatility * np.sqrt(dt)

        # Vectorised: one Gaussian shock per (simulation, day).
        shocks = self._rng.standard_normal(size=(num_simulations, horizon_days))
        log_increments = drift + diffusion_scale * shocks
        cumulative_log = np.cumsum(log_increments, axis=1)

        price_paths = np.empty((num_simulations, horizon_days + 1), dtype=float)
        price_paths[:, 0] = self.current_price
        price_paths[:, 1:] = self.current_price * np.exp(cumulative_log)

        result = SimulationResult(
            ticker=self.ticker,
            initial_price=self.current_price,
            horizon_days=horizon_days,
            num_simulations=num_simulations,
            annual_drift=self.annual_drift,
            annual_volatility=self.annual_volatility,
            price_paths=price_paths,
            final_prices=price_paths[:, -1],
        )
        self._result = result
        return result

    # ------------------------------------------------------------------
    # Risk analytics
    # ------------------------------------------------------------------

    def calculate_risk_metrics(
        self,
        result: SimulationResult | None = None,
    ) -> RiskMetrics:
        """Analyse the terminal return distribution for VaR and CVaR.

        Computes 95%/99% Value at Risk (the loss quantile) and the matching
        Expected Shortfall / Conditional VaR (mean loss in the tail beyond VaR).

        Parameters
        ----------
        result:
            A specific :class:`SimulationResult` to analyse. Defaults to the
            most recent run cached on the instance.

        Returns
        -------
        RiskMetrics

        Raises
        ------
        RuntimeError
            If no simulation result is available (call :meth:`run_simulation`).
        """
        result = self._require_result(result)

        horizon_returns = result.horizon_returns
        final_prices = result.final_prices

        var_95 = float(np.quantile(horizon_returns, 0.05))
        var_99 = float(np.quantile(horizon_returns, 0.01))
        cvar_95 = self._expected_shortfall(horizon_returns, var_95)
        cvar_99 = self._expected_shortfall(horizon_returns, var_99)

        return RiskMetrics(
            ticker=result.ticker,
            horizon_days=result.horizon_days,
            initial_price=result.initial_price,
            expected_final_price=float(np.mean(final_prices)),
            median_final_price=float(np.median(final_prices)),
            expected_return=float(np.mean(horizon_returns)),
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            prob_loss=float(np.mean(final_prices < result.initial_price)),
            var_95_price=result.initial_price * (1.0 + var_95),
            var_99_price=result.initial_price * (1.0 + var_99),
        )

    @staticmethod
    def _expected_shortfall(returns: np.ndarray, var_threshold: float) -> float:
        """Mean of the returns at or below ``var_threshold`` (the tail)."""
        tail = returns[returns <= var_threshold]
        if tail.size == 0:
            return float(var_threshold)
        return float(tail.mean())

    def _require_result(
        self, result: SimulationResult | None
    ) -> SimulationResult:
        """Return ``result`` or the cached one, raising if neither exists."""
        chosen = result if result is not None else self._result
        if chosen is None:
            raise RuntimeError(
                "No simulation available; call run_simulation() first."
            )
        return chosen

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_simulation(
        self,
        result: SimulationResult | None = None,
        num_paths_to_plot: int = 100,
        metrics: RiskMetrics | None = None,
        show: bool = True,
        savepath: str | None = None,
    ) -> "plt.Figure":
        """Draw a two-panel summary of a simulation.

        Left panel: a sample of up to ``num_paths_to_plot`` price paths with the
        median path highlighted. Right panel: a histogram of terminal prices
        annotated with the 95% and 99% VaR thresholds.

        Parameters
        ----------
        result:
            Simulation to plot (defaults to the latest cached run).
        num_paths_to_plot:
            Maximum number of individual paths to draw on the left panel.
        metrics:
            Pre-computed metrics; recomputed from ``result`` when omitted.
        show:
            Whether to call :func:`matplotlib.pyplot.show`.
        savepath:
            Optional path to save the figure as an image.

        Returns
        -------
        matplotlib.figure.Figure
        """
        result = self._require_result(result)
        if metrics is None:
            metrics = self.calculate_risk_metrics(result)

        try:
            plt.style.use("seaborn-v0_8-darkgrid")
        except OSError:  # pragma: no cover - style availability varies
            pass

        fig, (ax_paths, ax_hist) = plt.subplots(
            1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1.4, 1.0]}
        )

        # --- Left: sampled price paths + median path ---
        sample_size = min(num_paths_to_plot, result.num_simulations)
        sample_idx = self._rng.choice(
            result.num_simulations, size=sample_size, replace=False
        )
        time_grid = result.time_grid
        ax_paths.plot(
            time_grid,
            result.price_paths[sample_idx].T,
            color="steelblue",
            alpha=0.18,
            linewidth=0.8,
        )
        median_path = np.median(result.price_paths, axis=0)
        ax_paths.plot(
            time_grid, median_path, color="crimson", linewidth=2.4,
            label="Median path",
        )
        ax_paths.axhline(
            result.initial_price, color="black", linestyle="--", linewidth=1.0,
            label=f"Start {result.initial_price:,.0f}",
        )
        ax_paths.set_title(
            f"{result.ticker}: {sample_size} of {result.num_simulations:,} GBM paths"
        )
        ax_paths.set_xlabel("Day")
        ax_paths.set_ylabel("Simulated price")
        ax_paths.legend(loc="upper left")

        # --- Right: terminal price distribution with VaR lines ---
        ax_hist.hist(
            result.final_prices, bins=60, color="steelblue", alpha=0.75,
            orientation="vertical",
        )
        ax_hist.axvline(
            metrics.var_95_price, color="darkorange", linestyle="--", linewidth=2.0,
            label=f"VaR 95% ({metrics.var_95:+.1%})",
        )
        ax_hist.axvline(
            metrics.var_99_price, color="red", linestyle="--", linewidth=2.0,
            label=f"VaR 99% ({metrics.var_99:+.1%})",
        )
        ax_hist.axvline(
            result.initial_price, color="black", linestyle="-", linewidth=1.2,
            label=f"Start {result.initial_price:,.0f}",
        )
        ax_hist.set_title(f"{result.ticker}: day-{result.horizon_days} price distribution")
        ax_hist.set_xlabel("Terminal price")
        ax_hist.set_ylabel("Frequency")
        ax_hist.legend(loc="upper right")

        fig.tight_layout()
        if savepath is not None:
            fig.savefig(savepath, dpi=120, bbox_inches="tight")
        if show:
            plt.show()
        return fig


# ---------------------------------------------------------------------------
# Portfolio allocation (bonus)
# ---------------------------------------------------------------------------


def suggest_allocation(
    results: dict[str, SimulationResult],
    method: AllocationMethod = "sharpe",
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Suggest a long-only allocation split across assets from simulations.

    Two scoring rules are supported:

    * ``"sharpe"`` -- score = (mean horizon return - rf) / std of horizon return.
    * ``"kelly"``  -- score = (mean horizon return - rf) / variance of horizon
      return (the continuous Kelly fraction for a mean-variance bet).

    Assets with a positive score receive weight proportional to that score
    (normalised to sum to 1). If no asset scores positively, the function falls
    back to inverse-volatility (risk-parity) weights so it always returns a
    valid split. The returned dict maps each ticker to a weight in ``[0, 1]``.

    Parameters
    ----------
    results:
        Mapping of ticker to its :class:`SimulationResult`.
    method:
        ``"sharpe"`` (default) or ``"kelly"``.
    risk_free_rate:
        Per-horizon risk-free return subtracted from the mean (default 0).

    Raises
    ------
    ValueError
        If ``results`` is empty or ``method`` is not recognised.
    """
    if not results:
        raise ValueError("results must contain at least one asset.")
    if method not in ("sharpe", "kelly"):
        raise ValueError(f"Unknown method {method!r}; use 'sharpe' or 'kelly'.")

    tickers = list(results)
    means = np.array(
        [float(np.mean(results[t].horizon_returns)) for t in tickers]
    )
    stds = np.array(
        [float(np.std(results[t].horizon_returns, ddof=1)) for t in tickers]
    )
    # Guard against zero-volatility assets (degenerate simulation).
    safe_stds = np.where(stds > 0.0, stds, np.nan)
    excess = means - risk_free_rate

    if method == "sharpe":
        scores = excess / safe_stds
    else:  # kelly
        scores = excess / safe_stds**2
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    positive = np.clip(scores, a_min=0.0, a_max=None)
    if positive.sum() > 0.0:
        weights = positive / positive.sum()
    else:
        # No attractive asset: fall back to inverse-volatility risk parity.
        inv_vol = np.where(stds > 0.0, 1.0 / stds, 0.0)
        weights = (
            inv_vol / inv_vol.sum()
            if inv_vol.sum() > 0.0
            else np.full(len(tickers), 1.0 / len(tickers))
        )

    return {ticker: float(weight) for ticker, weight in zip(tickers, weights)}


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------


def main() -> None:
    """Run a mocked Layer 3 demonstration for BTC-USD and ETH-USD.

    The ML-layer drift and GARCH-layer volatility are mocked here so the module
    is runnable standalone; in production these come from
    ``CryptoMLPredictor.predict_next_day_return`` and
    ``CryptoVolatilityModel`` respectively.
    """
    # Mocked inputs: (current price, ML daily log-return drift, GARCH annual vol).
    mock_profiles: dict[str, dict[str, float]] = {
        "BTC-USD": {"price": 68_000.0, "daily_mu": 0.000_5, "annual_sigma": 0.55},
        "ETH-USD": {"price": 3_500.0, "daily_mu": -0.000_2, "annual_sigma": 0.75},
    }

    horizon_days = 30
    num_simulations = 10_000

    results: dict[str, SimulationResult] = {}
    print("=" * 72)
    print(f"Layer 3 Monte Carlo Risk Report ({horizon_days}-day horizon, "
          f"{num_simulations:,} sims)")
    print("=" * 72)

    for ticker, profile in mock_profiles.items():
        simulator = CryptoMonteCarloSimulator(
            current_price=profile["price"],
            expected_return=profile["daily_mu"],
            volatility=profile["annual_sigma"],
            ticker=ticker,
            random_state=42,
        )
        result = simulator.run_simulation(
            horizon_days=horizon_days, num_simulations=num_simulations
        )
        results[ticker] = result

        metrics = simulator.calculate_risk_metrics(result)
        print(f"\n{metrics.report()}")

        try:
            simulator.plot_simulation(
                result,
                metrics=metrics,
                show=False,
                savepath=f"montecarlo_{ticker.replace('-', '_')}.png",
            )
            print(f"  Saved plot to montecarlo_{ticker.replace('-', '_')}.png")
        except Exception as exc:  # pragma: no cover - plotting is best-effort
            warnings.warn(
                f"Plotting failed for {ticker} ({exc!r}); report unaffected.",
                RuntimeWarning,
                stacklevel=2,
            )

    print("\n" + "=" * 72)
    print("Suggested Portfolio Allocation (from simulated outcomes)")
    print("=" * 72)
    for method in ("sharpe", "kelly"):
        allocation = suggest_allocation(results, method=method)  # type: ignore[arg-type]
        split = ", ".join(f"{t}: {w:.1%}" for t, w in allocation.items())
        print(f"  {method.capitalize():7s}: {split}")


if __name__ == "__main__":
    main()
