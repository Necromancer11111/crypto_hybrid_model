from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

from data.data import CryptoDataFetcher

# arch >= 6.x exports ConvergenceWarning; older versions do not
try:
    from arch.utility.exceptions import ConvergenceWarning
except ImportError:
    ConvergenceWarning = UserWarning  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class GARCHResult:
    """Holds the fitted ARCH result and extracted features for one ticker."""

    ticker: str
    fit_result: ARCHModelResult
    conditional_volatility: pd.Series
    annualised_volatility: pd.Series
    converged: bool
    aic: float
    bic: float
    params: pd.Series = field(default_factory=pd.Series)

    def summary(self) -> str:
        status = "converged" if self.converged else "DID NOT CONVERGE"
        return (
            f"GARCH(1,1) [{self.ticker}] — {status} | "
            f"AIC={self.aic:.4f}  BIC={self.bic:.4f}\n"
            f"{self.params.to_string()}"
        )


# ---------------------------------------------------------------------------
# GARCH(1,1) model
# ---------------------------------------------------------------------------

class GARCHModel:
    """Fits a GARCH(1,1) model on log-returns from :class:`CryptoDataFetcher`
    and extracts conditional volatility as a feature ready for downstream ML.

    Parameters
    ----------
    fetcher:
        A *fetched* :class:`~data.data.CryptoDataFetcher` instance.  Call
        ``fetcher.fetch()`` before passing it here, or let :meth:`fit` call it
        automatically.
    p, q:
        GARCH lag orders.  Default is ``p=1, q=1``.
    mean:
        Mean model passed to ``arch_model``.  Common choices are
        ``"Constant"`` and ``"Zero"``.
    vol:
        Volatility process.  ``"GARCH"`` is the standard symmetric model.
    dist:
        Innovation distribution.  ``"normal"`` or ``"t"`` (Student-t, better
        for fat-tailed crypto returns).
    rescale:
        Whether to let ``arch`` rescale returns internally.  Strongly
        recommended for crypto where returns may be very small decimals.
    max_iter:
        Maximum solver iterations.  Increase if convergence warnings appear.
    trading_days:
        Used when annualising conditional volatility.  Crypto = 365.
    """

    def __init__(
        self,
        fetcher: CryptoDataFetcher,
        p: int = 1,
        q: int = 1,
        mean: str = "Constant",
        vol: str = "GARCH",
        dist: Literal["normal", "t", "skewt"] = "t",
        rescale: bool = True,
        max_iter: int = 500,
        trading_days: int = 365,
    ) -> None:
        self.fetcher = fetcher
        self.p = p
        self.q = q
        self.mean = mean
        self.vol = vol
        self.dist = dist
        self.rescale = rescale
        self.max_iter = max_iter
        self.trading_days = trading_days

        self._results: dict[str, GARCHResult] = {}
        self._scale: float = 100.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, scale: float = 100.0) -> dict[str, GARCHResult]:
        """Fit GARCH(1,1) for every ticker in the fetcher.

        Parameters
        ----------
        scale:
            Multiply log-returns by this factor before fitting.  Scaling to
            percentage returns (×100) is a standard practice that improves
            numerical stability and GARCH convergence.

        Returns
        -------
        dict[str, GARCHResult]
            Mapping ``{ticker: GARCHResult}``.  Also stored as
            ``self._results`` for later access.
        """
        self._scale = scale
        log_rets = self.fetcher.log_returns()

        for ticker in log_rets.columns:
            series = log_rets[ticker].dropna() * scale
            result = self._fit_single(ticker, series, scale)
            self._results[ticker] = result

        return self._results

    def conditional_volatility_features(
        self,
        annualise: bool = True,
        scale: float | None = None,
    ) -> pd.DataFrame:
        """Return a DataFrame of conditional volatility for all tickers.

        Columns are named ``<ticker>_cond_vol``.  This is the primary feature
        extracted for downstream ML models.

        Parameters
        ----------
        annualise:
            Convert daily conditional vol to annualised form by multiplying by
            ``sqrt(trading_days)``.
        scale:
            Divides the raw conditional volatility back to decimal form.  When
            ``None`` (default) the exact ``scale`` used during :meth:`fit` is
            reused automatically, which avoids silent unit mismatches.
        """
        if not self._results:
            raise RuntimeError("Call .fit() before extracting features.")

        scale = self._scale if scale is None else scale

        frames: list[pd.Series] = []
        for ticker, res in self._results.items():
            vol = res.conditional_volatility / scale
            if annualise:
                vol = vol * np.sqrt(self.trading_days)
            vol.name = f"{ticker}_cond_vol"
            frames.append(vol)

        return pd.concat(frames, axis=1)

    def get_result(self, ticker: str) -> GARCHResult:
        """Retrieve the :class:`GARCHResult` for a single ticker."""
        if ticker not in self._results:
            raise KeyError(
                f"No result for '{ticker}'. Available: {list(self._results)}"
            )
        return self._results[ticker]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_single(
        self, ticker: str, series: pd.Series, scale: float
    ) -> GARCHResult:
        """Fit GARCH(1,1) on a single return series with convergence handling."""
        am = arch_model(
            series,
            mean=self.mean,
            vol=self.vol,
            p=self.p,
            q=self.q,
            dist=self.dist,
            rescale=self.rescale,
        )

        converged = False
        fit_result: ARCHModelResult | None = None

        # --- First attempt: default SLSQP settings ------------------------
        # arch always uses scipy's SLSQP; the optimiser cannot be swapped.
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit_result = am.fit(
                    disp="off",
                    options={"maxiter": self.max_iter},
                )
            converged = self._is_converged(fit_result, caught)

        except Exception as exc:
            warnings.warn(
                f"[{ticker}] primary fit failed ({exc}). "
                "Retrying with relaxed tolerance and more iterations.",
                RuntimeWarning,
                stacklevel=2,
            )
            fit_result = None

        # --- Fallback: retry SLSQP with more iters + looser tolerance -----
        if fit_result is None or not converged:
            try:
                with warnings.catch_warnings(record=True) as caught2:
                    warnings.simplefilter("always")
                    fit_result = am.fit(
                        disp="off",
                        tol=1e-6,
                        options={"maxiter": self.max_iter * 5, "ftol": 1e-6},
                    )
                converged = self._is_converged(fit_result, caught2)
            except Exception as exc2:
                warnings.warn(
                    f"[{ticker}] retry also failed ({exc2}). "
                    "Results may be unreliable.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # arch raises if fit completely breaks; guard defensively
        if fit_result is None:
            raise RuntimeError(
                f"GARCH(1,1) could not be fitted for ticker '{ticker}'. "
                "Check that the return series has sufficient non-zero variance."
            )

        cond_vol = pd.Series(
            fit_result.conditional_volatility,
            index=series.index,
            name=ticker,
        )

        annualised_vol = cond_vol / scale * np.sqrt(self.trading_days)

        return GARCHResult(
            ticker=ticker,
            fit_result=fit_result,
            conditional_volatility=cond_vol,
            annualised_volatility=annualised_vol,
            converged=converged,
            aic=float(fit_result.aic),
            bic=float(fit_result.bic),
            params=fit_result.params,
        )

    @staticmethod
    def _is_converged(
        fit_result: ARCHModelResult, caught: list[warnings.WarningMessage]
    ) -> bool:
        """Determine convergence from scipy's optimiser flag, falling back to
        inspecting captured ConvergenceWarnings."""
        opt = getattr(fit_result, "optimization_result", None)
        if opt is not None and hasattr(opt, "success"):
            return bool(opt.success)

        # Fallback: no optimiser metadata available -> trust unless warned
        for w in caught:
            if issubclass(w.category, ConvergenceWarning) or (
                "convergence" in str(w.message).lower()
            ):
                return False
        return True


