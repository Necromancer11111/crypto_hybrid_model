"""Feature engineering for the Layer 2 ML forecasting package.

Provides :class:`FeatureMixin`, the part of ``CryptoMLPredictor`` responsible
for fetching market data, building leakage-free per-ticker feature/target
matrices (returns, lags, trend, RSI, volatility and cross-asset context) and
injecting the Layer 1 GARCH conditional volatility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.data import CryptoDataFetcher
from mlbase import BASE_MARKET, TARGET_COLUMN
from statistical_analysis import CryptoVolatilityModel, annualization_periods


class FeatureMixin:
    """Feature/target construction methods for ``CryptoMLPredictor``.

    Relies on attributes defined by the host class' ``__init__`` (``tickers``,
    ``start``, ``end``, ``interval``, ``cache_dir``, ``use_cache``,
    ``periods_per_year`` and the feature/data caches).
    """

    def prepare_features(self) -> dict[str, pd.DataFrame]:
        """Fetch daily market data and build per-ticker feature/target datasets.

        All features use only information available at the current timestamp
        (no look-ahead). The engineered feature set per ticker includes:

        * ``log_return`` and lagged returns (1, 2, 3, 5, 7 days) for short-term
          autocorrelation and momentum.
        * Stationary trend features ``price_to_sma_7``, ``price_to_sma_21`` and
          ``sma_7_to_21``. Raw price-level SMAs are intentionally avoided
          because tree models cannot extrapolate to unseen price ranges.
        * ``rsi_14`` momentum oscillator.
        * Realized volatility ``rolling_vol_7`` and ``rolling_vol_21`` plus the
          Layer 1 annualized GARCH conditional volatility ``garch_vol``.
        * Liquidity features ``volume_change`` and ``volume_ratio_21``.
        * For non-base tickers, cross-asset context from the base market
          (``btc_return``, ``btc_return_lag_1``, ``btc_garch_vol``).

        The target is the next day's log return (``shift(-1)``). NaNs created by
        lags/rolling windows/shift are dropped later, before training.
        """
        fetcher = CryptoDataFetcher(
            tickers=self.tickers,
            start=self.start,
            end=self.end,
            interval=self.interval,
            cache_dir=self.cache_dir,
            use_cache=self.use_cache,
        )
        fetcher.fetch()

        self.close_prices = self._normalize_columns(fetcher.close_prices())
        self.log_returns = self._normalize_columns(fetcher.log_returns())
        self.volume_data = self._normalize_columns(fetcher.volumes())
        self.garch_features = self._build_garch_features(self.log_returns)

        base = BASE_MARKET if BASE_MARKET in self.tickers else None

        datasets: dict[str, pd.DataFrame] = {}
        feature_columns: dict[str, list[str]] = {}
        for ticker in self.tickers:
            if ticker not in self.close_prices.columns:
                raise KeyError(f"Missing close prices for ticker '{ticker}'.")
            if ticker not in self.log_returns.columns:
                raise KeyError(f"Missing log returns for ticker '{ticker}'.")

            garch_column = f"{ticker}_GARCH_Vol"
            if garch_column not in self.garch_features.columns:
                raise KeyError(f"Missing GARCH feature column '{garch_column}'.")

            frame = self._build_feature_frame(ticker, base)
            frame = frame.replace([np.inf, -np.inf], np.nan)

            datasets[ticker] = frame
            feature_columns[ticker] = [
                column for column in frame.columns if column != TARGET_COLUMN
            ]

        self.feature_data = datasets
        self.feature_columns = feature_columns
        return datasets

    def _build_feature_frame(self, ticker: str, base: str | None) -> pd.DataFrame:
        """Construct the leakage-free feature/target matrix for one ticker."""
        close = self.close_prices[ticker]  # type: ignore[index]
        returns = self.log_returns[ticker]  # type: ignore[index]
        garch = self.garch_features[f"{ticker}_GARCH_Vol"]  # type: ignore[index]

        sma_7 = close.rolling(window=7, min_periods=7).mean()
        sma_21 = close.rolling(window=21, min_periods=21).mean()

        frame = pd.DataFrame(index=returns.index)

        # Returns and short-term lags.
        frame["log_return"] = returns
        for lag in (1, 2, 3, 5, 7):
            frame[f"return_lag_{lag}"] = returns.shift(lag)

        # Stationary trend / momentum features (ratios, not price levels).
        frame["price_to_sma_7"] = close / sma_7 - 1.0
        frame["price_to_sma_21"] = close / sma_21 - 1.0
        frame["sma_7_to_21"] = sma_7 / sma_21 - 1.0
        frame["rsi_14"] = self.compute_rsi(close, period=14)

        # Volatility: realized (rolling) + Layer 1 GARCH conditional vol.
        annualizer = float(np.sqrt(self.periods_per_year))
        frame["rolling_vol_7"] = (
            returns.rolling(window=7, min_periods=7).std() * annualizer
        )
        frame["rolling_vol_21"] = (
            returns.rolling(window=21, min_periods=21).std() * annualizer
        )
        frame["garch_vol"] = garch

        # Liquidity / activity features from volume, when available.
        if self.volume_data is not None and ticker in self.volume_data.columns:
            volume = self.volume_data[ticker]
            frame["volume_change"] = volume.pct_change(fill_method=None)
            volume_sma_21 = volume.rolling(window=21, min_periods=21).mean()
            frame["volume_ratio_21"] = volume / volume_sma_21 - 1.0

        # Cross-asset context: base-market (BTC) state for other tickers.
        if base is not None and ticker != base:
            base_returns = self.log_returns[base]  # type: ignore[index]
            base_garch = self.garch_features[f"{base}_GARCH_Vol"]
            frame["btc_return"] = base_returns.reindex(frame.index)
            frame["btc_return_lag_1"] = (
                base_returns.shift(1).reindex(frame.index)
            )
            frame["btc_garch_vol"] = base_garch.reindex(frame.index)

        # Supervised target: next-day log return (shift -1 -> no leakage).
        frame[TARGET_COLUMN] = returns.shift(-1)
        return frame

    @staticmethod
    def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Compute standard RSI from close prices using present/past values.

        The default is a 14-day RSI. Rolling average gains and losses are based
        only on observations up to the current row, so the indicator does not
        leak future information.
        """
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        relative_strength = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + relative_strength))

    def _build_garch_features(self, log_returns: pd.DataFrame) -> pd.DataFrame:
        """Run Layer 1 GARCH feature engineering with daily annualization."""
        volatility_model = CryptoVolatilityModel(
            log_returns,
            trading_days=annualization_periods(self.interval),
        )
        return volatility_model.build_volatility_features()

    def _labeled_dataset(self, ticker: str) -> pd.DataFrame:
        """Return a clean feature matrix with target for model training."""
        if ticker not in self.feature_data:
            raise KeyError(f"[{ticker}] Feature data has not been prepared.")
        return self.feature_data[ticker].dropna(
            subset=[*self.feature_columns[ticker], TARGET_COLUMN]
        )

    def _normalize_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Ensure fetched price/return columns are named by ticker."""
        normalized = frame.copy()
        if len(self.tickers) == 1 and list(normalized.columns) != self.tickers:
            normalized.columns = self.tickers
        return normalized
