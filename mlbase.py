"""Shared building blocks for the Layer 2 ML forecasting package.

This module holds constants, the :class:`FoldMetric` record and the optional
XGBoost import shim used across the feature, model, backtest and persistence
modules. Keeping them here avoids circular imports between those modules.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_TICKERS = ["BTC-USD", "ETH-USD"]

# Reference market whose state is injected as cross-asset context into every
# other ticker (e.g. BTC regime helps explain ETH moves).
BASE_MARKET = "BTC-USD"

# Name of the supervised target column (next-day log return).
TARGET_COLUMN = "target_next_log_return"

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - depends on local environment
    XGBRegressor = None  # type: ignore[assignment]


@dataclass
class FoldMetric:
    """Evaluation metrics for one time-series validation fold."""

    ticker: str
    fold: int
    model: str
    train_size: int
    test_size: int
    mae: float
    r2: float
