"""Machine Learning Layer (Layer 2) for crypto return forecasting.

This is a thin backward-compatible facade. The implementation was split into
focused modules for readability:

* :mod:`mlbase` -- shared constants and the ``FoldMetric`` record.
* :mod:`features` -- feature engineering (``FeatureMixin``).
* :mod:`backtest` -- Long/Flat strategy evaluation (``BacktestMixin``).
* :mod:`persistence` -- model save/load (``PersistenceMixin``).
* :mod:`model` -- the ``CryptoMLPredictor`` estimator.
* :mod:`report` -- console formatters and the ``main`` pipeline.

Importing from ``ml_forecasting`` (e.g. ``from ml_forecasting import
CryptoMLPredictor``) and running ``python ml_forecasting.py`` both keep working
exactly as before.
"""

from __future__ import annotations

from mlbase import BASE_MARKET, DEFAULT_TICKERS, TARGET_COLUMN, FoldMetric
from model import CryptoMLPredictor
from report import (
    _format_financial,
    _format_metrics,
    _format_threshold_table,
    _walk_forward_table,
    main,
)

__all__ = [
    "BASE_MARKET",
    "DEFAULT_TICKERS",
    "TARGET_COLUMN",
    "FoldMetric",
    "CryptoMLPredictor",
    "main",
    "_format_financial",
    "_format_metrics",
    "_format_threshold_table",
    "_walk_forward_table",
]


if __name__ == "__main__":
    main()
