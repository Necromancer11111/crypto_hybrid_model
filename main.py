"""Top-level entry point for the Crypto Hybrid Model project.

Running this file executes the full three-layer pipeline:
  Layer 0 -- data acquisition and log-return computation
  Layer 1 -- GARCH(1,1) stationarity check and conditional volatility
  Layer 2 -- ML forecasting, financial backtesting and walk-forward validation
"""

from __future__ import annotations

from report import main

if __name__ == "__main__":
    main()
