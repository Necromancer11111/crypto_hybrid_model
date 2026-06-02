"""Top-level entry point for the Crypto Hybrid Model project.

Running this file executes the full pipeline with console reports and plots:
  Layer 0 -- data acquisition and log-return computation
  Layer 1 -- ADF + GARCH(1,1) volatility charts
  Layer 2 -- ML forecasting, financial backtesting and walk-forward validation
  Layer 3 -- Monte Carlo risk simulation (VaR/CVaR) with charts

By default charts are saved to PNG without opening windows. Pass --show-plots
to display interactive matplotlib figures (each window must be closed to continue).
"""

from __future__ import annotations

import argparse

from report import main as run_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto hybrid model full pipeline")
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Open matplotlib windows (blocks until each chart is closed).",
    )
    args = parser.parse_args()
    run_pipeline(show_plots=args.show_plots)
