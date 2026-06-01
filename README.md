# Crypto Hybrid Model

A three-layer quantitative research framework for Bitcoin and Ethereum that
combines statistical volatility modelling (GARCH) with machine learning to
forecast next-day log returns and evaluate systematic Long/Flat trading
strategies.

---

## Architecture

```
Layer 0 — Data
  data/data.py          CryptoDataFetcher
                        yfinance download → log-returns, rolling vol, volume
                        on-disk cache (parquet / CSV fallback)

Layer 1 — Statistical
  statistical_analysis.py   CryptoVolatilityModel
                            ADF stationarity test
                            GARCH(1,1) conditional volatility (annualised)

Layer 2 — Machine Learning
  mlbase.py             Shared constants and FoldMetric record
  features.py           FeatureMixin  – leakage-free feature engineering
  backtest.py           BacktestMixin – Long/Flat strategy evaluation
  persistence.py        PersistenceMixin – save/load models (joblib)
  model.py              CryptoMLPredictor – training, prediction, tuning
  report.py             Console formatters + main() pipeline

  finance_metrics.py    Pure risk/return functions (Sharpe, Sortino,
                        max drawdown, VaR, CVaR, walk-forward calibration)

  ml_forecasting.py     Thin facade – keeps the original entry point and
                        all existing imports working unchanged
```

---

## Features

| Category | Details |
|---|---|
| **Data** | BTC-USD, ETH-USD daily OHLCV via yfinance; optional on-disk cache |
| **Statistical** | ADF test, GARCH(1,1) with convergence retry, dynamic annualisation |
| **Feature set** | Log-return lags (1/2/3/5/7d), SMA ratios, RSI-14, rolling vol (7/21d), GARCH vol, volume features, BTC cross-asset context |
| **Model** | XGBRegressor (preferred) or RandomForestRegressor; TimeSeriesSplit |
| **Tuning** | Optional RandomizedSearchCV with TimeSeriesSplit (leakage-free) |
| **Backtest** | Long/Flat strategy with basis-point transaction costs |
| **Threshold** | Grid-search (Sortino-optimal) + nested walk-forward calibration |
| **Metrics** | MAE, R², directional accuracy, Sharpe, Sortino, max drawdown, VaR, CVaR, win rate, cumulative return |
| **Verdict** | Rule-based classification: PELNA PRZEWAGA / DEFENSYWNA PRZEWAGA / BRAK PRZEWAGI |
| **Persistence** | joblib model files + metadata.json sidecar |

---

## Quick Start

### 1. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell
# source venv/bin/activate     # Linux / macOS
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

> **Windows note**: if `xgboost` fails to install, first install the
> [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

### 3. Run the full pipeline

```powershell
# From the project root (crypto_hybrid_model/)
python main.py
# or equivalently
python ml_forecasting.py
```

The first run downloads data from yfinance and writes a local cache to
`data_cache/`. Subsequent runs load from cache and start immediately.

---

## Output

Running `main.py` produces a console report with six sections:

1. **Layer 2 ML Forecasting Metrics** — MAE and R² per fold and on average
2. **Feature Importances** — ranked feature contributions per ticker
3. **Financial Metrics** — Long/Flat strategy vs buy-and-hold (fixed threshold)
4. **Investment Verdict** — rule-based assessment with caveats
5. **Calibrated Threshold** — Sortino-optimal threshold (optimistic upper bound)
6. **Walk-Forward Validation** — leakage-free strategy vs benchmark per window

---

## Module Reference

| Module | Public API |
|---|---|
| `data/data.py` | `CryptoDataFetcher(tickers, start, end, interval, cache_dir)` |
| `statistical_analysis.py` | `CryptoVolatilityModel(returns, trading_days)` |
| `model.py` | `CryptoMLPredictor(tickers, start, end, ..., cache_dir)` |
| `finance_metrics.py` | `summarize`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `walk_forward_threshold`, … |

```python
from model import CryptoMLPredictor

predictor = CryptoMLPredictor(cache_dir="data_cache")
predictor.train_all()

# Next-day forecast
btc_mu = predictor.predict_next_day_return("BTC-USD")

# Walk-forward validated backtest
result = predictor.walk_forward_threshold("BTC-USD")
print(result["verdict"])

# Save / restore
predictor.save_models("models_artifacts")
```

---

## Interpretation Guide

| Metric | What it means |
|---|---|
| R² < 0 | Model is worse than predicting the mean — expected for daily crypto returns |
| Dir. accuracy ~49 % | No directional edge; the model learns risk avoidance, not direction |
| Lower volatility, shallower drawdown | Primary benefit of the strategy vs buy-and-hold |
| PELNA PRZEWAGA | Strategy beats benchmark on return AND risk-adjusted metrics |
| DEFENSYWNA PRZEWAGA | Better risk metrics, lower total return — safer but less profitable |
| BRAK PRZEWAGI | No improvement over buy-and-hold |

> The walk-forward verdict is the most trustworthy result: neither the model
> nor the threshold ever sees the evaluation data.

---

## Requirements

- Python 3.10+
- See `requirements.txt` for package versions
- Internet connection for the first run (yfinance download); offline thereafter

---

## Disclaimer

This project is for research and educational purposes only.
It does not constitute financial advice.
Past performance of backtested strategies does not guarantee future results.
