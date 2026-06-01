"""Core estimator for the Layer 2 ML forecasting package.

Defines :class:`CryptoMLPredictor`, which composes the feature-engineering,
backtesting and persistence mixins and adds the training loop, next-day
prediction, feature-importance reporting and optional hyperparameter tuning.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from backtest import BacktestMixin
from features import FeatureMixin
from mlbase import DEFAULT_TICKERS, TARGET_COLUMN, FoldMetric, XGBRegressor
from persistence import PersistenceMixin
from statistical_analysis import annualization_periods


class CryptoMLPredictor(FeatureMixin, BacktestMixin, PersistenceMixin):
    """Train ML models that predict next-day crypto log returns.

    Parameters
    ----------
    tickers:
        Yahoo Finance ticker symbols to model.
    start, end:
        Date range used to fetch daily market data.
    interval:
        Data interval. Defaults to ``"1d"`` because the forecasting target is
        the next day's log return.
    n_splits:
        Number of folds for :class:`~sklearn.model_selection.TimeSeriesSplit`.
    random_state:
        Seed used by tree-based regressors.
    signal_threshold:
        Long/Flat entry threshold on the predicted return. Go long only when
        the predicted next-day return exceeds this value, else stay in cash.
    cost_bps:
        Transaction cost in basis points charged on position changes.
    tune:
        When ``True``, :meth:`train` runs a time-series-aware randomized
        hyperparameter search per ticker before fitting. Defaults to ``False``
        so the default behaviour is unchanged.
    n_iter_search:
        Number of parameter settings sampled by
        :class:`~sklearn.model_selection.RandomizedSearchCV` when ``tune`` is
        enabled.
    cache_dir:
        Optional directory for an on-disk cache of the raw market download,
        forwarded to :class:`~data.data.CryptoDataFetcher`. When ``None``
        (default) every run hits yfinance; when set, repeated runs are
        reproducible and offline-friendly.
    use_cache:
        If ``True`` (default) and ``cache_dir`` is set, an existing cache file
        is reused; set to ``False`` to force a fresh download while keeping the
        cache directory configured.
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        start: str = "2020-01-01",
        end: str = "2026-06-01",
        interval: str = "1d",
        n_splits: int = 5,
        random_state: int = 42,
        signal_threshold: float = 0.0,
        cost_bps: float = 10.0,
        tune: bool = False,
        n_iter_search: int = 20,
        cache_dir: str | None = None,
        use_cache: bool = True,
    ) -> None:
        self.tickers = tickers or DEFAULT_TICKERS
        self.start = start
        self.end = end
        self.interval = interval
        self.n_splits = n_splits
        self.random_state = random_state
        self.signal_threshold = signal_threshold
        self.cost_bps = cost_bps
        self.tune = tune
        self.n_iter_search = n_iter_search
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.periods_per_year = annualization_periods(interval)
        if interval != "1d":
            warnings.warn(
                "CryptoMLPredictor target is defined as NEXT-DAY return; "
                f"received interval='{interval}', so target is next '{interval}' "
                "period instead. Prefer interval='1d' for daily forecasting.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.close_prices: pd.DataFrame | None = None
        self.log_returns: pd.DataFrame | None = None
        self.volume_data: pd.DataFrame | None = None
        self.garch_features: pd.DataFrame | None = None
        self.feature_data: dict[str, pd.DataFrame] = {}
        self.feature_columns: dict[str, list[str]] = {}
        self.models: dict[str, Any] = {}
        self.metrics: dict[str, list[FoldMetric]] = {}
        self.oos_predictions: dict[str, pd.DataFrame] = {}
        self.best_params: dict[str, dict] = {}

    def train(self, ticker: str) -> list[FoldMetric]:
        """Train and evaluate one ticker with walk-forward time-series splits.

        Cross-validation metrics are computed fold by fold. After evaluation, a
        fresh model is fitted on all labeled rows so it can be used for the
        next-day prediction method.
        """
        if ticker not in self.tickers:
            raise KeyError(f"'{ticker}' is not configured: {self.tickers}")
        if not self.feature_data:
            self.prepare_features()

        dataset = self._labeled_dataset(ticker)
        if dataset.shape[0] <= self.n_splits:
            raise ValueError(
                f"[{ticker}] Need more rows than n_splits={self.n_splits}; "
                f"got {dataset.shape[0]} labeled rows."
            )

        x = dataset[self.feature_columns[ticker]]
        y = dataset[TARGET_COLUMN]
        splitter = TimeSeriesSplit(n_splits=self.n_splits)

        if self.tune and ticker not in self.best_params:
            self.tune_hyperparameters(ticker)
        tuned = self.best_params.get(ticker) if self.tune else None

        fold_metrics: list[FoldMetric] = []
        oos_frames: list[pd.DataFrame] = []
        for fold, (train_idx, test_idx) in enumerate(splitter.split(x), start=1):
            model = self._new_model(tuned)
            model.fit(x.iloc[train_idx], y.iloc[train_idx])
            predictions = model.predict(x.iloc[test_idx])

            y_test = y.iloc[test_idx]
            fold_metrics.append(
                FoldMetric(
                    ticker=ticker,
                    fold=fold,
                    model=self._model_name(model),
                    train_size=len(train_idx),
                    test_size=len(test_idx),
                    mae=float(mean_absolute_error(y_test, predictions)),
                    r2=float(r2_score(y_test, predictions)),
                )
            )
            oos_frames.append(
                pd.DataFrame(
                    {"y_true": y_test.to_numpy(), "y_pred": predictions},
                    index=y_test.index,
                )
            )

        # Continuous out-of-sample series (test folds are chronological and
        # non-overlapping), used for honest financial backtesting.
        self.oos_predictions[ticker] = pd.concat(oos_frames).sort_index()

        final_model = self._new_model(tuned)
        final_model.fit(x, y)
        self.models[ticker] = final_model
        self.metrics[ticker] = fold_metrics
        return fold_metrics

    def train_all(self) -> dict[str, list[FoldMetric]]:
        """Prepare features and train one model per configured ticker."""
        if not self.feature_data:
            self.prepare_features()
        return {ticker: self.train(ticker) for ticker in self.tickers}

    def predict_next_day_return(self, ticker: str) -> float:
        """Return predicted expected log return for the next day.

        If the model has not been trained yet, this method trains it first.
        The prediction row is the latest fully available feature row and does
        not require the shifted target to be present.
        """
        if ticker not in self.tickers:
            raise KeyError(f"'{ticker}' is not configured: {self.tickers}")
        if ticker not in self.models:
            self.train(ticker)
        # A model may have been restored via load_models() without features
        # being built yet; ensure the feature matrix exists before predicting.
        if ticker not in self.feature_data:
            self.prepare_features()

        prediction_rows = (
            self.feature_data[ticker][self.feature_columns[ticker]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if prediction_rows.empty:
            raise ValueError(f"[{ticker}] No valid feature rows for prediction.")

        latest_features = prediction_rows.tail(1)
        prediction = self.models[ticker].predict(latest_features)[0]
        return float(prediction)

    def feature_importances(self, ticker: str) -> pd.Series:
        """Return feature importances for a trained ticker model."""
        if ticker not in self.models:
            raise KeyError(f"[{ticker}] Model is not trained yet.")

        model = self.models[ticker]
        if not hasattr(model, "feature_importances_"):
            raise AttributeError(
                f"[{ticker}] {self._model_name(model)} has no feature_importances_."
            )

        return pd.Series(
            model.feature_importances_,
            index=self.feature_columns[ticker],
            name=ticker,
        ).sort_values(ascending=False)

    def metrics_table(self) -> pd.DataFrame:
        """Return fold and average metrics as a plain pandas DataFrame."""
        rows: list[dict[str, Any]] = []
        for ticker, metrics in self.metrics.items():
            for metric in metrics:
                rows.append(metric.__dict__)
            if metrics:
                rows.append(
                    {
                        "ticker": ticker,
                        "fold": "avg",
                        "model": metrics[0].model,
                        "train_size": "",
                        "test_size": "",
                        "mae": float(np.mean([metric.mae for metric in metrics])),
                        "r2": float(np.mean([metric.r2 for metric in metrics])),
                    }
                )
        return pd.DataFrame(rows)

    def _new_model(self, params: dict | None = None) -> Any:
        """Create a fresh regressor, preferring XGBoost when available.

        When ``params`` is provided (typically the result of
        :meth:`tune_hyperparameters`), the given values are merged over the
        sensible defaults of the active estimator class. ``random_state`` and
        ``n_jobs`` are always enforced so reproducibility is preserved.
        """
        if XGBRegressor is not None:
            defaults: dict[str, Any] = {
                "objective": "reg:squarederror",
                "n_estimators": 300,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            }
            if params:
                defaults.update(params)
            defaults["random_state"] = self.random_state
            defaults["n_jobs"] = -1
            return XGBRegressor(**defaults)

        defaults = {
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_leaf": 5,
        }
        if params:
            defaults.update(params)
        defaults["random_state"] = self.random_state
        defaults["n_jobs"] = -1
        return RandomForestRegressor(**defaults)

    def tune_hyperparameters(self, ticker: str) -> dict:
        """Search hyperparameters for one ticker respecting time order.

        Uses :class:`~sklearn.model_selection.RandomizedSearchCV` with a
        :class:`~sklearn.model_selection.TimeSeriesSplit` so that every
        validation fold is strictly later than its training data (no
        look-ahead). The parameter distribution matches the active estimator
        class (XGBoost when available, otherwise random forest). The best
        parameters are cached in :attr:`best_params` and returned.
        """
        if ticker not in self.tickers:
            raise KeyError(f"'{ticker}' is not configured: {self.tickers}")
        if not self.feature_data:
            self.prepare_features()

        dataset = self._labeled_dataset(ticker)
        x = dataset[self.feature_columns[ticker]]
        y = dataset[TARGET_COLUMN]

        if XGBRegressor is not None:
            base_estimator: Any = XGBRegressor(
                objective="reg:squarederror",
                random_state=self.random_state,
                n_jobs=-1,
            )
            param_distributions: dict[str, list[Any]] = {
                "n_estimators": [200, 300, 500, 800],
                "max_depth": [2, 3, 4, 5],
                "learning_rate": [0.01, 0.02, 0.05, 0.1],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "min_child_weight": [1, 3, 5],
                "reg_lambda": [0.0, 1.0, 5.0],
            }
        else:
            base_estimator = RandomForestRegressor(
                random_state=self.random_state,
                n_jobs=-1,
            )
            param_distributions = {
                "n_estimators": [200, 300, 500],
                "max_depth": [4, 6, 8, 12, None],
                "min_samples_leaf": [1, 3, 5, 10],
                "max_features": ["sqrt", "log2", 1.0],
            }

        search = RandomizedSearchCV(
            estimator=base_estimator,
            param_distributions=param_distributions,
            n_iter=self.n_iter_search,
            scoring="neg_mean_absolute_error",
            cv=TimeSeriesSplit(n_splits=self.n_splits),
            random_state=self.random_state,
            n_jobs=-1,
            refit=True,
        )
        search.fit(x, y)
        self.best_params[ticker] = dict(search.best_params_)
        return self.best_params[ticker]

    @staticmethod
    def _model_name(model: Any) -> str:
        """Return a concise model class name for reporting."""
        return type(model).__name__
