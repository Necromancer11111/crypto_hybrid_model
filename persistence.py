"""Model persistence for the Layer 2 ML forecasting package.

Provides :class:`PersistenceMixin`, the part of ``CryptoMLPredictor`` that
saves trained per-ticker models (and a feature-column metadata sidecar) to
disk and restores them later.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib


class PersistenceMixin:
    """Save/load methods for ``CryptoMLPredictor``.

    Relies on attributes of the host class (``models``, ``feature_columns``,
    ``tickers``, ``interval``, ``start`` and ``end``).
    """

    @staticmethod
    def _sanitize_ticker(ticker: str) -> str:
        """Return a filesystem-safe variant of a ticker symbol."""
        return "".join(
            ch if (ch.isalnum() or ch in ("-", "_")) else "-" for ch in ticker
        )

    def save_models(self, directory: str) -> None:
        """Persist every trained model in :attr:`models` to ``directory``.

        Each model is written to ``<directory>/<sanitized_ticker>_model.joblib``
        via :func:`joblib.dump`. A ``metadata.json`` sidecar records the
        tickers, interval, date range and the feature columns used per ticker,
        so saved models can later be matched to a compatible feature set.

        Parameters
        ----------
        directory:
            Target directory. Created (with parents) if it does not exist.

        Raises
        ------
        RuntimeError
            If no models have been trained yet (nothing to persist).
        """
        if not self.models:
            raise RuntimeError(
                "No trained models to save; call train()/train_all() first."
            )

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)

        for ticker, model in self.models.items():
            model_path = target / f"{self._sanitize_ticker(ticker)}_model.joblib"
            joblib.dump(model, model_path)

        metadata: dict[str, Any] = {
            "tickers": list(self.models.keys()),
            "interval": self.interval,
            "start": self.start,
            "end": self.end,
            "feature_columns": {
                ticker: self.feature_columns.get(ticker, [])
                for ticker in self.models
            },
        }
        with (target / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    def load_models(self, directory: str) -> None:
        """Load persisted models from ``directory`` into :attr:`models`.

        Reads ``<directory>/<sanitized_ticker>_model.joblib`` for each
        configured ticker and, when present, restores the per-ticker feature
        columns from ``metadata.json``. Missing model files are skipped with a
        warning rather than raising.

        Parameters
        ----------
        directory:
            Directory previously populated by :meth:`save_models`.

        Raises
        ------
        FileNotFoundError
            If ``directory`` does not exist.
        """
        source = Path(directory)
        if not source.is_dir():
            raise FileNotFoundError(f"Model directory not found: {directory}")

        metadata: dict[str, Any] = {}
        metadata_path = source / "metadata.json"
        if metadata_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            except (json.JSONDecodeError, OSError) as exc:
                warnings.warn(
                    f"Could not read metadata.json ({exc!r}); "
                    "feature columns will not be restored.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        saved_columns: dict[str, list[str]] = metadata.get("feature_columns", {})
        candidate_tickers = list(saved_columns) or list(self.tickers)

        for ticker in candidate_tickers:
            model_path = source / f"{self._sanitize_ticker(ticker)}_model.joblib"
            if not model_path.exists():
                warnings.warn(
                    f"No saved model for '{ticker}' at {model_path}; skipped.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            self.models[ticker] = joblib.load(model_path)
            if ticker in saved_columns:
                self.feature_columns[ticker] = list(saved_columns[ticker])
