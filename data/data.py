from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


class CryptoDataFetcher:
    """Downloads OHLCV data for crypto (or any) tickers via yfinance and
    computes derived financial features such as log-returns and rolling
    annualised volatility.

    Parameters
    ----------
    tickers:
        List of Yahoo Finance ticker symbols, e.g. ``["BTC-USD", "ETH-USD"]``.
    start:
        Start date string in ``YYYY-MM-DD`` format.
    end:
        End date string in ``YYYY-MM-DD`` format.
    interval:
        Data frequency accepted by yfinance (``"1d"``, ``"1h"``, etc.).
        Default is daily.
    cache_dir:
        Optional directory for an on-disk cache of the raw download. When
        ``None`` (the default) caching is disabled and :meth:`fetch` always
        hits yfinance. When set, a cache file keyed by
        ``tickers + start + end + interval`` is written/read so repeated runs
        are reproducible and offline-friendly.
    use_cache:
        If ``True`` (default) and ``cache_dir`` is set, :meth:`fetch` reads an
        existing cache file when present and writes one after a fresh
        download. If ``False``, the cache is bypassed even when ``cache_dir``
        is provided.
    """

    def __init__(
        self,
        tickers: list[str],
        start: str,
        end: str,
        interval: str = "1d",
        cache_dir: str | None = None,
        use_cache: bool = True,
    ) -> None:
        self.tickers = tickers
        self.start = start
        self.end = end
        self.interval = interval
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self._raw: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Data acquisition
    # ------------------------------------------------------------------

    def fetch(self) -> pd.DataFrame:
        """Download OHLCV data from Yahoo Finance and cache it internally.

        When ``cache_dir`` is set and ``use_cache`` is ``True``, an existing
        on-disk cache file is loaded instead of downloading; otherwise the data
        is downloaded from yfinance and then written to the cache. Behaviour is
        identical to a plain download when ``cache_dir`` is ``None``.

        Returns
        -------
        pd.DataFrame
            Multi-level columns ``(field, ticker)`` for multiple tickers,
            or flat columns for a single ticker.  Index is a
            ``DatetimeIndex``.
        """
        if self.cache_dir is not None and self.use_cache:
            cached = self._load_from_cache()
            if cached is not None and not cached.empty:
                self._raw = cached
                return self._raw

        self._raw = yf.download(
            tickers=self.tickers,
            start=self.start,
            end=self.end,
            interval=self.interval,
            auto_adjust=True,
            progress=False,
        )
        if self._raw.empty:
            raise ValueError(
                f"No data returned for tickers={self.tickers} "
                f"between {self.start} and {self.end}."
            )

        if self.cache_dir is not None and self.use_cache:
            self._save_to_cache(self._raw)
        return self._raw

    # ------------------------------------------------------------------
    # On-disk cache (optional, reproducibility / offline reruns)
    # ------------------------------------------------------------------

    def _cache_key(self) -> str:
        """Build a filesystem-safe cache key from the request parameters."""
        raw_key = (
            f"{'_'.join(self.tickers)}_{self.start}_{self.end}_{self.interval}"
        )
        return "".join(
            ch if (ch.isalnum() or ch in ("-", "_")) else "-" for ch in raw_key
        )

    def _ensure_cache_dir(self) -> Path:
        """Return the cache directory as a ``Path``, creating it if missing."""
        directory = Path(self.cache_dir)  # type: ignore[arg-type]
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _load_from_cache(self) -> pd.DataFrame | None:
        """Load a previously cached download, preferring parquet over CSV.

        Returns ``None`` when no usable cache file exists so the caller can
        fall back to a fresh download.
        """
        directory = Path(self.cache_dir)  # type: ignore[arg-type]
        key = self._cache_key()
        parquet_path = directory / f"{key}.parquet"
        csv_path = directory / f"{key}.csv"

        if parquet_path.exists():
            try:
                return pd.read_parquet(parquet_path)
            except ImportError:
                warnings.warn(
                    "parquet engine unavailable; cannot read parquet cache, "
                    "trying CSV fallback.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            except Exception as exc:  # pragma: no cover - corrupt/partial file
                warnings.warn(
                    f"Failed to read parquet cache ({exc!r}); ignoring it.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if csv_path.exists():
            try:
                return self._read_csv_cache(csv_path)
            except Exception as exc:  # pragma: no cover - corrupt/partial file
                warnings.warn(
                    f"Failed to read CSV cache ({exc!r}); ignoring it.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return None

    def _read_csv_cache(self, path: Path) -> pd.DataFrame:
        """Read a CSV cache, reconstructing the column layout used on save.

        Multi-ticker downloads use a ``(field, ticker)`` MultiIndex header
        (two header rows); single-ticker downloads use a flat header.
        """
        header: int | list[int] = [0, 1] if len(self.tickers) > 1 else 0
        frame = pd.read_csv(path, header=header, index_col=0)
        frame.index = pd.to_datetime(frame.index)
        return frame

    def _save_to_cache(self, frame: pd.DataFrame) -> None:
        """Persist a freshly downloaded frame, preferring parquet.

        Falls back to CSV when the parquet engine (``pyarrow``) is missing or
        the parquet write fails (e.g. MultiIndex columns on older engines).
        Any failure degrades to a warning rather than corrupting the cache.
        """
        directory = self._ensure_cache_dir()
        key = self._cache_key()
        parquet_path = directory / f"{key}.parquet"

        try:
            frame.to_parquet(parquet_path)
            return
        except ImportError:
            warnings.warn(
                "parquet engine (pyarrow) unavailable; falling back to CSV "
                "cache.",
                RuntimeWarning,
                stacklevel=2,
            )
        except Exception as exc:  # pragma: no cover - engine-specific failure
            warnings.warn(
                f"Failed to write parquet cache ({exc!r}); falling back to CSV.",
                RuntimeWarning,
                stacklevel=2,
            )

        csv_path = directory / f"{key}.csv"
        try:
            frame.to_csv(csv_path)
        except Exception as exc:  # pragma: no cover - filesystem failure
            warnings.warn(
                f"Failed to write CSV cache ({exc!r}); caching skipped.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _ensure_fetched(self) -> pd.DataFrame:
        if self._raw is None:
            self.fetch()
        return self._raw  # type: ignore[return-value]

    def close_prices(self, price_col: str = "Close") -> pd.DataFrame:
        """Return closing prices for all tickers as a DataFrame.

        Parameters
        ----------
        price_col:
            Price field to extract (``"Close"`` by default). With
            ``auto_adjust=True`` the ``Close`` column is already adjusted.
        """
        df = self._ensure_fetched()
        if isinstance(df.columns, pd.MultiIndex):
            return df[price_col]
        return df[[price_col]]

    def volumes(self) -> pd.DataFrame:
        """Return traded volume for all tickers as a DataFrame.

        Columns are named by ticker (matching :meth:`close_prices`). Useful as
        a liquidity/activity feature for downstream models.
        """
        return self.close_prices("Volume")

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def log_returns(self, price_col: str = "Close") -> pd.DataFrame:
        """Compute daily log-returns: ``ln(P_t / P_{t-1})``.

        Parameters
        ----------
        price_col:
            Price field to base returns on (passed through to
            :meth:`close_prices`).

        Returns
        -------
        pd.DataFrame
            Log-return series with the first row dropped (NaN from shift).
        """
        prices = self.close_prices(price_col)
        returns = np.log(prices / prices.shift(1)).dropna()
        return returns

    def rolling_volatility(
        self,
        window: int = 30,
        annualise: bool = True,
        trading_days: int = 252,
    ) -> pd.DataFrame:
        """Compute rolling realised volatility from log-returns.

        Parameters
        ----------
        window:
            Rolling window size in periods (days by default).
        annualise:
            If ``True``, multiply the rolling std by ``sqrt(trading_days)``.
        trading_days:
            Number of trading days per year used for annualisation.
            Crypto markets trade 365 days; equities typically 252.

        Returns
        -------
        pd.DataFrame
            Rolling volatility aligned on the same index as log-returns.
        """
        rets = self.log_returns()
        vol = rets.rolling(window).std()
        if annualise:
            vol = vol * np.sqrt(trading_days)
        return vol.dropna()
