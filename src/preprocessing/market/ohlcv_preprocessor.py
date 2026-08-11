"""

OHLCV Preprocessor: czyszczenie i normalizacja danych cenowych.

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed/features")


class OHLCVPreprocessor:
    def __init__(self, 
                 processed_dir: Path = PROCESSED_DIR,
                 timeframe: str = "4h"
                 ) -> None:
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, 
                                 exist_ok=True
                                 )
        self.timeframe = timeframe

    def process(self, 
                df: pd.DataFrame, 
                symbol: str
                ) -> dict[str, 
                          pd.DataFrame
                          ]:
        """
        Metoda przetwarza surowe dane OHLCV 1h.
        Zwraca słownik: {'1h': df_1h, self.timeframe: df_tf}.
        """
        df = self._validate_and_sort(df)
        df = self._fill_missing_candles(df, 
                                        freq="1h"
                                        )
        df = self._remove_outliers(df)
        df = self._add_returns(df)

        df_tf = self._resample(df,
                               self.timeframe
                               )

        self._save(df, 
                   f"{symbol}_1h_clean"
                   )
        self._save(df_tf, 
                   f"{symbol}_{self.timeframe}_clean"
                   )

        return {"1h": df, 
                self.timeframe: df_tf
                }

    @staticmethod
    def _validate_and_sort(df: pd.DataFrame) -> pd.DataFrame:
        required = {"open", 
                    "high", 
                    "low", 
                    "close", 
                    "volume"
                    }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Brakujące kolumny OHLCV: {missing}")

        df = df.copy()
        df.index = pd.to_datetime(df.index, 
                                  utc=True
                                  )
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
        return df

    @staticmethod
    def _fill_missing_candles(df: pd.DataFrame, 
                              freq: str = "1h"
                              ) -> pd.DataFrame:
        """
        Metoda uzupełnia brakujące świece (exchange downtime, weekend gaps).
        Strategia: forward-fill OHLC (cena się nie zmieniła), volume = 0.
        """
        full_index = pd.date_range(start=df.index.min(), 
                                   end=df.index.max(), 
                                   freq=freq, 
                                   tz="UTC"
                                   )
        df = df.reindex(full_index)

        df["volume"] = df["volume"].fillna(0.0)

        for col in ["open", 
                    "high", 
                    "low", 
                    "close"
                    ]:
            df[col] = df[col].ffill()

        missing_count = df["close"].isna().sum()
        if missing_count > 0:
            df = df.dropna(subset=["close"])

        return df

    @staticmethod
    def _remove_outliers(df: pd.DataFrame, 
                         z_threshold: float = 10.0
                         ) -> pd.DataFrame:
        """
        Metoda usuwa świece z anomalnymi zwrotami (błędy API, flash crashe).
        z_threshold: próg Z-score dla log-return (10σ = ekstremalny outlier).
        """
        df = df.copy()
        log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()
        z_scores = (log_returns - log_returns.mean()) / log_returns.std()
        outlier_mask = z_scores.abs() > z_threshold
        outlier_count = outlier_mask.sum()

        if outlier_count > 0:
            print(f"[OHLCV] Usunięto {outlier_count} outlierów (|Z| > {z_threshold})")
            df = df[~df.index.isin(z_scores[outlier_mask].index)]

        return df

    @staticmethod
    def _add_returns(df: pd.DataFrame) -> pd.DataFrame:
        """Metoda dodaje kolumny zwrotów"""
        df = df.copy()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["pct_return"] = df["close"].pct_change()
        return df

    @staticmethod
    def _resample(df: pd.DataFrame,
                  freq: str
                  ) -> pd.DataFrame:
        """Metoda agreguje dane 1h do self.timeframe"""
        df_tf = df.resample(freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "log_return": "sum",
            "pct_return": lambda x: (1 + x).prod() - 1
            })
        df_tf = df_tf.dropna(subset=["close"])
        return df_tf

    def _save(self, 
              df: pd.DataFrame, 
              name: str
              ) -> None:
        path = self.processed_dir / f"{name}.parquet"
        df.to_parquet(path)
        print(f"[OHLCV Preprocessor] {name}: {len(df)} świec -> {path}")
