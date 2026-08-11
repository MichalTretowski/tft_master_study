"""

Obliczanie wskaźników technicznych na danych 4h lub 1d.

Wszystkie cechy są obliczane na danych 4h lub 1d (interwał treningowy modelu).
Używana biblioteka: pandas-ta.

Grupy cech:
  Trend: EMA, SMA, MACD, ADX, Ichimoku
  Momentum: RSI, Stochastic, ROC, Williams %R
  Volatility: ATR, Bollinger Bands, Historical Vol
  Volume: OBV, VWAP, Volume SMA ratio
  Candle: body size, upper/lower wick, candle type
  Cycle: godzina dnia, dzień tygodnia

  """

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    print("[FeatureEngineer] pandas_ta niedostępne. Zainstaluj: pip install pandas-ta")

PROCESSED_DIR = Path("data/processed/features")


class FeatureEngineer:
    def __init__(self, 
                 processed_dir: Path = PROCESSED_DIR,
                 bars_per_day: int = 6,
                 timeframe: str = "4h"
                 ) -> None:
        self.processed_dir = processed_dir
        self.bars_per_day = bars_per_day
        self.timeframe = timeframe

    def build_features(self, 
                       df: pd.DataFrame, 
                       symbol: str
                       ) -> pd.DataFrame:
        """
        Metoda oblicza wszystkie wskaźniki techniczne dla DataFrame 4h/1d.
        Wejście: df z kolumnami [open, high, low, close, volume].
        Wyjście: df + kolumny cech.
        """
        df = df.copy()

        df = self._add_trend_features(df)
        df = self._add_momentum_features(df)
        df = self._add_volatility_features(df)
        df = self._add_volume_features(df)
        df = self._add_candle_features(df)
        df = self._add_cyclical_time_features(df)
        df = self._add_multi_scale_returns(df)

        # Usunięcie NaN z okresu warmup wskaźników
        df = df.dropna()

        path = self.processed_dir / f"{symbol}_{self.timeframe}_features.parquet"
        df.to_parquet(path)
        print(f"[Features] {symbol}: {len(df)} wierszy × {len(df.columns)} cech -> {path}")
        return df

    @staticmethod
    def _add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
        if not HAS_PANDAS_TA:
            return df
        df.ta.ema(length=9, 
                  append=True
                  )
        df.ta.ema(length=21, 
                  append=True
                  )
        df.ta.ema(length=50, 
                  append=True
                  )
        df.ta.ema(length=200, 
                  append=True
                  )
        df.ta.sma(length=20, 
                  append=True
                  )
        df.ta.macd(fast=12, 
                   slow=26, 
                   signal=9, 
                   append=True
                   )
        df.ta.adx(length=14, 
                  append=True
                  )
        return df

    @staticmethod
    def _add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        if not HAS_PANDAS_TA:
            return df
        df.ta.rsi(length=14, 
                  append=True
                  )
        df.ta.rsi(length=25, 
                  append=True
                  )
        df.ta.rsi(length=50, 
                  append=True
                  )
        df.ta.stoch(k=14, 
                    d=3, 
                    smooth_k=3, 
                    append=True
                    )
        df.ta.roc(length=10, 
                  append=True
                  )
        df.ta.willr(length=14, 
                    append=True
                    )
        df.ta.cci(length=14, 
                  append=True
                  )
        return df

    def _add_volatility_features(self,
                                 df: pd.DataFrame
                                 ) -> pd.DataFrame:
        if not HAS_PANDAS_TA:
            return df
        df.ta.atr(length=14, 
                  append=True
                  )
        df.ta.bbands(length=20, 
                     std=2, 
                     append=True
                     )

        bpd = self.bars_per_day
        ann = np.sqrt(bpd * 365)
        log_ret = np.log(df["close"] / df["close"].shift(1))
        if bpd > 1:
            df["hist_vol_24"] = log_ret.rolling(bpd).std() * ann
        df["hist_vol_7d"] = log_ret.rolling(7 * bpd).std() * ann
        df["hist_vol_30d"] = log_ret.rolling(30 * bpd).std() * ann
        return df


    def _add_volume_features(self,
                             df: pd.DataFrame
                             ) -> pd.DataFrame:
        if not HAS_PANDAS_TA:
            return df
        df.ta.obv(append=True)

        vol_sma = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / vol_sma.replace(0, np.nan)

        w = 4 * self.bars_per_day
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap_4d"] = (typical_price * df["volume"]).rolling(w).sum() / df["volume"].rolling(w).sum()
        df["price_to_vwap"] = df["close"] / df["vwap_4d"].replace(0, np.nan)
        return df

    @staticmethod
    def _add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
        """Geometria świecy jako cechy."""
        df = df.copy()
        body = (df["close"] - df["open"]).abs()
        candle_range = (df["high"] - df["low"]).replace(0, np.nan)

        df["candle_body_ratio"] = body / candle_range
        df["upper_wick_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / candle_range
        df["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / candle_range
        df["is_bullish"] = (df["close"] > df["open"]).astype(float)
        return df

    @staticmethod
    def _add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Cykliczne kodowanie czasu: godzina dnia, dzień tygodnia."""
        df = df.copy()
        hour = df.index.hour
        dow = df.index.dayofweek

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        return df


    def _add_multi_scale_returns(self,
                                 df: pd.DataFrame
                                 ) -> pd.DataFrame:
        """Zwroty na różnych horyzontach"""
        df = df.copy()
        bpd = self.bars_per_day
        if bpd > 1:
            df["return_4h"] = df["close"].pct_change(1)    
        for days, label in [(1, "1d"), 
                               (7, "7d"), 
                               (15, "15d"), 
                               (30, "30d")
                               ]:
            df[f"return_{label}"] = df["close"].pct_change(days * bpd)
        return df
