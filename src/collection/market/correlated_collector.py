"""

Correlated Assets Collector — instrumenty korelujące z krypto.

Zrodlo: yfinance (darmowe, bez klucza API).
Instrumenty:
  Indeksy:    ^GSPC (S&P500), ^IXIC (NASDAQ), ^DJI (Dow Jones), ^VIX
  Surowce:    GC=F (zloto), CL=F (ropa WTI), SI=F (srebro)
  Waluty:     DX-Y.NYB (DXY), EURUSD=X, JPY=X
  Obligacje:  ^TNX (10Y US Treasury yield)

Interwal: 1d (dzienny).
yfinance ogranicza dane intraday (1h) do ostatnich 730 dni.

"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

RAW_DIR = Path("data/raw/correlated")

TICKERS: dict[str, str] = {
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "dji": "^DJI",
    "vix": "^VIX",
    "gold": "GC=F",
    "oil_wti": "CL=F",
    "silver": "SI=F",
    "dxy": "DX-Y.NYB",
    "eurusd": "EURUSD=X",
    "usdjpy": "JPY=X",
    "us10y": "^TNX",
}



class CorrelatedCollector:
    def __init__(self, 
                 raw_dir: Path = RAW_DIR
                 ) -> None:
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, 
                           exist_ok=True)

    def collect_all(self, 
                    start: str = "2017-08-01", 
                    end: str = "2026-01-01"
                    ) -> None:
        """Metoda pobiera dane dzienne dla wszystkich instrumentow."""
        for name, ticker in TICKERS.items():
            self._collect_ticker(name, 
                                 ticker, 
                                 start=start, 
                                 end=end)

    def _collect_ticker(self, 
                        name: str, 
                        ticker: str, 
                        start: str, 
                        end: str
                        ) -> pd.DataFrame:
        """Metoda pobiera pelny zakres danych dziennych (1d)"""
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            print(f"[Correlated] {name} ({ticker}): brak danych")
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df.index.name = "timestamp"
        df = df.sort_index()

        path = self.raw_dir / f"{name}_1d.parquet"
        df.to_parquet(path)
        print(f"[Correlated] {name}: {len(df)} dni ({df.index.min().date()} -> {df.index.max().date()}) -> {path}")
        return df

    def load(self, name: str) -> pd.DataFrame:
        path = self.raw_dir / f"{name}_1d.parquet"
        if not path.exists():
            path = self.raw_dir / f"{name}_1h.parquet"
        return pd.read_parquet(path)
