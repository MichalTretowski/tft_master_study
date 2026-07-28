"""

Skrypt pobiera dane OHLCV z giełdy Binance przy użyciu biblioteki CCTX

Zakres: Sierpień 2017 - grudzień 2025
Interwał: 1H
Instrumenty: BTC/USDT, ETH/USDT.

Uwaga na przyszłość: CCTX daje dostęp do wielu giełd. 
Limit na zapytanie dla różnych giełd różni się.
Przed zmianą giełdy warto sprawdzić maksymalny limit i zmienić wartość w kodzie.


"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import ccxt
import pandas as pd

EXCHANGE = "binance"
SYMBOLS = ["BTC/USTD", "ETH/USDT"]
TIMEFRAME = "1h"
SINCE = "2017-08-01T00:00:00Z"
UNTIL = "2026-01*01T00:00:00Z"
LIMIT = 1000                        # Maksymalna liczba świec na request
RAW_DIR = Path("data/raw/ohlcv")

class OHLCVCollector:
    def __init__(
            self,
            exchange_id: str = EXCHANGE,
            symbols: list[str] = SYMBOLS,
            timeframe: str = TIMEFRAME,
            raw_dir: Path = RAW_DIR,
    ) -> None:
        self.exchange: ccxt.Exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        self.symbols = symbols
        self.timeframe = timeframe
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def collect_all(
            self,
            since: str = SINCE,
            until: str = UNTIL
    ) -> None:
        """Metoda pobierająca pełny zakres historyczny dla wszystkich zdefiniowanych symboli"""
        for symbol in self.symbols:
            self.collect_symbol(symbol, 
                                since=since, 
                                until=until
                                )

    def collect_symbol(
            self,
            symbol: str, 
            since: str = SINCE,
            until: str = UNTIL
    ) -> pd.DataFrame:
        """Metoda pobiera dane dla jednego symbolu i zapisuje je w formacie parquet"""
        since_ms = self._iso_to_ms(since)
        until_ms = self._iso_to_ms(until)

        all_candles: list[list] = []
        cursor = since_ms

        while cursor < until_ms:
            candles = self.exchange.fetch_ohlcv(symbol, 
                                                self.timeframe, 
                                                since=cursor,
                                                limit=LIMIT
                                                )
            if not candles:
                break
            all_candles.extend(candles)
            time.sleep(self.exchange.rateLimit / 1000)
        
        df = self._to_dataframe(all_candles)
        df = df[df.index < pd.Timestamp(until, tz="UTC")]

        out_path = self.raw_dir / f"{symbol.replace('/', '_')}_{self.timeframe}.parquet"
        df.to_parquet(out_path)
        print(f"[OHLCV] {symbol}: {len(df)} świec -> {out_path}")
        return df
    
    def load(self,
             symbol: str
             ) -> pd.DataFrame:
        """Metoda wczytuje zapisany plik parquet"""
        path = self.raw_dir / f"{symbol.replace('/', '_')}_{self.timeframe}.parquet"
        return pd.read_parquet(path)
    
    @staticmethod
    def _iso_to_ms(iso:str) -> int:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    
    @staticmethod
    def _to_dataframe(candles: list[list]) -> pd.DataFrame:
        df = pd.DataFrame(candles, 
                          columns = ["timestamp",
                                     "open",
                                     "high",
                                     "low",
                                     "close",
                                     "volume"
                                     ])
        df["timestamp"] = pd.to_datetime(df["timestamp"],
                                         unit="ms",
                                         utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        return df.astype(float)
