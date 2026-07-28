"""

Derivatives Collector — dane z rynku kontraktów perpetual dla BTC i ETH.

Zbiera trzy typy danych, każdy z darmowego API (bez klucza):

  Funding rate     Binance Futures   co 8h    pełna historia od 2019
  Open Interest    Bybit v5          1h       historia od 2020 (paginacja backward)
  Long/Short ratio Bybit v5          1h       historia od 2020 (paginacja backward)

Zakresy (zweryfikowane):
  BTC OI/LS:  2020-07-20 → obecnie
  ETH OI/LS:  2020-10-21 → obecnie

Zapis: data/raw/derivatives/{COIN}_{typ}_{źródło}.parquet

Dokumentacja API:
  Binance Futures: https://binance-docs.github.io/apidocs/futures/en/
  Bybit v5:        https://bybit-exchange.github.io/docs/v5/market/open-interest

"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path("data/raw/derivatives")

BINANCE_BASE = "https://fapi.binance.com"
BYBIT_BASE = "https://api.bybit.com"

BINANCE_SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
BYBIT_SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}

COINS = ["BTC", "ETH"]
START = "2020-01-01"
END = "2026-01-01"
INTERVAL = "1h"

class DerivativesCollector:
    def __init__(self,
                 raw_dir: Path = RAW_DIR
                 ) -> None:
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "crypto-research/1.0"})



    def collect_all(self,
                    coins: list[str] | None = None,
                    start: str = START,
                    end: str = END
                    ) -> None:
        if coins is None:
            coins = COINS
        """Metoda pobiera wszystkie dane pochodnych dla listy coinów"""
        for coin in coins:
            print(f"\n=== {coin} ===")
            self.collect_funding_rates(coin)
            self.collect_open_interest(coin, start=start, end=end)
            self.collect_long_short_ratio(coin, start=start, end=end)



    def collect_funding_rates(self,
                              coin: str
                              ) -> pd.DataFrame:
        """Metoda pobierająca historyczne funding rates z giełdy Binance"""
        symbol = BINANCE_SYMBOLS[coin]
        records: list[dict] = []
        end_time: int | None = None

        LIMIT = 200

        start_time = self._to_ms("2019-01-01")

        while True:
            params: dict = {
                "symbol": symbol,
                "limit": LIMIT,
                "startTime": start_time
            }

            data = self._get(BINANCE_BASE, "/fapi/v1/fundingRate", params)
            if not data:
                break

            records.extend(data)
            start_time = data[-1]["fundingTime"] + 1 # data[-1] to najnowszy w batchu
            time.sleep(0.1)

            if len(data) < LIMIT:
                break

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["funding_rate"] = df["fundingRate"].astype(float)
        df = (df[["timestamp", "funding_rate"]]
              .sort_values("timestamp")
              .drop_duplicates("timestamp")
              .reset_index(drop=True))

        self._save(df, f"{coin}_funding_rates_binance")
        return df
    


    def collect_open_interest(
        self,
        coin: str,
        start: str = START,
        end: str   = END,
        interval:  str = INTERVAL,
    ) -> pd.DataFrame:
        """Metoda pobierająca historyczne Open Interest z giełdy Bybit"""

        symbol = BYBIT_SYMBOLS[coin]
        start_ms = self._to_ms(start)
        end_ms = self._to_ms(end)
        step_ms = 200 * 3_600_000  # 200 świec × 1h w ms
        all_rows: list[dict] = []

        cursor_ms = start_ms
        consecutive_empty = 0
        MAX_EMPTY = 10

        while cursor_ms < end_ms:
            params = {
                "category": "linear",
                "symbol": symbol,
                "intervalTime": interval,
                "startTime": cursor_ms,
                "endTime": min(cursor_ms + step_ms, end_ms),
                "limit": 200,
            }
            data = self._get(BYBIT_BASE, "/v5/market/open-interest", params)
            if data is None:
                cursor_ms += step_ms
                time.sleep(2)
                continue
            rows = data.get("result", {}).get("list", []) if data else []

            if not rows:
                consecutive_empty += 1
                if consecutive_empty >= MAX_EMPTY:
                    from datetime import timedelta
                    ninety_days_ago = end_ms - 90 * 24 * 3_600_000
                    if cursor_ms < ninety_days_ago:
                        cursor_ms = ninety_days_ago
                        consecutive_empty = 0
                        continue
                    break
                cursor_ms += step_ms
                time.sleep(0.1)
                continue

            consecutive_empty = 0
            all_rows.extend(rows)
            oldest_ts = int(rows[-1]["timestamp"])
            cursor_ms = oldest_ts + 3_600_000 
            time.sleep(0.1)

        if not all_rows:
            print(f"[Derivatives] {coin} OI: brak danych z Bybit")
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        df["timestamp"]     = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
        df["open_interest"] = df["openInterest"].astype(float)
        df = (df[["timestamp", "open_interest"]]
              .sort_values("timestamp")
              .drop_duplicates("timestamp")
              .reset_index(drop=True))

        self._save(df, f"{coin}_open_interest_1h_bybit")
        return df
    


    def collect_long_short_ratio(
        self,
        coin: str,
        start: str = START,
        end: str = END,
        interval: str = INTERVAL,
    ) -> pd.DataFrame:
        """Metoda pobierająca historyczne Long/Short Ratio z giełdy Bybit"""
        symbol = BYBIT_SYMBOLS[coin]
        start_ms = self._to_ms(start)
        end_ms = self._to_ms(end)
        step_ms = 500 * 3_600_000
        all_rows: list[dict] = []

        cursor_ms = start_ms
        consecutive_empty = 0
        MAX_EMPTY = 10

        while cursor_ms < end_ms:
            params = {
                "category": "linear",
                "symbol": symbol,
                "period": interval,
                "startTime": cursor_ms,
                "endTime": min(cursor_ms + step_ms, end_ms),
                "limit": 500,
            }
            data = self._get(BYBIT_BASE, "/v5/market/account-ratio", params)
            if data is None:
                cursor_ms += step_ms
                time.sleep(2)
                continue
            rows = data.get("result", {}).get("list", []) if data else []

            if not rows:
                consecutive_empty += 1
                if consecutive_empty >= MAX_EMPTY:
                    ninety_days_ago = end_ms - 90 * 24 * 3_600_000
                    if cursor_ms < ninety_days_ago:
                        cursor_ms = ninety_days_ago
                        consecutive_empty = 0
                        continue
                    break
                cursor_ms += step_ms
                time.sleep(0.1)
                continue

            consecutive_empty = 0
            all_rows.extend(rows)
            oldest_ts = int(rows[-1]["timestamp"])
            cursor_ms = oldest_ts + 3_600_000
            time.sleep(0.1)

        if not all_rows:
            print(f"[Derivatives] {coin} L/S ratio: brak danych z Bybit")
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
        df["long_short_ratio"] = df["buyRatio"].astype(float) / df["sellRatio"].astype(float)
        df["long_ratio"] = df["buyRatio"].astype(float)
        df["short_ratio"] = df["sellRatio"].astype(float)
        df = (df[["timestamp", "long_short_ratio", "long_ratio", "short_ratio"]]
              .sort_values("timestamp")
              .drop_duplicates("timestamp")
              .reset_index(drop=True))

        self._save(df, f"{coin}_long_short_ratio_1h_bybit")
        return df

    def _get(self, 
             base: str, 
             endpoint: str, 
             params: dict
             ) -> dict | list | None:
        """Metoda zwraca None przy błędzie sieciowym, {} przy pustej odpowiedzi API"""
        for attempt in range(3):
            try:
                resp = self._session.get(f"{base}{endpoint}", params=params, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                wait = 10 * (attempt + 1)
                print(f"[Derivatives] {base}{endpoint}: błąd ({attempt+1}/3) — {e}")
                if attempt < 2:
                    time.sleep(wait)
        return None 

    @staticmethod
    def _to_ms(date_str: str) -> int:
        dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    def _save(self, 
              df: pd.DataFrame, 
              name: str
              ) -> None:
        path = self.raw_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"[Derivatives] {name}: {len(df)} wierszy -> {path}")
