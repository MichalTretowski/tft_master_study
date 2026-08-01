"""

On-Chain Collector - metryki on-chain BTC i ETH z CoinMetrics Community.

Źródło: CoinMetrics Community API.
  - częstotliwość 1d
  - max 10 000 rekordów / request
  - dokumentacja: https://docs.coinmetrics.io/api/v4/

Zbierane dane:
  BTC: HashRate, TxCnt, AdrActCnt, SplyCur, IssTotUSD
  ETH: TxCnt, AdrActCnt, SplyCur, IssTotUSD   (brak HashRate)
  Stablecoin supply: USDT + USDC SplyCur (USDT od 2017-08, USDC od 2018-12)
  Market dominance: proxy btc_eth_dominance = BTC_cap / (BTC_cap + ETH_cap)

Zapis: data/raw/onchain/{COIN}_onchain_daily.parquet, stablecoin_supply_daily.parquet, 
        market_dominance_daily.parquet

"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests


RAW_DIR = Path("data/raw/onchain")
COINMETRICS_BASE = "https://community-api.coinmetrics.io/v4"
COINS = ["BTC", 
         "ETH"
         ]
START = "2017-08-01"
END = "2026-01-01"

BTC_METRICS = [
    "HashRate",         # hash rate sieci (TH/s)
    "TxCnt",            # liczba transakcji dziennie
    "AdrActCnt",        # aktywne adresy
    "SplyCur",          # podaż w obiegu (BTC)
    "IssTotUSD"         # wartość wyemitowanych coinów (USD, proxy: miner revenue)
    ]

ETH_METRICS = [
    "TxCnt",            # liczba transakcji dziennie
    "AdrActCnt",        # aktywne adresy
    "SplyCur",          # podaż ETH
    "IssTotUSD"         # wartość wyemitowanych ETH (USD)
    ]

ASSET_MAP = {"BTC": "btc", 
             "ETH": "eth"
             }


class CoinMetricsCollector:
    """
    Klasa służy do pobierania metryk on-chain z Coin Metrics Community API.
    """

    def __init__(self, 
                 raw_dir: Path = RAW_DIR
                 ) -> None:
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, 
                           exist_ok=True
                           )
        self._session = requests.Session()

    def collect_all(
        self,
        coins: list[str] | None = None,
        start: str = START,
        end: str = END
    ) -> None:
        """Metoda pobierająca metryki on-chain dla wybranych coinów"""
        if coins is None:
            coins = COINS
        metrics_map = {"BTC": BTC_METRICS, 
                       "ETH": ETH_METRICS
                       }
        for coin in coins:
            asset = ASSET_MAP[coin]
            metrics = metrics_map[coin]
            df = self.collect_asset(asset, 
                                    metrics, 
                                    start=start, 
                                    end=end
                                    )
            self._save(df, 
                       f"{coin}_onchain_daily"
                       )

    def collect_asset(
        self,
        asset: str,
        metrics: list[str],
        start: str = START,
        end: str = END,
        page_size: int = 10_000
    ) -> pd.DataFrame:
        """
        Metoda pobiera metryki dla jednego assetu przez paginację.
        Coin Metrics zwraca max 10 000 rekordów na request.
        """
        all_rows: list[dict] = []
        next_page_token: str | None = None

        params: dict = {
            "assets": asset,
            "metrics": ",".join(metrics),
            "frequency": "1d",
            "start_time": start,
            "end_time": end,
            "page_size": page_size,
            "sort": "time"
        }

        while True:
            if next_page_token:
                params["next_page_token"] = next_page_token

            resp = self._session.get(
                f"{COINMETRICS_BASE}/timeseries/asset-metrics",
                params=params,
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()

            rows = data.get("data", [])
            all_rows.extend(rows)

            next_page_token = data.get("next_page_token")
            if not next_page_token or not rows:
                break
            time.sleep(0.2)

        if not all_rows:
            print(f"[CoinMetrics] {asset}: brak danych")
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        df["timestamp"] = pd.to_datetime(df["time"], 
                                         utc=True
                                         )
        df = df.drop(columns=["asset", 
                              "time"
                              ], 
                     errors="ignore")
        df = df.set_index("timestamp").sort_index()

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], 
                                    errors="coerce"
                                    )

        print(f"[CoinMetrics] {asset.upper()}: {len(df)} dni x {len(df.columns)} metryk "
              f"(od {df.index.min().date()} do {df.index.max().date()})")
        return df

    def collect_stablecoin_supply(
        self,
        start: str = START,
        end: str = END
        ) -> None:
        """
        Metoda pobiera dzienny supply USDT i USDC z CoinMetrics Community.
        """
        frames = {}
        for asset in ["usdt", 
                      "usdc"
                      ]:
            df = self.collect_asset(asset, 
                                    ["SplyCur"], 
                                    start=start, 
                                    end=end
                                    )
            if not df.empty:
                frames[asset] = df["SplyCur"].rename(f"{asset}_supply")

        if not frames:
            print("[CoinMetrics] stablecoin supply: brak danych")
            return

        combined = pd.concat(frames.values(), 
                             axis=1
                             )

        combined = combined.fillna(0.0)
        combined["total_stablecoin_supply"] = combined.sum(axis=1)

        self._save(combined, 
                   "stablecoin_supply_daily"
                   )

    def collect_market_dominance(
        self,
        start: str = START,
        end: str = END
        ) -> None:
        """
        Metoda pobiera BTC i ETH market cap z CoinMetrics Community.
        Liczy proxy BTC dominance = BTC_cap / (BTC_cap + ETH_cap).
        """
        frames = {}
        for asset in ["btc", 
                      "eth"
                      ]:
            df = self.collect_asset(asset, 
                                    ["CapMrktCurUSD"], 
                                    start=start, 
                                    end=end
                                    )
            if not df.empty:
                frames[asset] = df["CapMrktCurUSD"].rename(f"{asset}_market_cap_usd")

        if len(frames) < 2:
            print("[CoinMetrics] market dominance: brak danych")
            return

        combined = pd.concat(frames.values(), 
                             axis=1
                             )
        combined = combined.ffill()
        total = combined["btc_market_cap_usd"] + combined["eth_market_cap_usd"]
        combined["btc_eth_dominance"] = combined["btc_market_cap_usd"] / total

        self._save(combined, "market_dominance_daily")


    def _save(self, 
              df: pd.DataFrame, 
              name: str
              ) -> None:
        path = self.raw_dir / f"{name}.parquet"
        df.to_parquet(path)
        print(f"[CoinMetrics] zapisano -> {path}")

