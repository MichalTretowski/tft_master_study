"""

Fear & Greed Index Collector

Źródło: alternative.me Crypto Fear & Greed Index
API: https://api.alternative.me/fng/
Dane od 2018-02-01, dzienny interwał.
Skala: 0 (extreme fear) - 100 (extreme greed).

"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path("data/raw/macro")
API_URL = "https://api.alternative.me/fng/"


class FearGreedCollector:
    def __init__(self, 
                 raw_dir: Path = RAW_DIR
                 ) -> None:
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, 
                           exist_ok=True
                           )

    def collect(self, 
                limit: int = 0
                ) -> pd.DataFrame:
        """
        Metoda pobiera historię Fear & Greed Index.
        """
        params = {"limit": limit, 
                  "format": "json"
                  }
        resp = requests.get(API_URL, 
                            params=params, 
                            timeout=30
                            )
        resp.raise_for_status()
        data = resp.json().get("data", [])

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), 
                                         unit="s", 
                                         utc=True
                                         )
        df["fear_greed_value"] = df["value"].astype(int)
        df["fear_greed_class"] = df["value_classification"]
        df = df[["timestamp", "fear_greed_value", "fear_greed_class"]]
        df = df.sort_values("timestamp").reset_index(drop=True)

        path = self.raw_dir / "fear_greed_index.parquet"
        df.to_parquet(path, 
                      index=False
                      )
        print(f"[Fear&Greed] {len(df)} dni -> {path}")
        return df

    def load(self) -> pd.DataFrame:
        return pd.read_parquet(self.raw_dir / "fear_greed_index.parquet")
