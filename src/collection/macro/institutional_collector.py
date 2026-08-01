"""

COT Collector - Commitment of Traders z CFTC dla CME Bitcoin Futures.
CFTC Commitment of Traders - tygodniowy raport pozycji spekulantów na CME Bitcoin Futures.

Źródło: CFTC Public Reporting API (Socrata)
  https://publicreporting.cftc.gov/resource/6dca-aqww.json
Kontrakt: CME Bitcoin Futures, kod 133741 (tylko BTC - brak ETH)
Częstotliwość: tygodniowa

Zbierane kolumny:
  noncomm_positions_long/short_all: pozycje spekulantów (niekomercyjni)
  comm_positions_long/short_all: pozycje hedgerów (komercyjni)
  open_interest_all: łączne otwarte pozycje
  net_speculator_position (derived): long - short spekulantów

Zapis: data/raw/institutional/cot_btc_futures.parquet

"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path("data/raw/institutional")
START = "2017-01-01"

CFTC_BTC_CODE = "133741"
CFTC_API = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"


class COTCollector:


    def __init__(self, 
                 raw_dir: Path = RAW_DIR
                 ) -> None:
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, 
                           exist_ok=True
                           )

    def collect(self, 
                start: str = START, 
                limit: int = 5000
                ) -> pd.DataFrame:
        """Metoda pobiera pełną historię COT dla Bitcoin Futures"""
        params = {
            "$where": f"cftc_contract_market_code='{CFTC_BTC_CODE}' AND report_date_as_yyyy_mm_dd>='{start}'",
            "$limit": limit,
            "$order": "report_date_as_yyyy_mm_dd ASC",
        }
        resp = requests.get(CFTC_API, 
                            params=params, 
                            timeout=60
                            )
        resp.raise_for_status()
        data = resp.json()

        df = pd.DataFrame(data)
        if df.empty:
            print("[COT] Brak danych")
            return df

        df["timestamp"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"], 
                                         utc=True
                                         )
        df = df.set_index("timestamp").sort_index()

        # Kolumny: pozycje long/short
        keep_cols = [
            "noncomm_positions_long_all",    # spekulanci long
            "noncomm_positions_short_all",   # spekulanci short
            "comm_positions_long_all",       # hedgerzy long
            "comm_positions_short_all",      # hedgerzy short
            "open_interest_all"
            ]
        available = [c for c in keep_cols if c in df.columns]
        df = df[available].astype(float)

        # Derived: net spekulant position
        if "noncomm_positions_long_all" in df and "noncomm_positions_short_all" in df:
            df["net_speculator_position"] = (
                df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]
                )

        path = self.raw_dir / "cot_btc_futures.parquet"
        df.to_parquet(path)
        print(f"[COT] BTC Futures: {len(df)} tygodni -> {path}")
        return df
