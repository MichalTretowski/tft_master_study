"""

Google Trends Collector - zainteresowanie wyszukiwaniami krypto

Źródło: pytrends (nieoficjalny wrapper Google Trends)

3 grupy słów kluczowych - BTC / ETH / ogólne
Interwał tygodniowy
Zakres: 2017–2025, pobierane chunkami rocznymi.

Ograniczenia:
  - Normalizacja 0-100 liczona osobno na chunk (u nas rok) → wartości NIE są porównywalne
    między chunkami.
  - Sleep między requestami.

Zapis: data/raw/macro/google_trends.parquet + google_trends_{btc,eth,general}.parquet

"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from pytrends.request import TrendReq

RAW_DIR = Path("data/raw/macro")
START = "2017-08-01"
END = "2026-01-01"


KEYWORDS: dict[str, list[str]] = {
    "BTC": ["bitcoin", "buy bitcoin", "bitcoin price", "bitcoin crash"],
    "ETH": ["ethereum", "buy ethereum", "ethereum price"],
    "GENERAL": ["crypto", "cryptocurrency", "blockchain"],
    }


class GoogleTrendsCollector:
    def __init__(self, 
                 raw_dir: Path = RAW_DIR, 
                 hl: str = "en-US", 
                 tz: int = 0
                 ) -> None:
        self.pytrends = TrendReq(hl=hl, 
                                 tz=tz, 
                                 timeout=(10, 30)
                                 )
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, 
                           exist_ok=True
                           )

    def collect_all(self, 
                    start: str = START, 
                    end: str = END
                    ) -> pd.DataFrame:
        """Metoda pobiera trendy dla wszystkich słów kluczowych przez chunking roczny."""
        all_frames: list[pd.DataFrame] = []

        for group_name, kw_list in KEYWORDS.items():
            df = self._collect_keywords(kw_list, 
                                        group_name, 
                                        start=start, 
                                        end=end
                                        )
            if df is not None:
                all_frames.append(df)

        if not all_frames:
            return pd.DataFrame()

        combined = pd.concat(all_frames, 
                             axis=1
                             )
        combined = combined[~combined.index.duplicated(keep="first")].sort_index()

        path = self.raw_dir / "google_trends.parquet"
        combined.to_parquet(path)
        print(f"[GoogleTrends] zapisano {len(combined.columns)} słów kluczowych -> {path}")
        return combined

    def _collect_keywords(
        self,
        keywords: list[str],
        group_name: str,
        start: str,
        end: str,
    ) -> pd.DataFrame | None:
        """Metoda pobiera dane dla grupy słów kluczowych przez chunking roczny."""
        periods = pd.date_range(start=start, 
                                end=end, 
                                freq="YS")
        chunks: list[pd.DataFrame] = []

        for i in range(len(periods) - 1):
            chunk_start = str(periods[i].date())
            chunk_end = str(periods[i + 1].date())
            timeframe = f"{chunk_start} {chunk_end}"

            try:
                self.pytrends.build_payload(keywords[:5], 
                                            timeframe=timeframe
                                            )  # max 5 keywords
                df_chunk = self.pytrends.interest_over_time()
                if not df_chunk.empty:
                    df_chunk = df_chunk.drop(columns=["isPartial"], 
                                             errors="ignore"
                                             )
                    chunks.append(df_chunk)
                time.sleep(2.0)
            except Exception as e:
                print(f"[GoogleTrends] {group_name} {chunk_start}: błąd — {e}")
                time.sleep(10)

        if not chunks:
            return None

        combined = pd.concat(chunks).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.index = pd.to_datetime(combined.index, 
                                        utc=True
                                        )

        path = self.raw_dir / f"google_trends_{group_name.lower()}.parquet"
        combined.to_parquet(path)
        print(f"[GoogleTrends] {group_name}: {len(combined)} tygodni -> {path}")
        return combined

    def load(self) -> pd.DataFrame:
        return pd.read_parquet(self.raw_dir / "google_trends.parquet")
