"""

FRED Collector - dane makroekonomiczne z Federal Reserve Economic Data.

Źródło: FRED API.
Rejestracja: https://fred.stlouisfed.org/docs/api/api_key.html
Biblioteka: fredapi

Zbierane serie:
  Polityka monetarna:   fed_funds_rate (FEDFUNDS, m), 
                        fed_balance_sheet (WALCL, w),
                        sofr (SOFR, d)
  Agregaty pieniężne:   m1_usa (M1SL, m), 
                        m2_usa (M2SL, m)
  Inflacja:             cpi_all (CPIAUCSL, m), 
                        cpi_core (CPILFESL, m),
                        pce (PCEPI, m), 
                        ppi (PPIACO, m)
  Rynek pracy:          unemployment_rate (UNRATE, m), 
                        nonfarm_payrolls (PAYEMS, m)
  Wzrost / sentyment:   real_gdp (GDPC1, q), 
                        consumer_sentiment (UMCSENT, m),
                        ism_manufacturing (MANEMP, m)
  Krzywa dochodowości:  yield_10y (DGS10, d), 
                        yield_2y (DGS2, d),
                        yield_spread_10y_2y (T10Y2Y, d),
                        yield_spread_10y_3m (T10Y3M, d - recession indicator)
  Płynność globalna:    tga_balance (WTREGEN, w), 
                        rrp_overnight (RRPONTSYD, d)

Częstotliwość w nawiasie: 
  d=dzienna, 
  w=tygodniowa, 
  m=miesięczna, 
  q=kwartalna

Zapis: data/raw/macro/fred_macro_raw.parquet

"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fredapi import Fred

RAW_DIR = Path("data/raw/macro")
START = "2017-01-01"
END = "2026-01-01"

FRED_SERIES: dict[str, 
                  str
                  ] = {
    # Polityka monetarna USA
    "fed_funds_rate": "FEDFUNDS",          # stopa funduszy federalnych (monthly)
    "fed_balance_sheet": "WALCL",          # bilans Fed w mld USD (weekly)
    "sofr": "SOFR",                        # SOFR (daily)

    # Agregaty pieniężne USA
    "m1_usa": "M1SL",                      # M1 (monthly)
    "m2_usa": "M2SL",                      # M2 (monthly)

    # Inflacja USA
    "cpi_all": "CPIAUCSL",                 # CPI All Items (monthly)
    "cpi_core": "CPILFESL",                # Core CPI (monthly)
    "pce": "PCEPI",                        # PCE (monthly)
    "ppi": "PPIACO",                       # PPI (monthly)

    # Rynek pracy USA
    "unemployment_rate": "UNRATE",         # stopa bezrobocia (monthly)
    "nonfarm_payrolls": "PAYEMS",          # Non-Farm Payrolls (monthly)

    # Wzrost i sentyment USA
    "real_gdp": "GDPC1",                   # Real GDP (quarterly)
    "ism_manufacturing": "MANEMP",         # proxy koniunktury przemysłowej oparty na zatrudnieniu (monthly)
    "consumer_sentiment": "UMCSENT",       # Michigan Consumer Sentiment (monthly)

    # Krzywa dochodowości USA
    "yield_10y": "DGS10",                  # 10Y Treasury (daily)
    "yield_2y": "DGS2",                    # 2Y Treasury (daily)
    "yield_spread_10y_2y": "T10Y2Y",       # 10Y-2Y spread (daily)
    "yield_spread_10y_3m": "T10Y3M",       # 10Y-3M spread (daily, recession indicator)

    # Płynność globalna 
    "tga_balance": "WTREGEN",              # Treasury General Account (weekly)
    "rrp_overnight": "RRPONTSYD",          # Reverse Repo overnight (daily)
    }


class FREDCollector:
    def __init__(self, 
                 api_key: str, 
                 raw_dir: Path = RAW_DIR
                 ) -> None:
        """api_key: klucz z fred.stlouisfed.org/docs/api/api_key.html"""
        self.fred = Fred(api_key=api_key)
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, 
                           exist_ok=True
                           )

    def collect_all(
        self,
        start: str = START,
        end: str = END
    ) -> pd.DataFrame:
        """
        Metoda pobiera wszystkie serie i scala je w jeden DataFrame (index = date)
        """
        frames: dict[str, 
                     pd.Series
                     ] = {}

        for name, series_id in FRED_SERIES.items():
            try:
                s = self.fred.get_series(series_id, 
                                         observation_start=start, 
                                         observation_end=end
                                         )
                s.name = name
                frames[name] = s
                print(f"[FRED] {name} ({series_id}): {len(s)} obserwacji")
            except Exception as e:
                print(f"[FRED] {name} ({series_id}): błąd — {e}")

        if not frames:
            return pd.DataFrame()

        combined = pd.DataFrame(frames)
        combined.index = pd.to_datetime(combined.index, 
                                        utc=True
                                        )
        combined.index.name = "timestamp"

        path = self.raw_dir / "fred_macro_raw.parquet"
        combined.to_parquet(path)
        print(f"[FRED] zapisano {len(combined.columns)} serii -> {path}")
        return combined

    def collect_series(self, 
                       name: str, 
                       start: str = START, 
                       end: str = END
                       ) -> pd.Series:
        """Metoda pobiera pojedynczą serię FRED"""
        series_id = FRED_SERIES[name]
        s = self.fred.get_series(series_id, 
                                 observation_start=start, 
                                 observation_end=end
                                 )
        s.name = name
        return s

    def load(self) -> pd.DataFrame:
        return pd.read_parquet(self.raw_dir / "fred_macro_raw.parquet")
