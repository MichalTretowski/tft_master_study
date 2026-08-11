"""

Macro Preprocessor - resampling i interpolacja danych makro do interwału 4h.

Dane makro mają różne częstotliwości:
  daily:     Fear&Greed, yields, DXY
  weekly:    COT, Fed balance sheet, Google Trends
  monthly:   CPI, M1/M2, PCE, NFP, unemployment
  quarterly: GDP

Strategia: forward-fill + interpolacja liniowa.

"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

PROCESSED_DIR = Path("data/processed/features")
InterpolationMethod = Literal["ffill", 
                              "linear", 
                              "cubic"
                              ]


class MacroPreprocessor:
    def __init__(self, 
                 processed_dir: Path = PROCESSED_DIR,
                 bars_per_day: int = 6,
                 timeframe: str = "4h"
                 ) -> None:
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, 
                                 exist_ok=True
                                 )
        self.bars_per_day = bars_per_day
        self.timeframe = timeframe

    def resample_to_target(
        self,
        df: pd.DataFrame,
        method: InterpolationMethod = "ffill",
        target_index: pd.DatetimeIndex | None = None
        ) -> pd.DataFrame:
        """
        Metoda resampluje dane makro do wybranego interwału
        """
        df = df.copy()
        df.index = pd.to_datetime(df.index, 
                                  utc=True
                                  )
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        if target_index is not None:
            df = df.reindex(df.index.union(target_index))
        else:
            new_index = pd.date_range(
                start=df.index.min().floor(self.timeframe),
                end=df.index.max().ceil(self.timeframe),
                freq=self.timeframe,
                tz="UTC",
            )
            df = df.reindex(df.index.union(new_index))

        if method == "ffill":
            df = df.ffill()
        elif method in ("linear", 
                        "cubic"
                        ):
            df = df.interpolate(method=method, 
                                limit_direction="forward"
                                )
            df = df.ffill() 
        else:
            raise ValueError(f"Nieznana metoda: {method}")

        if target_index is not None:
            df = df.reindex(target_index)

        return df

    def add_macro_change_features(self, 
                                  df: pd.DataFrame
                                  ) -> pd.DataFrame:
        """
        Metoda dodaje cechy pochodne dla serii makro:
          - zmiana MoM / WoW dla wolnozmiennych serii
          - odchylenie od długoterminowej średniej (z-score 12M)
        """
        df = df.copy()
        new_cols: dict[str, 
                       pd.Series
                       ] = {}

        bpd = self.bars_per_day
        shift_30d = 30 * bpd
        win_12m = 360 * bpd

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                new_cols[f"{col}_mom_30d"] = df[col].pct_change(shift_30d)
                roll_mean = df[col].rolling(win_12m, 
                                            min_periods=30
                                            ).mean()
                roll_std = df[col].rolling(win_12m, 
                                           min_periods=30
                                           ).std()
                new_cols[f"{col}_zscore_12m"] = (df[col] - roll_mean) / roll_std.replace(0, pd.NA)

        for name, series in new_cols.items():
            df[name] = series

        return df

    def process_and_save(
        self,
        df: pd.DataFrame,
        name: str,
        target_index: pd.DatetimeIndex | None = None,
        method: InterpolationMethod = "ffill",
        add_derived: bool = True
        ) -> pd.DataFrame:
        df = self.resample_to_target(df, 
                                 method=method, 
                                 target_index=target_index
                                 )
        if add_derived:
            df = self.add_macro_change_features(df)
        df = df.dropna(how="all")

        path = self.processed_dir / f"{name}_{self.timeframe}.parquet"
        df.to_parquet(path)
        print(f"[Macro] {name}: {len(df)} wierszy {self.timeframe} -> {path}")
        return df
