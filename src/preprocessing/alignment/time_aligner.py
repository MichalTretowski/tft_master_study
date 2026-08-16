"""

Time Aligner: wyrównanie wszystkich źródeł danych do wspólnego inerwału 4h.

Wszystkie źródła danych mają różne:
  - interwały (1h, 4h, daily, weekly, monthly)
  - zakresy dat (np. Binance Futures od 2019, Fear&Greed od 2018)
  - brakujące wartości (weekendy w danych notowań klasycznych instrumentów, 
  przerwy giełdowe)

Wyjście: jeden DataFrame w interwale 4h,
Tam gdzie nie ma danych: NaN

"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed/datasets")

PROJECT_START = "2017-08-01"
PROJECT_END = "2026-01-01"

SPLIT_DATES = {
    "train_start": "2017-08-01",
    "train_end": "2022-12-31",
    "val_start": "2023-01-01",
    "val_end": "2024-06-30",
    "test_start": "2024-07-01",
    "test_end": "2025-12-31"
    }

COINS = ["BTC", 
         "ETH"
         ]


class AlignedDataset(NamedTuple):
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    feature_cols: list[str]
    target_col: str


class TimeAligner:
    def __init__(self, 
                 processed_dir: Path = PROCESSED_DIR
                 ) -> None:
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)


    def build_master_index(
        self,
        start: str = PROJECT_START,
        end: str = PROJECT_END,
        freq: str = "4h"
        ) -> pd.DatetimeIndex:
        """Metoda tworzy interwał 4h dla całego projektu."""
        return pd.date_range(start=start, 
                             end=end, 
                             freq=freq, 
                             tz="UTC"
                             )

    def align_sources(
        self,
        sources: dict[str, pd.DataFrame],
        master_index: pd.DatetimeIndex,
        fill_strategies: dict[str, str] | None = None
        ) -> pd.DataFrame:
        """
        Metoda wyrównuje wiele DataFrame'ów do wspólnego interwału.

        Parametry:
            sources:         słownik {nazwa: DataFrame}
            master_index:    docelowy interwał / indeks 4h
            fill_strategies: {nazwa_kolumny: strategia} — 'ffill', 'zero', 'interpolate'
                             Domyślnie 'ffill' dla wszystkich.

        Zwraca: jeden scalony DataFrame z master_index.
        """
        fill_strategies = fill_strategies or {}
        aligned_frames: list[pd.DataFrame] = []

        for source_name, df in sources.items():
            df = df.copy()
            df.index = pd.to_datetime(df.index, 
                                      utc=True
                                      )
            df = df[~df.index.duplicated(keep="last")]


            df = df.reindex(master_index, 
                            method=None
                            )

            for col in df.columns:
                strategy = fill_strategies.get(col, 
                                               "ffill"
                                               )
                if strategy == "ffill":
                    df[col] = df[col].ffill()
                elif strategy == "zero":
                    df[col] = df[col].fillna(0.0)
                elif strategy == "interpolate":
                    df[col] = df[col].interpolate(method="linear", 
                                                  limit_direction="forward").ffill()

            print(f"[Aligner] {source_name}: {df.notna().sum().sum()} wartości, "
                  f"{df.isna().sum().sum()} NaN")
            aligned_frames.append(df)

        if not aligned_frames:
            return pd.DataFrame(index=master_index)

        combined = pd.concat(aligned_frames, 
                             axis=1
                             )
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined

    def build_coin_dataset(
        self,
        aligned_df: pd.DataFrame,
        coin: str,
        target_col: str,
        drop_na_subset: list[str] | None = None,
        purge_bars: int = 1
        ) -> AlignedDataset:
        """
        Metoda buduje finalny dataset dla jednego coina z podziałem train/val/test.
        Dodaje kolumnę static 'coin_id'.
        """
        df = aligned_df.copy()
        df["coin_id"] = 0 if coin == "BTC" else 1

        if drop_na_subset:
            df = df.dropna(subset=drop_na_subset)

        feature_cols = [
            c for c in df.columns
            if c not in {target_col, "forward_log_return"}
            ]

        def split_with_purge(start: str, end: str) -> pd.DataFrame:
            out = df.loc[start:end].copy()

            # Target t uzywa ceny po horyzoncie — nie moze przekraczac splitu.
            if purge_bars > 0 and len(out) >= purge_bars:
                tail = out.index[-purge_bars:]
                out.loc[tail, target_col] = pd.NA
                out.loc[tail, "forward_log_return"] = float("nan")

            return out

        train = split_with_purge(
            SPLIT_DATES["train_start"], SPLIT_DATES["train_end"]
            )
        val = split_with_purge(
            SPLIT_DATES["val_start"], SPLIT_DATES["val_end"]
            )
        test = split_with_purge(
            SPLIT_DATES["test_start"], SPLIT_DATES["test_end"]
            )

        print(f"[Aligner] {coin} dataset: "
              f"train={len(train)}, val={len(val)}, test={len(test)} świec 4h | "
              f"{len(feature_cols)} features")

        for split_name, split_df in [("train", train), 
                                     ("val", val), 
                                     ("test", test)
                                     ]:
            path = self.processed_dir / f"{coin}_{split_name}.parquet"
            split_df.to_parquet(path)

        return AlignedDataset(
            train=train,
            val=val,
            test=test,
            feature_cols=feature_cols,
            target_col=target_col
            )

    def add_target(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        horizon_bars: int = 6,
        dead_zone_pct: float = 0.003
        ) -> pd.DataFrame:
        """
        Metoda dodaje kolumnę target: kierunek ceny za horizon_bars świec 4h.

        Target:
            1  = cena wzrośnie o więcej niż dead_zone_pct
            0  = cena spadnie o więcej niż dead_zone_pct
            NaN = zmiana w dead zone (opcjonalnie filtrować)

        dead_zone_pct: minimalna zmiana żeby oznaczyć sygnał (default 0.3%).
                       Redukuje szum w płaskich rynkach.
        """
        df = df.copy()
        df["forward_log_return"] = np.log(
            df[price_col].shift(-horizon_bars) / df[price_col]
            )
        future_return = np.expm1(df["forward_log_return"])

        df["target"] = pd.NA
        df.loc[future_return > dead_zone_pct, "target"] = 1
        df.loc[future_return < -dead_zone_pct, "target"] = 0
        df["target"] = df["target"].astype("Int64")

        n_long = (df["target"] == 1).sum()
        n_short = (df["target"] == 0).sum()
        n_neutral = df["target"].isna().sum()
        print(f"[Target] Long: {n_long} | Short: {n_short} | Neutral/NaN: {n_neutral}")

        return df
