"""
Sliding-window Dataset dla TFT i LSTM.

Kolumny brane wprost ze schematu.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DATASETS_DIR = Path("data/processed/datasets")

class Batch(NamedTuple):
    observed: torch.Tensor
    known_future: torch.Tensor
    static_cat: torch.Tensor
    target: torch.Tensor
    price_return: torch.Tensor


def _to_array(df: pd.DataFrame, 
              cols: list[str], 
              label: str
              ) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 0), 
                        dtype=np.float32
                        )

    sub = (df[cols].astype("float32")
           .replace([np.inf, -np.inf], np.nan).
           ffill()
        )
    n_nan = int(sub.isna().sum().sum())
    if n_nan:
        frac = sub.isna().mean().sort_values(ascending=False)
        top = ", ".join(
            f"{c} {v:.0%}" for c, v in frac.head(3).items() if v > 0
            )
        print(f"[Dataset] {label}: {n_nan} NaN po ffill -> 0 ({top})")

    return sub.fillna(0.0).to_numpy(dtype=np.float32).copy()


class CryptoDataset(Dataset):
    def __init__(
        self,
        coin: str,
        split: str,
        schema,
        scaler=None,
        known_scaler=None,
        tf: str = "4h",
        datasets_dir: Path = DATASETS_DIR,
        sanity_target: bool=False
        ) -> None:
        self.schema = schema
        self.scaler = scaler
        self.known_scaler = known_scaler
        self.coin_id = 0 if coin == "BTC" else 1

        path = datasets_dir / tf / f"{coin}_{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Brak {path}. Uruchom: "
                f"python scripts/preprocess.py --freq {tf}"
                )

        df = pd.read_parquet(path).sort_index()
        schema.validate(df.columns)

        tag = f"{coin}/{split}"
        self._observed_arr = _to_array(
            df, schema.observed_reals, f"{tag} observed"
            )
        self._known_arr = _to_array(
            df, schema.known_reals, f"{tag} known"
            )

        if scaler is not None:
            self._observed_arr = scaler.transform(
                self._observed_arr
            ).astype(np.float32)

        if known_scaler is not None:
            self._known_arr = known_scaler.transform(
                self._known_arr
            ).astype(np.float32)

        self._log_returns = (
            df["log_return"].astype("float32").fillna(0.0).to_numpy()
            )


        self._targets = (
            df[schema.target].astype("float32").fillna(-1.0).to_numpy()
            )

        if sanity_target:
            self._targets = (self._log_returns > 0).astype(np.float32)
            print(f"[Dataset] {tag}: SANITY TARGET (znak log_return)")

        self._enc = schema.encoder_length


        self._dec = schema.decoder_length
        self._T = len(df)

        self._indices = [
            i
            for i in range(self._enc, self._T - self._dec + 1)
            if self._targets[i + self._dec - 1] >= 0
            ]

        print(
            f"[Dataset] {tag}/{tf}: {len(self._indices)} okien "
            f"(enc={self._enc}, dec={self._dec}) z {self._T} wierszy"
            )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, 
                    idx: int
                    ) -> Batch:
        start = self._indices[idx]
        enc_start = start - self._enc + 1
        dec_end = start + self._dec

        return Batch(
            observed=torch.from_numpy(
                self._observed_arr[enc_start:start + 1]
                ),
            known_future=torch.from_numpy(
                self._known_arr[start:dec_end]
                ),
            static_cat=torch.tensor(self.coin_id, dtype=torch.long),
            target=torch.tensor(
                self._targets[dec_end - 1], dtype=torch.float32
                ),
            price_return=torch.tensor(
                float(self._log_returns[dec_end - 1]),
                dtype=torch.float32
                )
            )
