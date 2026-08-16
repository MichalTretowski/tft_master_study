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
        print(f"[Dataset] {label}: {n_nan} NaN po ffill "
              f"-> uzupelnione po skalowaniu ({top})")

    return sub.to_numpy(dtype=np.float32).copy()


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
        require_class_target: bool = True,
        keep_nans: bool = False,
        context_df: pd.DataFrame | None = None
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

        target_df = pd.read_parquet(path).sort_index()


        if context_df is not None:
            context_df = context_df.tail(schema.encoder_length).copy()
            df = pd.concat([context_df, target_df]).sort_index()
            self._first_prediction_index = len(context_df)
        else:
            df = target_df
            self._first_prediction_index = 0

        self._index = df.index
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


        if not keep_nans:
            self._observed_arr = np.nan_to_num(
                self._observed_arr, nan=0.0, posinf=0.0, neginf=0.0
                )
            self._known_arr = np.nan_to_num(
                self._known_arr, nan=0.0, posinf=0.0, neginf=0.0
                )


        self._forward_returns = (
            df["forward_log_return"].astype("float32").to_numpy()
            )

        self._targets = (
            df[schema.target].astype("float32").fillna(-1.0).to_numpy()
            )

        self._enc = schema.encoder_length
        self._dec = schema.decoder_length
        self._T = len(df)

        self._indices = [
            t
            for t in range(
                max(self._enc - 1, self._first_prediction_index),
                self._T - self._dec
                )
            if np.isfinite(self._forward_returns[t])
            and (not require_class_target or self._targets[t] >= 0)
            ]

        print(
            f"[Dataset] {tag}/{tf}: {len(self._indices)} okien "
            f"(enc={self._enc}, dec={self._dec}) z {self._T} wierszy "
            f"(kontekst: {self._first_prediction_index})"
            )

    def __len__(self) -> int:
        return len(self._indices)


    @property
    def window_timestamps(self) -> pd.DatetimeIndex:
        """Timestampy okien w kolejnosci, w jakiej wydaje je DataLoader."""
        return self._index[np.asarray(self._indices)]

    @property
    def window_targets(self) -> np.ndarray:
        """Targety okien w tej samej kolejnosci. Moga zawierac -1 przy DSR."""
        return self._targets[np.asarray(self._indices)]

    @property
    def class_targets(self) -> np.ndarray:
        t = self.window_targets
        return t[t >= 0]
    

    @property
    def class_prior(self) -> float:
        targets = self.class_targets

        if len(targets) == 0:
            raise ValueError(
                "Brak binarnych targetow do wyliczenia priora."
                )

        return float(targets.mean())



    
    def __getitem__(self,
                    idx: int
                    ) -> Batch:
        t = self._indices[idx]
        enc_start = t - self._enc + 1

        # Znane cechy naleza do prognozowanego kroku t+1.
        future_start = t + 1
        future_end = future_start + self._dec

        return Batch(
            observed=torch.from_numpy(
                self._observed_arr[enc_start:t + 1]
                ),
            known_future=torch.from_numpy(
                self._known_arr[future_start:future_end]
                ),
            static_cat=torch.tensor(self.coin_id, dtype=torch.long),
            target=torch.tensor(
                self._targets[t], dtype=torch.float32
                ),
            price_return=torch.tensor(
                float(self._forward_returns[t]),
                dtype=torch.float32
                )
            )