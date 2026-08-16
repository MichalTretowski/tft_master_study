"""

DataLoader: train/val/test dla jednego coina.

"""

from __future__ import annotations

import pickle
from hashlib import sha256
from pathlib import Path
from typing import TypedDict

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import DATASETS_DIR, CryptoDataset

SCALERS_DIR = Path("data/processed/scalers")


class DataLoaders(TypedDict):
    train: DataLoader
    val: DataLoader
    test: DataLoader

def _columns_fingerprint(columns) -> str:
    payload = "\x1f".join(columns).encode("utf-8")
    return sha256(payload).hexdigest()[:12]


def build_dataloaders(
    coin: str,
    schema,
    tf: str = "4h",
    batch_size: int = 64,
    num_workers: int = 0,
    datasets_dir: Path = DATASETS_DIR,
    scalers_dir: Path = SCALERS_DIR,
    pin_memory: bool = True,
    loss_mode: str = "bce"
    ) -> DataLoaders:
    scalers_dir.mkdir(parents=True, 
                      exist_ok=True
                      )
    obs_fp = _columns_fingerprint(schema.observed_reals)
    known_fp = _columns_fingerprint(schema.known_reals)
    scaler_path = scalers_dir / f"{coin}_{tf}_obs_{obs_fp}.pkl"
    known_path = scalers_dir / f"{coin}_{tf}_known_{known_fp}.pkl"

    if scaler_path.exists() and known_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(known_path, "rb") as f:
            known_scaler = pickle.load(f)
        print(f"[DataLoader] Scaler z {scaler_path}")
    else:
        from sklearn.preprocessing import RobustScaler

        train_raw = CryptoDataset(
            coin=coin,
            split="train",
            schema=schema,
            scaler=None,
            known_scaler=None,
            tf=tf,
            datasets_dir=datasets_dir,
            keep_nans=True
            )
        
        scaler = RobustScaler(quantile_range=(5, 95))
        scaler.fit(train_raw._observed_arr)

        known_scaler = RobustScaler(quantile_range=(5, 95))
        known_scaler.fit(train_raw._known_arr)

        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        with open(known_path, "wb") as f:
            pickle.dump(known_scaler, f)
        print(f"[DataLoader] Scaler dopasowany -> {scalers_dir}")

    contexts = {
        "train": None,
        "val": pd.read_parquet(
            datasets_dir / tf / f"{coin}_train.parquet"
            ).sort_index().tail(schema.encoder_length),
        "test": pd.read_parquet(
            datasets_dir / tf / f"{coin}_val.parquet"
            ).sort_index().tail(schema.encoder_length)
        }

    require_class_target = loss_mode == "bce"

    datasets = {}
    for split in ("train", "val", "test"):
        datasets[split] = CryptoDataset(
            coin=coin,
            split=split,
            schema=schema,
            scaler=scaler,
            known_scaler=known_scaler,
            tf=tf,
            datasets_dir=datasets_dir,
            require_class_target=require_class_target,
            context_df=contexts[split]
            )

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory and torch.cuda.is_available()
        }
    if num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=2
            )

    loaders: DataLoaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            **loader_kwargs
            ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs
            ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs
            )
            }

    for split, loader in loaders.items():
        print(
            f"[DataLoader] {coin}/{split}: {len(datasets[split])} okien, "
            f"{len(loader)} batchy (batch={batch_size})"
            )

    return loaders
