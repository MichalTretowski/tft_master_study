"""

DataLoader: train/val/test dla jednego coina.

"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TypedDict

import torch
from torch.utils.data import DataLoader

from src.data.dataset import DATASETS_DIR, CryptoDataset

SCALERS_DIR = Path("data/processed/scalers")


class DataLoaders(TypedDict):
    train: DataLoader
    val: DataLoader
    test: DataLoader


def build_dataloaders(
    coin: str,
    schema,
    tf: str = "4h",
    batch_size: int = 64,
    num_workers: int = 0,
    datasets_dir: Path = DATASETS_DIR,
    scalers_dir: Path = SCALERS_DIR,
    pin_memory: bool = True,
    sanity_target: bool = False
    ) -> DataLoaders:
    scalers_dir.mkdir(parents=True, 
                      exist_ok=True
                      )
    n_obs = len(schema.observed_reals)
    n_known = len(schema.known_reals)
    scaler_path = scalers_dir / f"{coin}_{tf}_obs{n_obs}.pkl"
    known_path = scalers_dir / f"{coin}_{tf}_known{n_known}.pkl"

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
            sanity_target=sanity_target
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

    datasets = {
        split: CryptoDataset(
            coin=coin, 
            split=split, 
            schema=schema,
            scaler=scaler, 
            known_scaler=known_scaler,
            tf=tf, 
            datasets_dir=datasets_dir,
            sanity_target=sanity_target
            )
        for split in ("train", "val", "test")
        }

    use_pin = pin_memory and torch.cuda.is_available()

    loaders: DataLoaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin,
            drop_last=True
            ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin
            ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin
            )
            }

    for split, loader in loaders.items():
        print(
            f"[DataLoader] {coin}/{split}: {len(datasets[split])} okien, "
            f"{len(loader)} batchy (batch={batch_size})"
            )

    return loaders
