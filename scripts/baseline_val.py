"""

Dokladny baseline walidacyjny i przewaga checkpointu.

Baseline = staly predyktor z czestosci klasy liczonej na oknach treningowych,
oceniony na dokladnie tych oknach walidacyjnych, ktore widzi model.

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataloader import build_dataloaders
from src.models.tft.input_schema import TFTInputSchema

DATASETS = Path("data/processed/datasets")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--coin",
                   default="BTC",
                   choices=["BTC", "ETH"]
                   )
    p.add_argument("--tf",
                   default="1d",
                   choices=["4h", "1d"],
                   help="ignorowane, gdy podano --experiment"
                   )
    p.add_argument("--include",
                   default="ohlcv",
                   help="ignorowane, gdy podano --experiment"
                   )
    p.add_argument("--experiment",
                   default=None
                   )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ck = None
    if args.experiment:
        ck_path = (
            Path("checkpoints") / args.experiment / "best_model.pt"
            )
        if not ck_path.exists():
            raise FileNotFoundError(f"Brak checkpointu: {ck_path}")

        ck = torch.load(ck_path,
                        map_location="cpu",
                        weights_only=False
                        )

        if ck.get("schema") is None:
            raise ValueError(
                "Checkpoint bez schematu - pochodzi sprzed poprawki."
                )
        if ck["loss_mode"] != "bce":
            raise ValueError(
                "baseline_val.py dotyczy tylko checkpointow BCE."
                )

        schema = TFTInputSchema(**ck["schema"])
        tf = ck["timeframe"]
    else:
        tf = args.tf
        bpd = 6 if tf == "4h" else 1
        cols = pd.read_parquet(
            DATASETS / tf / f"{args.coin}_train.parquet"
            ).columns
        schema = TFTInputSchema.from_columns(
            cols,
            bars_per_day=bpd,
            include=args.include.split(",") if args.include else None
            )

    loaders = build_dataloaders(
        coin=args.coin,
        schema=schema,
        tf=tf,
        batch_size=256,
        loss_mode="bce"
        )

    p_train = float(
        np.clip(loaders["train"].dataset.class_prior, 1e-6, 1 - 1e-6)
        )
    y = loaders["val"].dataset.class_targets

    baseline = float(
        -(y * np.log(p_train) + (1 - y) * np.log(1 - p_train)).mean()
        )

    print("")
    print(f"  coin/tf            {args.coin} / {tf}")
    print(f"  observed           {len(schema.observed_reals)}")
    print(f"  okien walidacji    {len(y)}")
    print(f"  p_train (okna)     {p_train:.6f}")
    print(f"  p_val   (okna)     {float(y.mean()):.6f}")
    print(f"  baseline_val_loss  {baseline:.6f}")

    if ck is not None:
        print("")
        print(f"  {args.experiment}")
        print(f"    epoka          {ck['epoch']}")
        print(f"    val_loss       {ck['val_loss']:.6f}")
        print(f"    przewaga       {baseline - ck['val_loss']:+.6f}")
        print(f"    val_dir_acc    {ck['val_dir_acc']:.4f}")


if __name__ == "__main__":
    main()
