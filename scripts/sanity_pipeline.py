"""

Sonda hydrauliki: czy dane docieraja do glowicy i czy gradient wraca.

acc ~ 1.0  => pipeline zdrowy
acc ~ baza => blad w kodzie, nie brak sygnalu

"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataloader import build_dataloaders
from src.models.tft.architecture import TemporalFusionTransformer
from src.models.tft.input_schema import TFTInputSchema
from src.training.trainer import Trainer

DATASETS = Path("data/processed/datasets")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--coin", 
                   default="BTC", 
                   choices=["BTC", "ETH"]
                   )
    p.add_argument("--tf", 
                   default="1d", 
                   choices=["4h", "1d"]
                   )
    p.add_argument("--epochs", 
                   type=int, 
                   default=5
                   )
    p.add_argument("--hidden-size", 
                   type=int, 
                   default=32
                   )
    p.add_argument("--signal-col", 
                   default="log_return"
                   )
    args = p.parse_args()

    bpd = 6 if args.tf == "4h" else 1
    cols = pd.read_parquet(
        DATASETS / args.tf / f"{args.coin}_train.parquet"
        ).columns

    schema = TFTInputSchema.from_columns(
        cols, bars_per_day=bpd, include=["ohlcv"]
        )
    schema.summary()

    if args.signal_col not in schema.observed_reals:
        raise ValueError(
            f"{args.signal_col} nie jest kolumna obserwowana. "
            f"Dostepne: {schema.observed_reals[:10]}"
            )


    loaders = build_dataloaders(
        coin=args.coin,
        schema=schema,
        tf=args.tf,
        batch_size=64,
        loss_mode="dsr"
        )

    col = schema.observed_reals.index(args.signal_col)

    for split, loader in loaders.items():
        ds = loader.dataset
        ds._targets = (ds._observed_arr[:, col] > 0).astype(np.float32)
        print(f"[Sanity] {split}: cel = znak {args.signal_col}")

    model = TemporalFusionTransformer(
        schema=schema,
        hidden_size=args.hidden_size,
        lstm_layers=1
        )

    trainer = Trainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        loss_mode="bce",
        timeframe=args.tf,
        early_stopping_patience=args.epochs,
        experiment_name="sanity",
        checkpoint_dir=Path(tempfile.mkdtemp(prefix="tft_sanity_"))
        )

    trainer.fit(loaders["train"], loaders["val"], epochs=args.epochs)

    print("")
    print("acc bliskie 1.0 => pipeline zdrowy.")
    print("acc bliskie czestosci bazowej => blad w kodzie.")


if __name__ == "__main__":
    main()