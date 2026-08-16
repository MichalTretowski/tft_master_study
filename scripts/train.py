"""
Trening TFT / LSTM.

Wyniki:
  checkpoints/{experiment}/best_model.pt
  logs/{experiment}_history.json
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import json
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataloader import build_dataloaders
from src.models.tft.input_schema import TFTInputSchema
from src.training.trainer import Trainer
from src.utils.logger import get_logger

logger = get_logger("train")

DATASETS = Path("data/processed/datasets")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", 
                   choices=["tft", "lstm"], 
                   default="tft"
                   )
    p.add_argument("--coin", 
                   default="BTC", 
                   choices=["BTC", "ETH"]
                   )
    p.add_argument("--tf", 
                   default="1d", 
                   choices=["4h", "1d"]
                   )
    p.add_argument("--loss", 
                   default="dsr", 
                   choices=["bce", "dsr"]
                   )
    p.add_argument("--experiment", 
                   default=None
                   )

    p.add_argument("--epochs", 
                   type=int, 
                   default=100
                   )
    p.add_argument("--batch-size", 
                   type=int, 
                   default=64
                   )
    p.add_argument("--lr", 
                   type=float, 
                   default=1e-3
                   )
    p.add_argument("--weight-decay", 
                   type=float, 
                   default=1e-4
                   )

    p.add_argument("--hidden-size", 
                   type=int, 
                   default=128
                   )
    p.add_argument("--lstm-layers", 
                   type=int, 
                   default=2
                   )
    p.add_argument("--n-heads", 
                   type=int, 
                   default=4
                   )
    p.add_argument("--dropout", 
                   type=float, 
                   default=0.1
                   )
    p.add_argument("--encoder-days", 
                   type=int, 
                   default=30
                   )

    p.add_argument("--position-penalty", 
                   type=float, 
                   default=0.01
                   )
    p.add_argument("--transaction-cost", 
                   type=float, 
                   default=0.001
                   )
    p.add_argument("--patience", 
                   type=int, 
                   default=20
                   )
    p.add_argument("--es-metric",
                   default=None,
                   choices=["sharpe", "dir_acc", "val_loss"]
                   )
    p.add_argument("--es-min-delta",
                   type=float,
                   default=None
                   )
    p.add_argument("--min-position-std",
                   type=float,
                   default=0.02
                   )
    p.add_argument("--num-workers",
                   type=int,
                   default=0
                   )
    p.add_argument("--diag-every",
                   type=int,
                   default=50
                   )
    p.add_argument("--amp",
                   action="store_true"
                   )

    p.add_argument("--include", 
                   default=None,
                   help="grupy cech po przecinku (ablacja)"
                   )
    p.add_argument("--exclude", 
                   default=None
                   )

    p.add_argument("--seed", 
                   type=int, 
                   default=42
                   )

    p.add_argument("--device", 
                   default=None
                   )
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args, 
                schema
                ):
    if args.model == "tft":
        from src.models.tft.architecture import TemporalFusionTransformer

        return TemporalFusionTransformer(
            schema=schema,
            hidden_size=args.hidden_size,
            lstm_layers=args.lstm_layers,
            n_heads=args.n_heads,
            dropout=args.dropout
            )

    from src.models.baselines.lstm_baseline import VanillaLSTM

    return VanillaLSTM(
        input_size=len(schema.observed_reals),
        hidden_size=args.hidden_size,
        num_layers=args.lstm_layers,
        dropout=args.dropout
        )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    name = args.experiment or (
        f"{args.model}_{args.coin.lower()}_{args.tf}_{args.loss}"
        f"_h{args.hidden_size}"
        )
    bpd = 6 if args.tf == "4h" else 1

    cols = pd.read_parquet(
        DATASETS / args.tf / f"{args.coin}_train.parquet"
        ).columns

    schema = TFTInputSchema.from_columns(
        cols,
        bars_per_day=bpd,
        include=args.include.split(",") if args.include else None,
        exclude=args.exclude.split(",") if args.exclude else None,
        encoder_days=args.encoder_days
        )
    schema.summary()

    es_metric = args.es_metric or (
        "val_loss" if args.loss == "bce" else "sharpe"
        )

    loaders = build_dataloaders(
        coin=args.coin,
        schema=schema,
        tf=args.tf,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        loss_mode=args.loss
        )

    model = build_model(args, schema)
    n_par = sum(p.numel() for p in model.parameters())
    logger.info(f"[{name}] parametrow: {n_par:,}")

    meta = {
        "argv": sys.argv[1:],
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True
            ).stdout.strip(),
        "dirty": bool(subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True
            ).stdout.strip()),
        "parametrow": n_par,
        "observed": len(schema.observed_reals),
        "encoder_length": schema.encoder_length
        }
    meta_dir = Path("checkpoints") / name
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
        )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min" if es_metric == "val_loss" else "max",
        factor=0.5,
        patience=5,
        min_lr=1e-6
        )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_mode=args.loss,
        timeframe=args.tf,
        transaction_cost=args.transaction_cost,
        position_penalty=args.position_penalty,
        early_stopping_patience=args.patience,
        early_stopping_metric=es_metric,
        early_stopping_min_delta=args.es_min_delta,
        min_position_std=args.min_position_std,
        experiment_name=name,
        device=args.device,
        amp=args.amp,
        diag_every=args.diag_every
        )

    logger.info(
        f"[{name}] start: loss={args.loss}, tf={args.tf}, "
        f"batch={args.batch_size}, h={args.hidden_size}, "
        f"encoder={schema.encoder_length} barow"
        )

    trainer.fit(
        loaders["train"], loaders["val"], epochs=args.epochs
        )

    logger.info(f"[{name}] gotowe. Checkpoint w checkpoints/{name}/")


if __name__ == "__main__":
    main()
