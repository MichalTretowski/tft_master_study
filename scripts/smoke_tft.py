"""

Smoke test TFT: wymiary, liczba parametrow, zuzycie VRAM.

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataloader import build_dataloaders
from src.models.tft.architecture import TemporalFusionTransformer
from src.models.tft.input_schema import TFTInputSchema


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tf", 
                   default="1d", 
                   choices=["4h", "1d"]
                   )
    p.add_argument("--coin", 
                   default="BTC"
                   )
    p.add_argument("--batch", 
                   type=int, 
                   default=64
                   )
    p.add_argument("--hidden", 
                   type=int, 
                   default=128
                   )
    p.add_argument("--mode", 
                   default="position",
                   choices=["position", "probability"]
                   )
    args = p.parse_args()

    bpd = 6 if args.tf == "4h" else 1
    path = (Path("data/processed/datasets") / args.tf
            / f"{args.coin}_train.parquet")
    cols = pd.read_parquet(path).columns

    schema = TFTInputSchema.from_columns(cols, 
                                         bars_per_day=bpd
                                         )
    schema.summary()

    loaders = build_dataloaders(
        args.coin, 
        schema, 
        tf=args.tf, 
        batch_size=args.batch
        )
    batch = next(iter(loaders["train"]))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = TemporalFusionTransformer(
        schema=schema,
        hidden_size=args.hidden,
        output_mode=args.mode
        ).to(dev)

    n_par = sum(p.numel() for p in model.parameters())
    print("")
    print(f"[Model] parametrow: {n_par:,}  ({n_par * 4 / 1e6:.0f} MB fp32)")

    obs = batch.observed.to(dev)
    known = batch.known_future.to(dev)
    static = batch.static_cat.to(dev)

    print(f"[Wejscie] observed {tuple(obs.shape)}")
    print(f"[Wejscie] known_future {tuple(known.shape)}")
    print(f"[Wejscie] static_cat {tuple(static.shape)}")

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    out = model(obs, 
                known, 
                static
                )

    print("")
    print(f"[Wyjscie] output {tuple(out['output'].shape)}")
    print(f"[Wyjscie] encoder_weights {tuple(out['encoder_weights'].shape)}")
    print(f"[Wyjscie] attention_weights "
          f"{tuple(out['attention_weights'].shape)}")
    print(f"[Wyjscie] zakres output: {out['output'].min():.3f} "
          f".. {out['output'].max():.3f}")

    loss = out["output"].mean()
    loss.backward()
    print("[Backward] OK")

    if dev == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print("")
        print(f"[VRAM] szczyt {peak:.2f} GB / {total:.1f} GB")


if __name__ == "__main__":
    main()
