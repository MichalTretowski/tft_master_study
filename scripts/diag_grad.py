"""
Diagnostyka: czy wyjscie modelu zalezy od wejscia observed
i czy gradient tam dochodzi.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataloader import build_dataloaders
from src.models.tft.architecture import TemporalFusionTransformer
from src.models.tft.input_schema import TFTInputSchema

TF = "4h"
COIN = "BTC"

cols = pd.read_parquet(
    Path("data/processed/datasets") / TF / f"{COIN}_train.parquet"
).columns
schema = TFTInputSchema.from_columns(cols, 
                                     bars_per_day=6, 
                                     include=["ohlcv"]
                                     )

loaders = build_dataloaders(COIN, 
                            schema, 
                            tf=TF, 
                            batch_size=64
                            )

batch = next(iter(loaders["train"]))

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = TemporalFusionTransformer(schema=schema, 
                                  hidden_size=64
                                  ).to(dev)
model.eval()

obs = batch.observed.to(dev)
kf = batch.known_future.to(dev)
sc = batch.static_cat.to(dev)

print("")
print("--- statystyki wejscia ---")
print(f"observed: mean={obs.mean():.4f} std={obs.std():.4f} "
      f"min={obs.min():.3f} max={obs.max():.3f}")
print(f"known: mean={kf.mean():.4f} std={kf.std():.4f}")
print(f"na kolumne std: min={obs.std(dim=(0, 1)).min():.5f} "
      f"max={obs.std(dim=(0, 1)).max():.3f}")

with torch.no_grad():
    out_real = torch.tanh(model(obs, kf, sc)["logit"])
    out_zero = torch.tanh(model(torch.zeros_like(obs), kf, sc)["logit"])
    out_rand = torch.tanh(model(torch.randn_like(obs), kf, sc)["logit"])

print("")
print("--- wrazliwosc wyjscia na observed ---")
print(f"out(real): std={out_real.std():.6f} mean={out_real.mean():+.6f}")
print(f"out(zero): std={out_zero.std():.6f} mean={out_zero.mean():+.6f}")
print(f"out(rand): std={out_rand.std():.6f} mean={out_rand.mean():+.6f}")
print(f"|real-zero| mean={ (out_real - out_zero).abs().mean():.6f}")
print(f"|real-rand| mean={ (out_real - out_rand).abs().mean():.6f}")

model.train()
out = torch.tanh(model(obs, kf, sc)["logit"])
out.mean().backward()

print("")
print("--- normy gradientow ---")
groups = {
    "observed_projections": model.observed_projections,
    "known_projections": model.known_projections,
    "encoder_vsn": model.encoder_vsn,
    "decoder_vsn": model.decoder_vsn,
    "encoder_lstm": model.encoder_lstm,
    "decoder_lstm": model.decoder_lstm,
    "attention": model.attention,
    "output_head": model.output_head
    }
for name, mod in groups.items():
    g = [p.grad.norm().item() for p in mod.parameters()
         if p.grad is not None]
    total = sum(x ** 2 for x in g) ** 0.5 if g else 0.0
    print(f"  {name:<22} {total:.3e}")
