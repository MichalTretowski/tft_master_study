"""
Ewaluacja checkpointu na zbiorze testowym.

Punkty odniesienia:
  baseline_loss - staly predyktor z czestosci klasy w treningu
  baseline_acc - stala klasa wiekszosciowa Z treningu, oceniona na tescie
  entropy - entropia rozkladu testu (podloga stalego predyktora)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataloader import build_dataloaders
from src.models.tft.architecture import TemporalFusionTransformer
from src.models.tft.input_schema import TFTInputSchema
from src.training.loss_functions import (
    DifferentiableSharpeRatio,
    compute_sharpe_ratio,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", 
                   required=True
                   )
    p.add_argument("--coin", 
                   default="BTC", 
                   choices=["BTC", "ETH"]
                   )
    p.add_argument("--batch-size", 
                   type=int, 
                   default=256
                   )
    p.add_argument("--device", 
                   default=None
                   )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_path = Path("checkpoints") / args.experiment / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Brak checkpointu: {ckpt_path}")

    device = args.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
        )
    ckpt = torch.load(ckpt_path, 
                      map_location=device, 
                      weights_only=False
                      )

    if ckpt.get("schema") is None:
        raise ValueError(
            "Checkpoint bez schematu - pochodzi sprzed poprawki. "
            "Przelicz go na aktualnym trainerze."
            )

    schema = TFTInputSchema(**ckpt["schema"])
    schema.summary()

    tf = ckpt["timeframe"]
    loss_mode = ckpt["loss_mode"]
    train_cfg = ckpt["training_config"]

    model = TemporalFusionTransformer(
        schema=schema,
        **ckpt["model_config"]
        ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loaders = build_dataloaders(
        coin=args.coin,
        schema=schema,
        tf=tf,
        batch_size=args.batch_size,
        loss_mode=loss_mode
        )

    logits_all, pos_all, ret_all, tgt_all = [], [], [], []

    with torch.no_grad():
        for batch in loaders["test"]:
            logits = model(
                batch.observed.to(device),
                batch.known_future.to(device),
                batch.static_cat.to(device)
                )["logit"].squeeze(-1)

            logits_all.append(logits)
            pos_all.append(torch.tanh(logits))
            ret_all.append(batch.price_return.to(device))
            tgt_all.append(batch.target.to(device))

    logits = torch.cat(logits_all)
    pos = torch.cat(pos_all)
    ret = torch.cat(ret_all)
    tgt = torch.cat(tgt_all)
    valid = tgt >= 0

    dsr = DifferentiableSharpeRatio(
        timeframe=tf,
        transaction_cost=train_cfg["transaction_cost"]
        )
    pf_ret = dsr.portfolio_returns(pos.float(), 
                                   ret.float()
                                   )
    sharpe = compute_sharpe_ratio(pf_ret, 
                                  timeframe=tf
                                  )

    print("")
    print(f"=== {args.experiment} | {args.coin} / {tf} "
          f"| epoka {ckpt['epoch']} ===")
    print(f"  sharpe         {sharpe:+.4f}")
    print(f"  pos_std        {pos.std().item():.4f}")

    if loss_mode != "bce":
        print("  checkpoint DSR - metryki BCE pominiete, "
              "cel treningu byl inny")
        return

    if not valid.any():
        print("  brak binarnych targetow w tescie - pomijam metryki BCE")
        return

    p_train = loaders["train"].dataset.class_prior
    p_train = min(max(p_train, 1e-6), 1 - 1e-6)
    baseline_logit = math.log(p_train / (1 - p_train))

    test_loss = F.binary_cross_entropy_with_logits(
        logits[valid], tgt[valid]
        ).item()
    baseline_loss = F.binary_cross_entropy_with_logits(
        torch.full_like(logits[valid], baseline_logit), tgt[valid]
        ).item()

    accuracy = (
        (logits[valid] > 0).float() == tgt[valid]
        ).float().mean().item()


    majority = 1.0 if p_train > 0.5 else 0.0
    baseline_acc = (tgt[valid] == majority).float().mean().item()

    p_test = tgt[valid].mean().item()
    entropy = float(
        -(p_test * np.log(p_test) + (1 - p_test) * np.log(1 - p_test))
        )

    print(f"  test_loss      {test_loss:.6f}")
    print(f"  baseline_loss  {baseline_loss:.6f}  (p_train={p_train:.4f})")
    print(f"  entropia testu {entropy:.6f}  (p_test={p_test:.4f})")
    print(f"  przewaga       {baseline_loss - test_loss:+.6f} nata")
    print(f"  dir_acc        {accuracy:.4f}  vs baseline {baseline_acc:.4f}")


if __name__ == "__main__":
    main()