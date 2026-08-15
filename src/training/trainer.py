"""
Petla treningowa dla TFT i LSTM Baseline.

Dwa tryby lossa, wybierane parametrem loss_mode:

  "bce": Binary Cross-Entropy 
  "dsr": Differentiable Sharpe Ratio

Metryka early stopping: "sharpe" albo "dir_acc".
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.training.loss_functions import (
    DifferentiableSharpeRatio,
    compute_sharpe_ratio
    )

CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        loss_mode: str = "bce",
        timeframe: str = "1d",
        transaction_cost: float = 0.001,
        position_penalty: float = 0.01,
        max_grad_norm: float = 1.0,
        early_stopping_patience: int = 20,
        early_stopping_min_delta: float = 0.005,
        early_stopping_metric: str = "sharpe",
        checkpoint_dir: Path = CHECKPOINT_DIR,
        experiment_name: str = "experiment",
        device: str | None = None
        ) -> None:
        if loss_mode not in ("bce", "dsr"):
            raise ValueError(f"loss_mode: {loss_mode}")
        if early_stopping_metric not in ("sharpe", "dir_acc"):
            raise ValueError(f"early_stopping_metric: {early_stopping_metric}")

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_mode = loss_mode
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
            )
        self.max_grad_norm = max_grad_norm
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_metric = early_stopping_metric
        self.timeframe = timeframe
        self.transaction_cost = transaction_cost
        self.position_penalty = position_penalty
        self.experiment_name = experiment_name
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.dsr = DifferentiableSharpeRatio(
            timeframe=timeframe,
            transaction_cost=transaction_cost
            )
        self.model.to(self.device)

        self._head_pre = None
        head = getattr(self.model, 'output_head', None)
        if isinstance(head, nn.Sequential):
            head[1].register_forward_hook(
                lambda _m, _i, out: setattr(self, "_head_pre", out.detach())
            )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100
        ) -> dict[str, list[float]]:
        best = float("-inf")
        patience = 0
        hist: dict[str, list[float]] = {
            "train_loss": [], 
            "train_dead": [],
            "train_head_grad": [],
            "val_loss": [],
            "val_sharpe": [], 
            "val_dir_acc": [], 
            "val_pos_std": []
            }

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr = self._train_epoch(train_loader)
            val = self._validate_epoch(val_loader)

            for k in ("train_loss", "train_dead", "train_head_grad"):
                hist[k].append(tr[k])

            for k in ("val_loss", "val_sharpe", "val_dir_acc", "val_pos_std"):
                hist[k].append(val[k])

            score = (
                val["val_sharpe"]
                if self.early_stopping_metric == "sharpe"
                else val["val_dir_acc"]
                )

            if self.scheduler is not None:
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                    self.scheduler.step(score)
                else:
                    self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train={tr['train_loss']:+.4f} | "
                f"val={val['val_loss']:+.4f} | "
                f"sharpe={val['val_sharpe']:+.4f} | "
                f"acc={val['val_dir_acc']:.3f} | "
                f"pos_std={val['val_pos_std']:.3f} | "
                f"dead={tr['train_dead']:.2f} | "
                f"hgrad={tr['train_head_grad']:.2e} | "
                f"lr={lr:.2e} | {time.time() - t0:.1f}s"
                )

            if val["val_pos_std"] < 0.02:
                score = float("-inf")

            if score > best + self.early_stopping_min_delta:
                best = score
                patience = 0
                self._save_checkpoint(epoch, val)
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    print(
                        f"Early stopping po {epoch} epokach. "
                        f"Best {self.early_stopping_metric}={best:+.4f}"
                    )
                    break

        self._save_history(hist)
        return hist

    @torch.no_grad()
    def evaluate(self, 
                 test_loader: DataLoader
                 ) -> dict[str, float]:
        val = self._validate_epoch(test_loader)
        print(
            f"[Test] sharpe={val['val_sharpe']:+.4f} | "
            f"acc={val['val_dir_acc']:.3f} | "
            f"pos_std={val['val_pos_std']:.3f}"
            )
        return val

    def load_best_checkpoint(self) -> dict:
        path = self.checkpoint_dir / "best_model.pt"
        if not path.exists():
            raise FileNotFoundError(f"Brak checkpointu: {path}")
        ckpt = torch.load(path, 
                          map_location=self.device, 
                          weights_only=False
                          )
        self.model.load_state_dict(ckpt["model_state"])
        print(
            f"[Trainer] Checkpoint: epoch={ckpt['epoch']}, "
            f"sharpe={ckpt['val_sharpe']:+.4f}"
            )
        return ckpt



    def _train_epoch(self, 
                     loader: DataLoader
                     ) -> dict[str, float]:
        self.model.train()
        total = 0.0
        head_grad_total = 0.0
        dead_total = 0.0

        for batch in loader:
            obs = batch.observed.to(self.device)
            kf = batch.known_future.to(self.device)
            sc = batch.static_cat.to(self.device)
            pret = batch.price_return.to(self.device)
            tgt = batch.target.to(self.device)

            self.optimizer.zero_grad()
            pos = self._forward(obs, kf, sc).squeeze(-1)

            if self.loss_mode == "bce":
                loss = self._bce_loss(pos, tgt)
            else:
                loss = self._dsr_loss(pos, pret)

            loss.backward()

            head_grad_total += self._head_grad_norm()
            dead_total += self._dead_fraction()

            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.optimizer.step()
            total += loss.item()

        n = max(len(loader), 1)

        return {
            "train_loss": total / n,
            "train_dead": dead_total / n,
            "train_head_grad": head_grad_total / n
            }

    @torch.no_grad()
    def _validate_epoch(self, 
                        loader: DataLoader
                        ) -> dict[str, float]:
        self.model.eval()
        pos_all, ret_all, tgt_all = [], [], []

        for batch in loader:
            obs = batch.observed.to(self.device)
            kf = batch.known_future.to(self.device)
            sc = batch.static_cat.to(self.device)

            pos = self._forward(obs, kf, sc).squeeze(-1)
            pos_all.append(pos.cpu())
            ret_all.append(batch.price_return)
            tgt_all.append(batch.target)

        pos = torch.cat(pos_all)
        ret = torch.cat(ret_all)
        tgt = torch.cat(tgt_all)

        if self.loss_mode == "bce":
            val_loss = self._bce_loss(pos, tgt).item()
        else:
            val_loss = self._dsr_loss(pos, ret).item()

        pf_ret = self._portfolio_returns(pos, ret)
        sharpe = compute_sharpe_ratio(pf_ret, timeframe=self.timeframe)
        dir_acc = ((pos > 0).float() == tgt).float().mean().item()

        return {
            "val_loss": val_loss,
            "val_sharpe": sharpe,
            "val_dir_acc": dir_acc,
            "val_pos_std": pos.std().item()
            }



    @staticmethod
    def _bce_loss(pos: torch.Tensor, 
                  target: torch.Tensor
                  ) -> torch.Tensor:
        probs = ((pos + 1) / 2).clamp(1e-6, 1 - 1e-6)
        return F.binary_cross_entropy(probs, target)

    def _dsr_loss(
            self, 
            pos: torch.Tensor, 
            ret: torch.Tensor
            ) -> torch.Tensor:
        pf_ret = self._portfolio_returns(pos, 
                                         ret
                                         )
        loss = self.dsr._sharpe(pf_ret)
        return loss + self.position_penalty * pos.pow(2).mean()

    def _portfolio_returns(
            self, 
            pos: torch.Tensor, 
            ret: torch.Tensor
            ) -> torch.Tensor:
        changes = torch.abs(pos[1:] - pos[:-1])
        return pos[:-1] * ret[1:] - changes * self.transaction_cost



    def _forward(self, 
                 obs, 
                 kf, 
                 sc
                 ) -> torch.Tensor:
        out = self.model(obs, 
                         kf, 
                         sc
                         )
        return out["output"] if isinstance(out, dict) else out

    def _head_grad_norm(self) -> float:
        head = getattr(self.model, "output_head", None)
        if head is None:
            return float("nan")
        return sum(
            p.grad.norm().item() **2
            for p in head.parameters()
            if p.grad is not None
            ) ** 0.5

    def _dead_fraction(self) -> float:
        if self._head_pre is None:
            return float("nan")
        return (self._head_pre <= 0).all(dim=0).float().mean().item()

    def _save_checkpoint(self, 
                         epoch: int, 
                         val: dict
                         ) -> None:
        cfg = {
            k: getattr(self.model, k)
            for k in ("hidden_size", "lstm_layers", "n_heads", "dropout")
            if hasattr(self.model, k)
        }
        schema = getattr(self.model, "schema", None)
        ckpt = {
            "epoch": epoch,
            "val_sharpe": val["val_sharpe"],
            "val_dir_acc": val["val_dir_acc"],
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "model_config": cfg,
            "loss_mode": self.loss_mode,
            "timeframe": self.timeframe,
            "observed_reals": list(schema.observed_reals) if schema else [],
            "known_reals": list(schema.known_reals) if schema else [],
        }
        torch.save(ckpt, self.checkpoint_dir / "best_model.pt")

    def _save_history(self, 
                      history: dict
                      ) -> None:
        LOG_DIR.mkdir(exist_ok=True)
        path = LOG_DIR / f"{self.experiment_name}_history.json"
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"[Trainer] Historia -> {path}")
