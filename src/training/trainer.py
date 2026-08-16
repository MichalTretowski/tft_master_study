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
from dataclasses import asdict
from src.training.loss_functions import (
    DifferentiableSharpeRatio,
    compute_sharpe_ratio
    )

CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")

EARLY_STOPPING_CONFIG = {
    "sharpe":   {"val_key": "val_sharpe",  "mode": "max", "min_delta": 0.005},
    "dir_acc":  {"val_key": "val_dir_acc", "mode": "max", "min_delta": 0.005},
    "val_loss": {"val_key": "val_loss",    "mode": "min", "min_delta": 1e-4},
    }

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
        early_stopping_min_delta: float | None = None,
        early_stopping_metric: str | None = None,
        checkpoint_dir: Path = CHECKPOINT_DIR,
        experiment_name: str = "experiment",
        device: str | None = None,
        amp: bool = False,
        diag_every: int = 50,
        min_position_std: float = 0.02
        ) -> None:
        if loss_mode not in ("bce", "dsr"):
            raise ValueError(f"loss_mode: {loss_mode}")

        early_stopping_metric = early_stopping_metric or (
            "val_loss" if loss_mode == "bce" else "sharpe"
            )
        if early_stopping_metric not in EARLY_STOPPING_CONFIG:
            raise ValueError(f"early_stopping_metric: {early_stopping_metric}")
        if diag_every < 1:
            raise ValueError("diag_every musi byc >= 1")

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_mode = loss_mode
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
            )
        self.max_grad_norm = max_grad_norm
        cfg = EARLY_STOPPING_CONFIG[early_stopping_metric]
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_key = cfg["val_key"]
        self.metric_mode = cfg["mode"]
        self.early_stopping_min_delta = (
            early_stopping_min_delta
            if early_stopping_min_delta is not None
            else cfg["min_delta"]
            )
        self.min_position_std = min_position_std
        self.diag_every = diag_every
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
        self.amp_enabled = amp and self.device.startswith("cuda")
        self.grad_scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_enabled
            )
                
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100
        ) -> dict[str, list[float]]:
        best = float("-inf") if self.metric_mode == "max" else float("inf")
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

            score = val[self.early_stopping_key]

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

            collapsed = (
                self.loss_mode == "dsr"
                and val["val_pos_std"] < self.min_position_std
                )

            improved = (
                score > best + self.early_stopping_min_delta
                if self.metric_mode == "max"
                else score < best - self.early_stopping_min_delta
                )

            if improved and not collapsed:
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
        total = torch.zeros((), device=self.device)
        head_grad_total = 0.0
        dead_total = 0.0
        n_diag = 0

        for i, batch in enumerate(loader):
            obs = batch.observed.to(self.device, non_blocking=True)
            kf = batch.known_future.to(self.device, non_blocking=True)
            sc = batch.static_cat.to(self.device, non_blocking=True)
            pret = batch.price_return.to(self.device, non_blocking=True)
            tgt = batch.target.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            amp_ctx = torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=self.amp_enabled
                )

            # AMP obejmuje WYLACZNIE forward i BCE.
            with amp_ctx:
                logits = self._forward(obs, kf, sc).squeeze(-1)

                if self.loss_mode == "bce":
                    loss = self._bce_loss(logits, tgt)

            # DSR liczony poza autocast, w float32.
            if self.loss_mode == "dsr":
                pos = torch.tanh(logits.float())
                loss = self._dsr_loss(pos, pret.float())

            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)

            if i % self.diag_every == 0:
                head_grad_total += self._head_grad_norm()
                dead_total += self._dead_fraction()
                n_diag += 1

            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            total += loss.detach()

        n = max(len(loader), 1)
        d = max(n_diag, 1)

        return {
            "train_loss": total.item() / n,
            "train_dead": dead_total / d,
            "train_head_grad": head_grad_total / d
            }

    @torch.no_grad()
    def _validate_epoch(self,
                        loader: DataLoader
                        ) -> dict[str, float]:
        self.model.eval()
        logits_all, pos_all, ret_all, tgt_all = [], [], [], []

        for batch in loader:
            obs = batch.observed.to(self.device)
            kf = batch.known_future.to(self.device)
            sc = batch.static_cat.to(self.device)

            logits = self._forward(obs, kf, sc).squeeze(-1)

            logits_all.append(logits)
            pos_all.append(torch.tanh(logits))
            ret_all.append(batch.price_return.to(self.device))
            tgt_all.append(batch.target.to(self.device))

        logits = torch.cat(logits_all)
        pos = torch.cat(pos_all)
        ret = torch.cat(ret_all)
        tgt = torch.cat(tgt_all)

        valid = tgt >= 0

        if self.loss_mode == "bce":
            val_loss = self._bce_loss(logits[valid], tgt[valid]).item()
        else:
            val_loss = self._dsr_loss(pos.float(), ret.float()).item()

        pf_ret = self.dsr.portfolio_returns(pos.float(), ret.float())
        sharpe = compute_sharpe_ratio(pf_ret, timeframe=self.timeframe)

        dir_acc = (
            ((logits[valid] > 0).float() == tgt[valid])
            .float().mean().item()
            if valid.any() else float("nan")
            )

        return {
            "val_loss": val_loss,
            "val_sharpe": sharpe,
            "val_dir_acc": dir_acc,
            "val_pos_std": pos.std().item()
            }



    @staticmethod
    def _bce_loss(logits: torch.Tensor,
                  target: torch.Tensor
                  ) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, target)

    def _dsr_loss(
            self,
            pos: torch.Tensor,
            ret: torch.Tensor
            ) -> torch.Tensor:
        loss = self.dsr(pos, ret)
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
        return out["logit"] if isinstance(out, dict) else out

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
            "hidden_size": self.model.hidden_size,
            "lstm_layers": self.model.lstm_layers,
            "n_heads": self.model.n_heads,
            "dropout": self.model.dropout,
            "embedding_dim_per_categorical": (
                self.model.embedding_dim_per_categorical
                ),
        }
        schema = getattr(self.model, "schema", None)
        ckpt = {
            "epoch": epoch,
            "val_loss": val["val_loss"],
            "val_sharpe": val["val_sharpe"],
            "val_dir_acc": val["val_dir_acc"],
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "model_config": cfg,
            "loss_mode": self.loss_mode,
            "timeframe": self.timeframe,
            "schema": asdict(schema) if schema else None,
            "training_config": {
                "transaction_cost": self.transaction_cost,
                "position_penalty": self.position_penalty,
                "min_position_std": self.min_position_std,
                },
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
