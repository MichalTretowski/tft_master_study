"""
Walk-forward dla zamrozonej konfiguracji OHLCV/1d/BCE.

Siedem okien, trening rozszerzajacy sie, krok pol roku. W kazdym oknie
trening od zera na trzech seedach; checkpoint wybierany wylacznie na
wewnetrznej walidacji okna, ocena na kolejnym, nieuzytym odcinku.

Wyjscie:
  experiments/walkforward/predictions.parquet - jeden wiersz na obserwacje i seed
  experiments/walkforward/folds.json - granice okien + metadane uruchomienia

Agregacje liczy notebook. Ten skrypt nie raportuje zadnych wnioskow.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataloader import build_dataloaders
from src.models.tft.architecture import TemporalFusionTransformer
from src.models.tft.input_schema import TFTInputSchema
from src.preprocessing.alignment.time_aligner import SPLIT_DATES
from src.training.trainer import Trainer
from src.utils.logger import get_logger

logger = get_logger("walkforward")

DATASETS = Path("data/processed/datasets")
OUT_DIR = Path("experiments/walkforward")


RECIPE = {
    "coin": "BTC",
    "tf": "1d",
    "include": ["ohlcv"],
    "hidden_size": 32,
    "lstm_layers": 1,
    "n_heads": 4,
    "dropout": 0.1,
    "embedding_dim_per_categorical": 8,
    "encoder_days": 30,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "max_grad_norm": 1.0,
    "epochs": 100,
    "patience": 20,
    "loss": "bce",
    "es_metric": "val_loss",
    "es_min_delta": 1e-4,
    "seeds": [1, 2, 3],
    "num_workers": 0,
    "amp": False,
    "scheduler_factor": 0.5,
    "scheduler_patience": 5,
    "scheduler_min_lr": 1e-6
    }

ENSEMBLE_RULE = "mean_prob_over_seeds"
HORIZON_BARS = 1

DECISION_PROTOCOL = {
    "primary_subset": "pre_test",
    "primary_pooled_advantage_must_be_positive": True,
    "primary_min_positive_folds": 3,
    "primary_total_folds": 4,
    "full_min_positive_folds": 5,
    "full_total_folds": 7
    }

FOLDS = [
    ("2021-12-31", "2022-01-01", "2022-06-30", "2022-07-01", "2022-12-31"),
    ("2022-06-30", "2022-07-01", "2022-12-31", "2023-01-01", "2023-06-30"),
    ("2022-12-31", "2023-01-01", "2023-06-30", "2023-07-01", "2023-12-31"),
    ("2023-06-30", "2023-07-01", "2023-12-31", "2024-01-01", "2024-06-30"),
    ("2023-12-31", "2024-01-01", "2024-06-30", "2024-07-01", "2024-12-31"),
    ("2024-06-30", "2024-07-01", "2024-12-31", "2025-01-01", "2025-06-30"),
    ("2024-12-31", "2025-01-01", "2025-06-30", "2025-07-01", "2025-12-31")
    ]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_meta() -> dict:
    def run(cmd: list[str]) -> str:
        return subprocess.run(
            cmd, capture_output=True, text=True
            ).stdout.strip()

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(run(["git", "status", "--porcelain"]))
        }


def fingerprint_schema(schema: TFTInputSchema) -> str:
    payload = json.dumps(
        asdict(schema), sort_keys=True, default=str
        ).encode("utf-8")
    return sha256(payload).hexdigest()[:12]


def verify_test_boundary(coin: str, tf: str) -> str:
    """
    Porownuje granice z początkiej surowej partycji testu
    """
    raw_test_start = pd.read_parquet(
        DATASETS / tf / f"{coin}_test.parquet"
        ).sort_index().index.min()

    expected = pd.Timestamp(SPLIT_DATES["test_start"], tz="UTC")
    if raw_test_start != expected:
        raise ValueError(
            f"Poczatek partycji testowej {raw_test_start} nie zgadza sie "
            f"ze SPLIT_DATES['test_start'] = {expected}"
            )

    return expected.isoformat()


def fingerprint_data(coin: str, tf: str) -> str:
    h = sha256()
    for split in ("train", "val", "test"):
        h.update((DATASETS / tf / f"{coin}_{split}.parquet").read_bytes())
    return h.hexdigest()[:12]


def load_master(coin: str, tf: str) -> pd.DataFrame:
    """Sklada ciagly szereg z trzech pierwotnych splitow."""
    parts = [
        pd.read_parquet(DATASETS / tf / f"{coin}_{split}.parquet")
        for split in ("train", "val", "test")
        ]
    df = pd.concat(parts).sort_index()

    if df.index.duplicated().any():
        raise ValueError("Zduplikowane timestampy po sklejeniu splitow.")

    return df


def purge(df: pd.DataFrame, bars: int) -> pd.DataFrame:
    """Ostatnie 'bars' obserwacji odcinka nie moze byc przykladem uczacym."""
    out = df.copy()
    if bars > 0 and len(out) >= bars:
        tail = out.index[-bars:]
        out.loc[tail, "target"] = pd.NA
        out.loc[tail, "forward_log_return"] = float("nan")
    return out


def write_fold_slices(
    master: pd.DataFrame,
    bounds: tuple[str, str, str, str, str],
    coin: str,
    tf: str,
    tmp_datasets: Path
    ) -> dict[str, tuple[str, str]]:
    """
    Zapisuje wycinki okna pod nazwami, ktorych oczekuje CryptoDataset.
    """
    train_end, val_start, val_end, eval_start, eval_end = bounds

    segments = {
        "train": master.loc[:train_end],
        "val": master.loc[val_start:val_end],
        "test": master.loc[eval_start:eval_end]
        }

    target_dir = tmp_datasets / tf
    target_dir.mkdir(parents=True, exist_ok=True)

    ranges = {}
    for name, seg in segments.items():
        if seg.empty:
            raise ValueError(f"Pusty odcinek '{name}' dla granic {bounds}")
        purge(seg, HORIZON_BARS).to_parquet(
            target_dir / f"{coin}_{name}.parquet"
            )
        ranges[name] = (
            seg.index[0].isoformat(),
            seg.index[-1].isoformat()
            )

    return ranges


def predict(model, loader, device: str) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch.observed.to(device),
                batch.known_future.to(device),
                batch.static_cat.to(device)
                )["logit"].squeeze(-1)
            chunks.append(logits.cpu())
    return torch.cat(chunks).numpy()


def run_fold(
    fold_id: int,
    bounds: tuple[str, str, str, str, str],
    master: pd.DataFrame,
    schema: TFTInputSchema,
    device: str
    ) -> tuple[pd.DataFrame, dict]:
    coin = RECIPE["coin"]
    tf = RECIPE["tf"]

    tmp_root = Path(tempfile.mkdtemp(prefix=f"wf_fold{fold_id}_"))
    tmp_datasets = tmp_root / "datasets"
    tmp_scalers = tmp_root / "scalers"

    try:
        ranges = write_fold_slices(
            master, bounds, coin, tf, tmp_datasets
            )


        loaders = build_dataloaders(
            coin=coin,
            schema=schema,
            tf=tf,
            batch_size=RECIPE["batch_size"],
            num_workers=RECIPE["num_workers"],
            datasets_dir=tmp_datasets,
            scalers_dir=tmp_scalers,
            loss_mode=RECIPE["loss"]
            )

        p_train = float(loaders["train"].dataset.class_prior)

        segments = {
            "val": loaders["val"].dataset,
            "eval": loaders["test"].dataset
            }
        loaders_by_segment = {
            "val": loaders["val"],
            "eval": loaders["test"]
            }

        rows = []
        per_seed = []

        for seed in RECIPE["seeds"]:
            set_seed(seed)

            model = TemporalFusionTransformer(
                schema=schema,
                hidden_size=RECIPE["hidden_size"],
                lstm_layers=RECIPE["lstm_layers"],
                n_heads=RECIPE["n_heads"],
                dropout=RECIPE["dropout"],
                embedding_dim_per_categorical=RECIPE[
                    "embedding_dim_per_categorical"
                    ]
                )

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=RECIPE["lr"],
                weight_decay=RECIPE["weight_decay"]
                )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=RECIPE["scheduler_factor"],
                patience=RECIPE["scheduler_patience"],
                min_lr=RECIPE["scheduler_min_lr"]
                )

            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_mode=RECIPE["loss"],
                timeframe=tf,
                max_grad_norm=RECIPE["max_grad_norm"],
                amp=RECIPE["amp"],
                early_stopping_patience=RECIPE["patience"],
                early_stopping_metric=RECIPE["es_metric"],
                early_stopping_min_delta=RECIPE["es_min_delta"],
                checkpoint_dir=tmp_root / "checkpoints",
                experiment_name=f"fold{fold_id}_seed{seed}",
                device=device
                )

            logger.info(
                f"[fold {fold_id}] seed {seed}: trening "
                f"({len(loaders['train'].dataset)} okien)"
                )
            trainer.fit(
                loaders["train"],
                loaders["val"],
                epochs=RECIPE["epochs"]
                )

            ckpt = trainer.load_best_checkpoint()
            per_seed.append({
                "seed": seed,
                "checkpoint_epoch": int(ckpt["epoch"]),
                "val_loss": float(ckpt["val_loss"])
                })

            for segment, ds in segments.items():
                logits = predict(
                    trainer.model, loaders_by_segment[segment], device
                    )
                targets = ds.window_targets

                if len(logits) != len(targets):
                    raise ValueError(
                        f"Niezgodnosc dlugosci predykcji i okien "
                        f"({len(logits)} vs {len(targets)}) "
                        f"w segmencie {segment}"
                        )

                rows.append(pd.DataFrame({
                    "fold": fold_id,
                    "seed": seed,
                    "timestamp": ds.window_timestamps,
                    "segment": segment,
                    "target": targets.astype(np.float32),
                    "logit": logits.astype(np.float32),
                    "prob": (1.0 / (1.0 + np.exp(-logits))).astype(np.float32),
                    "p_train": np.float32(p_train),
                    }))

        predictions = pd.concat(rows, ignore_index=True)

        record = {
            "fold": fold_id,
            "train_start": ranges["train"][0],
            "train_end": ranges["train"][1],
            "val_start": ranges["val"][0],
            "val_end": ranges["val"][1],
            "eval_start": ranges["test"][0],
            "eval_end": ranges["test"][1],
            "n_train_windows": len(loaders["train"].dataset),
            "n_train_optimized_examples": (
                len(loaders["train"]) * RECIPE["batch_size"]
                ),
            "n_val_windows": len(segments["val"]),
            "n_eval_windows": len(segments["eval"]),
            "p_train": p_train,
            "per_seed": per_seed
            }

        return predictions, record

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None)
    p.add_argument("--folds",
                   default=None,
                   help="podzbior okien po przecinku, np. 1,2 - do testu"
                   )
    args = p.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
        )

    coin = RECIPE["coin"]
    tf = RECIPE["tf"]

    test_partition_start = verify_test_boundary(coin, tf)

    master = load_master(coin, tf)
    schema = TFTInputSchema.from_columns(
        master.columns,
        bars_per_day=6 if tf == "4h" else 1,
        include=RECIPE["include"],
        encoder_days=RECIPE["encoder_days"]
        )
    schema.summary()

    expected = set(range(1, len(FOLDS) + 1))
    selected = (
        [int(x) for x in args.folds.split(",")]
        if args.folds else sorted(expected)
        )

    if any(fold not in expected for fold in selected):
        raise ValueError(f"Nieprawidlowy numer folda w {selected}.")
    if len(selected) != len(set(selected)):
        raise ValueError("Foldy nie moga sie powtarzac.")


    partial_run = set(selected) != expected

    all_predictions = []
    records = []

    for fold_id in selected:
        bounds = FOLDS[fold_id - 1]
        predictions, record = run_fold(
            fold_id, bounds, master, schema, device
            )
        all_predictions.append(predictions)
        records.append(record)

        logger.info(
            f"[fold {fold_id}] gotowe: "
            f"train={record['n_train_windows']}, "
            f"val={record['n_val_windows']}, "
            f"eval={record['n_eval_windows']}, "
            f"p_train={record['p_train']:.6f}"
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_parquet(OUT_DIR / "predictions.parquet")

    meta = {
        "run": {
            **git_meta(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "schema_fingerprint": fingerprint_schema(schema),
            "data_fingerprint": fingerprint_data(coin, tf),
            "source_cutoff": master.index[-1].isoformat(),
            "recipe": RECIPE,
            "ensemble_rule": ENSEMBLE_RULE,
            "decision_protocol": DECISION_PROTOCOL,
            "horizon_bars": HORIZON_BARS,
            "original_test_partition_start": test_partition_start,
            "partial_run": partial_run
            },
        "folds": records,
        }
    (OUT_DIR / "folds.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
        )

    logger.info(
        f"Zapisano {len(predictions)} wierszy predykcji "
        f"z {len(records)} okien -> {OUT_DIR}"
        )


if __name__ == "__main__":
    main()