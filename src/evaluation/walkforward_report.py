"""

Agregacja wynikow walk-forward.

Cala arytmetyka raportu zyje w tym skrypcie. Notebook tylko wola i rysuje

Zasady, ktore ten modul egzekwuje:
  - metryki out-of-sample licza sie wyłącznie z segment == "eval",
  - ensemble to srednia prawdopodobieństw po seedach (nie logitow),
  - baseline liczy sie per obserwacia z priora treningowego jej folda,
  - podzbior pre-test wyznacza sie z original_test_partition_start
    zapisanego w artefakcie, nie z aktualnego SPLIT_DATES,
  - liczba seedow i liczba okien pochodza z metadanych, nie z danych,
  - przy partial_run regula decyzyjna nie jest liczona.

"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

WF_DIR = Path("experiments/walkforward")
EPS = 1e-12
NUMERIC_COLS = ["logit", "prob", "p_train", "target"]


def _to_utc(value) -> pd.Timestamp:
    """Jedna konwersja dla wszystkich porownan dat"""
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


@dataclass
class WalkForwardRun:
    predictions: pd.DataFrame
    meta: dict

    @property
    def run(self) -> dict:
        return self.meta["run"]

    @property
    def folds(self) -> list[dict]:
        return self.meta["folds"]

    @property
    def protocol(self) -> dict:
        return self.run["decision_protocol"]

    @property
    def seeds(self) -> list[int]:
        return list(self.run["recipe"]["seeds"])

    @property
    def is_partial(self) -> bool:
        return bool(self.run["partial_run"])


def _check_integrity(pred: pd.DataFrame, meta: dict) -> None:
    values = pred[NUMERIC_COLS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        bad = pred.loc[~np.isfinite(values).all(axis=1)]
        raise ValueError(
            f"Nieskonczone lub brakujace wartosci w {len(bad)} wierszach."
            )

    expected_prob = 1.0 / (1.0 + np.exp(-pred["logit"].to_numpy()))
    max_err = float(np.max(np.abs(pred["prob"].to_numpy() - expected_prob)))
    if not max_err <= 1e-5:
        raise ValueError(
            f"prob nie zgadza sie z sigmoid(logit): max blad {max_err:.2e}"
            )

    segments = set(pred["segment"].unique())
    if segments != {"val", "eval"}:
        raise ValueError(
            f"Nieoczekiwane segmenty: {sorted(segments)}. "
            "Dozwolone wylacznie 'val' i 'eval'."
            )

    if pred.duplicated(["fold", "seed", "segment", "timestamp"]).any():
        raise ValueError("Zduplikowane wiersze predykcji.")

    if pred.duplicated(["fold", "seed", "timestamp"]).any():
        raise ValueError(
            "Ten sam timestamp wystepuje w obu segmentach jednego folda "
            "- zakresy okna sie nakladaja."
            )

    targets = set(np.unique(pred["target"].to_numpy()))
    if not targets <= {0.0, 1.0}:
        raise ValueError(f"Targety nie sa binarne: {sorted(targets)}")

    in_pred = set(pred["fold"].unique())
    in_meta = {f["fold"] for f in meta["folds"]}
    if in_pred != in_meta:
        raise ValueError(
            f"Foldy w predykcjach {sorted(in_pred)} != "
            f"foldy w metadanych {sorted(in_meta)}"
            )

    seeds = sorted(meta["run"]["recipe"]["seeds"])
    n_seeds = len(seeds)

    for fold in meta["folds"]:
        fid = fold["fold"]
        rows = pred[pred["fold"] == fid]

        got = sorted(rows["seed"].unique())
        if got != seeds:
            raise ValueError(
                f"Fold {fid}: seedy {got} != oczekiwane {seeds}"
                )

        for segment, key_n in (("val", "n_val_windows"),
                               ("eval", "n_eval_windows")):
            expected = fold[key_n] * n_seeds
            actual = int((rows["segment"] == segment).sum())
            if actual != expected:
                raise ValueError(
                    f"Fold {fid}, segment {segment}: {actual} wierszy, "
                    f"oczekiwano {expected} ({fold[key_n]} okien x {n_seeds})"
                    )

        priors = rows["p_train"].to_numpy(dtype=np.float64)
        if not np.allclose(priors, fold["p_train"], atol=1e-9):
            raise ValueError(
                f"Fold {fid}: p_train w predykcjach nie jest stale albo "
                f"rozni sie od folds.json ({fold['p_train']})"
                )


def load(wf_dir: Path = WF_DIR) -> WalkForwardRun:
    meta = json.loads(
        (wf_dir / "folds.json").read_text(encoding="utf-8")
        )
    pred = pd.read_parquet(wf_dir / "predictions.parquet")
    _check_integrity(pred, meta)
    return WalkForwardRun(pred, meta)


def ensemble(run: WalkForwardRun, segment: str = "eval") -> pd.DataFrame:
    """Srednia prawdopodobienstw po seedach, jeden wiersz na obserwacje."""
    d = run.predictions[run.predictions["segment"] == segment]
    if d.empty:
        raise ValueError(f"Brak wierszy dla segmentu '{segment}'.")

    n_seeds = len(run.seeds)

    grouped = d.groupby(["fold", "timestamp"], as_index=False).agg(
        prob=("prob", "mean"),
        n_targets=("target", "nunique"),
        target=("target", "first"),
        p_train=("p_train", "first"),
        n_seeds=("seed", "nunique")
        )

    if (grouped["n_targets"] != 1).any():
        raise ValueError(
            "Ta sama obserwacja ma rozne targety miedzy seedami."
            )

    if (grouped["n_seeds"] != n_seeds).any():
        missing = grouped[grouped["n_seeds"] != n_seeds]
        raise ValueError(
            f"{len(missing)} obserwacji nie ma kompletu {n_seeds} seedow."
            )

    out = grouped.drop(columns=["n_targets", "n_seeds"])
    return out.sort_values(["fold", "timestamp"]).reset_index(drop=True)


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def metrics(df: pd.DataFrame) -> dict:
    """Metryki dla dowolnego podzbioru obserwacji ensemble'u."""
    y = df["target"].to_numpy(dtype=np.float64)
    p = df["prob"].to_numpy(dtype=np.float64)
    prior = np.clip(
        df["p_train"].to_numpy(dtype=np.float64), 1e-6, 1.0 - 1e-6
        )

    loss = _log_loss(y, p)
    baseline_loss = float(
        -(y * np.log(prior) + (1.0 - y) * np.log(1.0 - prior)).mean()
        )
    majority = (prior > 0.5).astype(np.float64)

    return {
        "n": int(len(df)),
        "loss": loss,
        "baseline_loss": baseline_loss,
        "advantage": baseline_loss - loss,
        "dir_acc": float(((p > 0.5) == (y > 0.5)).mean()),
        "baseline_acc": float((y == majority).mean()),
        "p_eval": float(y.mean())
        }


def per_fold(ens: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"fold": int(fold), **metrics(part)}
        for fold, part in ens.groupby("fold", sort=True)
        ]
    return pd.DataFrame(rows).set_index("fold")


def pre_test_folds(run: WalkForwardRun) -> list[int]:
    """Okna, ktorych ocena konczy sie przed poczatkiem uzytego testu."""
    boundary = _to_utc(run.run["original_test_partition_start"])
    return sorted(
        f["fold"] for f in run.folds
        if _to_utc(f["eval_end"]) < boundary
        )


def evaluate_protocol(run: WalkForwardRun, ens: pd.DataFrame) -> dict:
    """Stosuje regule decyzyjna zapisana w artefakcie."""
    if run.is_partial:
        raise ValueError(
            "partial_run=True - regula decyzyjna nie dotyczy podzbioru okien."
            )

    proto = run.protocol
    if proto["primary_subset"] != "pre_test":
        raise ValueError(
            f"Nieobslugiwany primary_subset: {proto['primary_subset']}"
            )

    table = per_fold(ens)

    def verdict(
        folds: list[int],
        min_positive: int,
        total: int,
        require_positive: bool
        ) -> dict:
        if len(folds) != total:
            raise ValueError(
                f"Oczekiwano {total} okien, jest {len(folds)}: {folds}"
                )

        pooled = metrics(ens[ens["fold"].isin(folds)])
        positive = int((table.loc[folds, "advantage"] > 0).sum())
        passed = positive >= min_positive and (
            pooled["advantage"] > 0 if require_positive else True
            )

        return {
            "folds": folds,
            "pooled": pooled,
            "positive_folds": positive,
            "min_positive_folds": min_positive,
            "require_pooled_positive": require_positive,
            "passed": bool(passed)
            }

    primary_positive = bool(
        proto["primary_pooled_advantage_must_be_positive"]
        )

    return {
        "primary": verdict(
            pre_test_folds(run),
            proto["primary_min_positive_folds"],
            proto["primary_total_folds"],
            primary_positive
            ),
        "full": verdict(
            sorted(table.index),
            proto["full_min_positive_folds"],
            proto["full_total_folds"],
            bool(proto.get(
                "full_pooled_advantage_must_be_positive", primary_positive
                ))
            )
        }


def summary(run: WalkForwardRun | None = None) -> None:
    run = run or load()
    ens = ensemble(run, "eval")
    table = per_fold(ens)

    meta = run.run
    print("")
    print(f"commit         {meta['commit'][:8]}  dirty={meta['dirty']}")
    print(f"schema / dane  {meta['schema_fingerprint']} / "
          f"{meta['data_fingerprint']}")
    print(f"ensemble       {meta['ensemble_rule']} "
          f"({len(run.seeds)} seedy)")
    print(f"granica testu  {meta['original_test_partition_start'][:10]}")
    print("")

    show = table[[
        "n", "loss", "baseline_loss", "advantage", "dir_acc", "baseline_acc"
        ]]
    print(show.round(6).to_string())

    if run.is_partial:
        print("")
        print("partial_run=True - regula decyzyjna pominieta.")
        return

    result = evaluate_protocol(run, ens)
    for name, key in (("PRE-TEST (wiazacy)", "primary"),
                      ("PELNE OKNA (wrazliwosc)", "full")):
        v = result[key]
        p = v["pooled"]
        print("")
        print(f"--- {name} ---")
        print(f"okna              {v['folds']}")
        print(f"obserwacji        {p['n']}")
        print(f"pooled loss       {p['loss']:.6f}")
        print(f"pooled baseline   {p['baseline_loss']:.6f}")
        print(f"przewaga          {p['advantage']:+.6f} nata")
        print(f"dir_acc           {p['dir_acc']:.4f} "
              f"vs baseline {p['baseline_acc']:.4f}")
        print(f"okna dodatnie     {v['positive_folds']} / {len(v['folds'])} "
              f"(wymagane {v['min_positive_folds']})")
        print(f"WYNIK             "
              f"{'SPELNIONE' if v['passed'] else 'NIESPELNIONE'}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    summary()
