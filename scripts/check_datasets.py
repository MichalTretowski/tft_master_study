"""

Weryfikacja spojnosci datasetow 4h i 1d.

Sprawdza, czy oba zbiory roznia sie wyłącznie interwalem:
schemat kolumn, granice splitow, zgodnosc cen, target,
sentyment, kolumny stale i braki danych.

"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATASETS = Path("data/processed/datasets")
SPLITS = ["train", 
          "val", 
          "test"
          ]

# Cechy, ktore z definicji nie moga istniec na swiecach dziennych
ONLY_4H = {"hist_vol_24", 
           "return_4h"
           }

# Kolumny, ktore MOGA byc stale (znane i zaakceptowane)
CONST_OK = {"hour_sin",
            "hour_cos",
            "coin_id",
            "twitter_sentiment_score",
            "news_sentiment_score",
            "twitter_n_texts",
            "news_n_texts"
            }

FAILS = []


def check(label, 
          ok, 
          detail=""
          ):
    mark = "OK  " if ok else "FAIL"
    line = "  [" + mark + "] " + label
    if detail:
        line += " -- " + detail
    print(line)
    if not ok:
        FAILS.append(label)


def load(tf, 
         coin
         ):
    out = {}
    for split in SPLITS:
        path = DATASETS / tf / f"{coin}_{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Brak {path}")
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, 
                                  utc=True
                                  )
        out[split] = df
    return out


def full(d):
    return pd.concat([d[s] for s in SPLITS]).sort_index()


def check_schema(d4, 
                 d1
                 ):
    print("")
    print("[1] Schemat kolumn")
    c4 = set(d4["train"].columns)
    c1 = set(d1["train"].columns)

    only_4h = c4 - c1
    only_1d = c1 - c4

    check("1d nie ma wlasnych kolumn",
          not only_1d,
          str(sorted(only_1d))
          )
    check("brakujace w 1d to cechy sub-dobowe",
          only_4h == ONLY_4H,
          str(sorted(only_4h))
          )
    check("target obecny w obu",
          "target" in c4 and "target" in c1
          )
    check("coin_id obecny w obu",
          "coin_id" in c4 and "coin_id" in c1
          )


def check_splits(d4, 
                 d1
                 ):
    print("")
    print("[2] Granice splitow")
    for tf, d in [("4h", d4), ("1d", d1)]:
        tr = d["train"]
        va = d["val"]
        te = d["test"]
        check(f"{tf}: train konczy sie przed val",
              tr.index.max() < va.index.min())
        check(f"{tf}: val konczy sie przed test",
              va.index.max() < te.index.min())

    for split in SPLITS:
        a = d4[split].index.min().date()
        b = d1[split].index.min().date()
        check(f"{split}: ta sama data startu",
              a == b,
              f"4h={a} 1d={b}")


def check_index(f4, 
                f1
                ):
    print("")
    print("[3] Indeks czasowy")
    check("1d: wszystkie bary o 00:00 UTC",
          bool((f1.index.hour == 0).all())
          )

    hours = set(f4.index.hour)
    check("4h: godziny 0/4/8/12/16/20",
          hours <= {0, 4, 8, 12, 16, 20}
          )

    in_range = (
        f1.index.min() >= f4.index.min().normalize()
        and f1.index.max() <= f4.index.max()
    )
    check("1d miesci sie w zakresie 4h", 
          in_range
          )


def check_price(f4, 
                f1
                ):
    print("")
    print("[4] Zgodnosc cen")
    days = f4.index.normalize()
    last_4h = f4["close"].groupby(days).last()

    joined = pd.concat(
        [last_4h.rename("c4"), f1["close"].rename("c1")],
        axis=1
        ).dropna()

    if joined.empty:
        check("sa wspolne dni do porownania", False)
        return

    diff = (joined["c4"] - joined["c1"]).abs()
    rel = diff / joined["c1"]
    check(f"close zgodny na {len(joined)} dniach",
          rel.max() < 1e-9,
          f"max blad wzgledny {rel.max():.2e}"
          )


def check_target(f4, 
                 f1
                 ):
    print("")
    print("[5] Target")
    for tf, f in [("4h", f4), ("1d", f1)]:
        t = f["target"].astype(float)
        check(f"{tf}: brak NaN",
              bool(t.notna().all()))
        check(f"{tf}: tylko wartosci 0/1",
              set(t.unique()) <= {0.0, 1.0})
        share = t.mean()
        check(f"{tf}: rozklad klas 40-60%",
              0.40 < share < 0.60,
              f"long {share:.1%}")


def check_sentiment(f4, 
                    f1
                    ):
    print("")
    print("[6] Sentyment")
    s4 = [c for c in f4.columns
          if "sentiment" in c or "n_texts" in c]
    s1 = [c for c in f1.columns
          if "sentiment" in c or "n_texts" in c]

    check("ta sama liczba kolumn sentymentu",
          len(s4) == len(s1),
          f"4h={len(s4)} 1d={len(s1)}"
          )

    score = f1["reddit_sentiment_score"].abs().sum()
    check("reddit_sentiment_score niezerowy w 1d",
          score > 0
          )

    n1 = f1["reddit_n_texts"].mean()
    n4 = f4["reddit_n_texts"].mean()
    ratio = n1 / max(n4, 1e-9)
    check("reddit_n_texts w 1d to suma doby",
          3.0 < ratio < 9.0,
          f"stosunek srednich {ratio:.2f}, oczekiwane ~6"
          )


def check_constant(d4, 
                   d1
                   ):
    print("")
    print("[7] Kolumny stale w train")
    for tf, d in [("4h", d4), ("1d", d1)]:
        tr = d["train"].select_dtypes("number")
        const = []
        for c in tr.columns:
            if tr[c].nunique(dropna=True) <= 1:
                const.append(c)

        print(f"  {tf}: {len(const)} kolumn stalych")
        for c in sorted(const):
            print("        " + c)

        odd = sorted(set(const) - CONST_OK)
        check(f"{tf}: brak nieoczekiwanych stalych",
              not odd,
              str(odd)
              )


def check_nan(d4, 
              d1
              ):
    print("")
    print("[8] Braki danych w train")
    for tf, d in [("4h", d4), ("1d", d1)]:
        tr = d["train"]
        cells = len(tr) * len(tr.columns)
        frac = tr.isna().sum().sum() / cells
        worst = tr.isna().mean().sort_values(ascending=False)

        print(f"  {tf}: {frac:.1%} brakujacych komorek")
        for c, v in worst.head(3).items():
            print(f"        {c}: {v:.0%}")

        check(f"{tf}: braki ponizej 30%",
              frac < 0.30,
              f"{frac:.1%}"
              )


def run(coin):
    print("")
    print("=" * 60)
    print("  " + coin)
    print("=" * 60)

    d4 = load("4h", 
              coin
              )
    d1 = load("1d", 
              coin
              )
    f4 = full(d4)
    f1 = full(d1)

    print(f"  4h: {len(f4)} wierszy x {len(f4.columns)} kolumn")
    print(f"  1d: {len(f1)} wierszy x {len(f1.columns)} kolumn")

    check_schema(d4, 
                 d1
                 )
    check_splits(d4, 
                 d1
                 )
    check_index(f4, 
                f1
                )
    check_price(f4, 
                f1
                )
    check_target(f4, 
                 f1
                 )
    check_sentiment(f4, 
                    f1
                    )
    check_constant(d4, 
                   d1
                   )
    check_nan(d4, 
              d1
              )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coins", 
                   default="BTC,ETH"
                   )
    args = p.parse_args()

    for coin in args.coins.split(","):
        run(coin)

    print("")
    print("=" * 60)
    if FAILS:
        print(f"  NIEPOWODZENIA: {len(FAILS)}")
        for f in FAILS:
            print("    - " + f)
    else:
        print("  Wszystkie kontrole przeszly.")
    print("=" * 60)


if __name__ == "__main__":
    main()
