"""

Inspekcja kolumn datasetu - przygotowanie input_schema.py.

Mapuje kazda kolumne datasetu na plik posredni, z ktorego pochodzi
(data/processed/features/*), dzieli na static / known / observed
i wypisuje gotowy fragment do wklejenia w input_schema.py.

"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROCESSED = Path("data/processed")
FEATURES = PROCESSED / "features"
DATASETS = PROCESSED / "datasets"
SENTIMENT = (PROCESSED / "sentiment_embeddings"
             / "sentiment_combined_4h.parquet")

# Kolumny znane z wyprzedzeniem
KNOWN_TIME = {"hour_sin", 
              "hour_cos", 
              "dow_sin", 
              "dow_cos"
              }

STATIC = {"coin_id"}
TARGET = {"target"}


def source_files(tf, 
                 coin
                 ):
    """Mapa: etykieta zrodla -> sciezka pliku posredniego."""
    return {
        "ohlcv+technical": FEATURES / f"{coin}_{tf}_features.parquet",
        "fred": FEATURES / f"fred_macro_{tf}.parquet",
        "cot": FEATURES / f"cot_{tf}.parquet",
        "fear_greed": FEATURES / f"fear_greed_{tf}.parquet",
        "google_trends": FEATURES / f"google_trends_{tf}.parquet",
        "correlated": FEATURES / f"correlated_{tf}.parquet",
        "onchain": FEATURES / f"onchain_{coin}_{tf}.parquet",
        "derivatives": FEATURES / f"derivatives_{coin}_{tf}.parquet",
        "stablecoin": FEATURES / f"stablecoin_supply_{tf}.parquet",
        "dominance": FEATURES / f"market_dominance_{tf}.parquet",
        "event_calendar": FEATURES / f"event_calendar_{tf}.parquet",
        "sentiment": SENTIMENT
        }


def build_origin_map(tf, 
                     coin
                     ):
    """Kolumna -> etykieta zrodla"""
    origin = {}
    for label, path in source_files(tf, coin).items():
        if not path.exists():
            print(f"  Brak {path.name} - pomijam")
            continue
        cols = pd.read_parquet(path).columns
        for c in cols:
            origin.setdefault(c, label)
    return origin


def classify(col, 
             origin
             ):
    if col in TARGET:
        return "target"
    if col in STATIC:
        return "static"
    if col in KNOWN_TIME:
        return "known"
    if origin.get(col) == "event_calendar":
        return "known"
    return "observed"


def emit_list(name, 
              cols, 
              origin
              ):
    """Funkcja wypisuje gotową listę z komentarzami grupujacymi"""
    print(f"    {name}: list[str] = field(default_factory=lambda: [")
    last = None
    for c in cols:
        src = origin.get(c, "???")
        if src != last:
            print(f"        # --- {src} ---")
            last = src
        print(f'        "{c}",')
    print("    ])")
    print("")


def run(tf, 
        coin
        ):
    path = DATASETS / tf / f"{coin}_train.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Brak {path}")

    df = pd.read_parquet(path)
    origin = build_origin_map(tf, 
                              coin
                              )

    print("")
    print("=" * 60)
    print(f"  {coin} / {tf} — {len(df.columns)} kolumn")
    print("=" * 60)

    groups = {"static": [], 
              "known": [], 
              "observed": [], 
              "target": []
              }
    unmapped = []

    for c in df.columns:
        groups[classify(c, origin)].append(c)
        if c not in origin and c not in STATIC and c not in TARGET:
            unmapped.append(c)

    print("")
    print("[Podzial]")
    for k in ["static", "known", "observed", "target"]:
        print(f"  {k:<10} {len(groups[k]):>4}")

    print("")
    print("[Zrodla kolumn obserwowanych]")
    counts = {}
    for c in groups["observed"]:
        src = origin.get(c, "???")
        counts[src] = counts.get(src, 0) + 1
    for src in sorted(counts, key=lambda s: -counts[s]):
        print(f"  {src:<18} {counts[src]:>4}")

    if unmapped:
        print("")
        print("[UWAGA] kolumny bez zrodla - sprawdz recznie:")
        for c in unmapped:
            print(f"  {c}")

    # posortuj obserwowane wg zrodla
    order = list(source_files(tf, coin).keys())

    def key(c):
        src = origin.get(c, "???")
        return (order.index(src) if src in order else 99, c)

    obs = sorted(groups["observed"], key=key)
    known = sorted(groups["known"], key=key)

    print("")
    print("=" * 60)
    print("  Fragment do input_schema.py")
    print("=" * 60)
    print("")
    print(f'    static_categoricals: list[str] = field('
          f'default_factory=lambda: {groups["static"]})')
    print("")
    emit_list("known_reals", 
              known, 
              origin
              )
    emit_list("observed_reals", 
              obs, 
              origin
              )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tf", 
                   default="4h", 
                   choices=["4h", "1d"]
                   )
    p.add_argument("--coin", 
                   default="BTC"
                   )
    args = p.parse_args()
    run(args.tf, args.coin)


if __name__ == "__main__":
    main()
