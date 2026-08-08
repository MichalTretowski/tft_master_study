"""
Text Aggregator: agregacja postów do wybranych interwałów czasowych.
Ze zbioru tekstów z timestampem stwórz listę tekstów na okno czasowe (1h, 4h).
Wynik jest wejściem do CryptoBERT Embedder, który przetworzy każde okno jako batch.

"""

from __future__ import annotations

from typing import Literal

import pandas as pd

AggStrategy = Literal["top_k_by_engagement", "all", "sample"]


def aggregate_texts_to_windows(
    df: pd.DataFrame,
    freq: str = "4h",
    text_col: str = "combined_text",
    timestamp_col: str = "timestamp",
    engagement_col: str | None = "score",
    strategy: AggStrategy = "top_k_by_engagement",
    top_k: int = 20,
    sample_n: int = 20,
    min_texts_per_window: int = 1
    ) -> pd.DataFrame:
    """
    Funkcja grupuje teksty do okien czasowych (freq) i zwraca DataFrame z listą tekstów na okno.

    Parametry:
        freq: częstotliwość okna ('1h', '4h', '1d')
        text_col: kolumna z oczyszczonym tekstem
        timestamp_col: kolumna z timestampem
        engagement_col: kolumna do sortowania w strategii top_k (None = brak sortowania)
        strategy: jak wybierać teksty z okna
        top_k: liczba tekstów przy strategii top_k_by_engagement
        sample_n: liczba tekstów przy strategii sample
        min_texts_per_window: minimalna liczba tekstów, by okno było uwzględnione

    Zwraca DataFrame z kolumnami:
        timestamp
        texts
        n_texts
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], 
                                       utc=True
                                       )
    df = df.dropna(subset=[text_col, 
                           timestamp_col
                           ])
    df = df[df[text_col].str.strip().astype(bool)]
    df = df.set_index(timestamp_col).sort_index()

    def _select_texts(group: pd.DataFrame) -> list[str]:
        if strategy == "top_k_by_engagement" and engagement_col and engagement_col in group.columns:
            group = group.nlargest(top_k, 
                                   engagement_col
                                   )
        elif strategy == "sample":
            group = group.sample(min(sample_n, 
                                     len(group)), 
                                     random_state=42
                                    )
        return group[text_col].tolist()

    # Grupowanie i selekcja
    records: list[dict] = []
    for window_start, group in df.resample(freq):
        if len(group) < min_texts_per_window:
            continue
        texts = _select_texts(group)
        records.append({
            "timestamp": window_start,
            "texts": texts,
            "n_texts": len(texts)
        })

    result = pd.DataFrame(records)
    if result.empty:
        return result

    result["timestamp"] = pd.to_datetime(result["timestamp"], 
                                         utc=True
                                         )
    result = result.set_index("timestamp").sort_index()
    return result
