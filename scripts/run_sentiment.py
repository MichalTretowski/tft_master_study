"""
Skrypt: Uruchom pipeline sentymentu CryptoBERT na danych Reddit.

Przepływ:
  1. Wczytuje 4 źródła Reddit (posts + comments, Bitcoin + CryptoCurrency)
  2. Oczyszcza teksty (text_cleaner)
  3. Dla każdego źródła: agreguje do okien 4h (top-20 na okno wg score)
  4. Koduje przez CryptoBERT (ElKulako/cryptobert) na GPU
  5. Łączy 4 źródła w jeden sygnał "reddit" na okno 4h
  6. Dodaje lagi i rolling averages
  7. Zapisuje do data/processed/sentiment_embeddings/sentiment_combined_4h.parquet

Przetworzone na RTX 4070 Laptop, batch_size=128, top_k=20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

RAW_DIR = Path("data/raw/text/reddit")
CACHE_DIR = Path("data/cache")
PROCESSED_DIR = Path("data/processed/sentiment_embeddings")

REDDIT_SOURCES = [
    ("posts_bitcoin.parquet", 
     "selftext", 
     "title", 
     "score", 
     "posts_btc"
     ),
    ("posts_cryptocurrency.parquet", 
     "selftext", 
     "title", 
     "score", 
     "posts_cc"
     ),
    ("posts_ethereum.parquet", 
     "selftext", 
     "title", 
     "score", 
     "posts_eth"
     ),
    ("posts_cryptomarkets.parquet", 
     "selftext", 
     "title", 
     "score", 
     "posts_cm"
     ),
    ("comments_bitcoin.parquet", 
     "body", 
     None, 
     "score", 
     "comments_btc"
     ),
    ("comments_cryptocurrency.parquet",
     "body",
     None, 
     "score", 
     "comments_cc"
     )]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--force", 
                   action="store_true", 
                   help="Ignoruj cache, przelicz od zera"
                   )
    p.add_argument("--dry-run", 
                   action="store_true", 
                   help="Policz okna/teksty, nie uruchamiaj GPU"
                   )
    p.add_argument("--batch-size", 
                   type=int, 
                   default=128, 
                   help="Batch size dla CryptoBERT (domyślnie 128)"
                   )
    p.add_argument("--top-k", 
                   type=int, 
                   default=20, 
                   help="Max tekstów per okno 4h (domyślnie 20)"
                   )
    return p.parse_args()


def load_reddit_source(
    filename: str,
    text_col: str,
    title_col: str | None,
    timestamp_col: str = "timestamp"
    ) -> pd.DataFrame | None:
    path = RAW_DIR / filename
    if not path.exists():
        print(f"[Skip] Brak pliku: {path}")
        return None

    df = pd.read_parquet(path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], 
                                       utc=True
                                       )

    # Ujednolicona kolumna tekstowa
    if title_col and title_col in df.columns:
        df["text"] = (df[title_col].fillna("") + " " + df[text_col].fillna("")).str.strip()
    else:
        df["text"] = df[text_col].fillna("")

    df = df[["text", "score", timestamp_col]].rename(columns={timestamp_col: "timestamp"})
    df = df[df["text"].str.len() >= 10].reset_index(drop=True)
    print(f"[Load] {filename}: {len(df):,} rekordów")
    return df


def dry_run(top_k: int) -> None:
    from src.preprocessing.text.text_cleaner import clean_dataframe
    from src.preprocessing.text.text_aggregator import aggregate_texts_to_windows

    total_windows = 0
    total_texts   = 0

    for filename, text_col, title_col, _, label in REDDIT_SOURCES:
        df = load_reddit_source(filename, 
                                text_col, 
                                title_col
                                )
        if df is None:
            continue

        clean_df  = clean_dataframe(df, 
                                    text_col="text", 
                                    title_col=None
                                    )
        windows   = aggregate_texts_to_windows(clean_df, 
                                               freq="4h", 
                                               text_col="combined_text", 
                                               engagement_col="score", 
                                               top_k=top_k
                                               )
        n_texts = windows["n_texts"].sum()
        print(f"  {label:<20} {len(windows):>6} okien 4h, {int(n_texts):>8,} tekstów")
        total_windows += len(windows)
        total_texts   += int(n_texts)

    print(f"\n  ŁĄCZNIE: {total_windows:,} okien, {total_texts:,} tekstów")
    print(f"  Szacowany czas GPU (RTX 4070, batch=128): ~{total_texts / 5000:.0f} min")


def encode_source(
    label: str,
    df: pd.DataFrame,
    embedder,
    top_k: int,
    force: bool
    ) -> pd.DataFrame | None:
    """Oczyszcza, agreguje do 4h, koduje przez CryptoBERT. Zwraca DataFrame na okno."""
    from src.preprocessing.text.text_cleaner import clean_dataframe
    from src.preprocessing.text.text_aggregator import aggregate_texts_to_windows

    cache_path = CACHE_DIR / f"sentiment_{label}_4h.parquet"

    if not force and cache_path.exists():
        print(f"[Cache] {label}: wczytano ({cache_path})")
        return pd.read_parquet(cache_path)

    print(f"\n[Encode] {label}: czyszczenie...")
    clean_df = clean_dataframe(df, 
                               text_col="text", 
                               title_col=None
                               )
    print(f"[Encode] {label}: {len(clean_df):,} tekstów po czyszczeniu")

    print(f"[Encode] {label}: agregacja do okien 4h (top_k={top_k})...")
    windows = aggregate_texts_to_windows(clean_df, 
                                         freq="4h", 
                                         text_col="combined_text", 
                                         engagement_col="score", 
                                         top_k=top_k
                                         )
    n_texts = int(windows["n_texts"].sum())
    print(f"[Encode] {label}: {len(windows):,} okien, {n_texts:,} tekstów łącznie")

    print(f"[Encode] {label}: CryptoBERT encoding...")
    emb_df = embedder.encode_window_dataframe(windows, 
                                              mode="scores"
                                              )
    print(f"[Encode] {label}: gotowe — {len(emb_df)} wierszy")

    emb_df.to_parquet(cache_path)
    print(f"[Encode] {label}: cache zapisany -> {cache_path}")
    return emb_df


def merge_reddit_sources(source_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Funkcja łączy 4 źródła Reddit w jeden sygnał 'reddit' na okno 4h.
    Strategia: średnia ważona (wagi: posts_btc=1, posts_cc=1, comments_btc=2, comments_cc=2)
    """
    WEIGHTS = {
        "posts_btc": 1.0,
        "posts_cc": 1.0,
        "posts_eth": 1.0,
        "posts_cm": 1.0,
        "comments_btc": 2.0,
        "comments_cc": 2.0
        }
    

    all_indices = pd.DatetimeIndex(
        pd.concat([pd.Series(index=df.index, dtype=float) for df in source_frames.values()])
        .index.unique()
        .sort_values()
    )

    weighted_sum = pd.Series(0.0, 
                             index=all_indices
                             )
    weight_total = pd.Series(0.0, 
                             index=all_indices
                             )
    n_texts_total = pd.Series(0, 
                              index=all_indices
                              )

    for label, df in source_frames.items():
        w = WEIGHTS.get(label, 
                        1.0
                        )
        score_col = "sentiment_score"
        n_texts_col = "n_texts"

        if score_col not in df.columns:
            print(f"[Merge] Brak kolumny '{score_col}' w {label} — pomijam")
            continue

        df_reindexed = df.reindex(all_indices).ffill()
        weighted_sum = weighted_sum.add(df_reindexed[score_col].fillna(0.0) * w, 
                                        fill_value=0.0
                                        )
        weight_total = weight_total.add(df_reindexed[score_col].notna().astype(float) * w, 
                                        fill_value=0.0
                                        )
        if n_texts_col in df.columns:
            n_texts_total = n_texts_total.add(df_reindexed[n_texts_col].fillna(0).astype(int), 
                                              fill_value=0
                                              )

    sentinel = weight_total > 0
    combined_score = pd.Series(0.0, 
                               index=all_indices
                               )
    combined_score[sentinel] = weighted_sum[sentinel] / weight_total[sentinel]

    result = pd.DataFrame({
        "reddit_sentiment_score": combined_score,
        "reddit_n_texts": n_texts_total,
        }, 
        index=all_indices)
    result.index.name = "timestamp"
    return result


def add_lags_and_rolling(df: pd.DataFrame, 
                         score_col: str = "reddit_sentiment_score"
                         ) -> pd.DataFrame:
    df = df.copy()
    for lag in [1, 6, 24, 42]:
        df[f"{score_col}_lag{lag}"] = df[score_col].shift(lag)
    for window, label in [(6, "1d"), (42, "7d"), (180, "30d")]:
        df[f"{score_col}_roll_{label}"] = df[score_col].rolling(window, min_periods=1).mean()
    return df


def build_combined_output(reddit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Funkcja buduje finalny DataFrame z kolumnami zgodnymi ze schematem TFT.
    """
    df = reddit_df.copy()
    df["sentiment_score_combined"] = df["reddit_sentiment_score"]
    df["twitter_sentiment_score"] = 0.0   # brak danych
    df["news_sentiment_score"] = 0.0   # brak danych
    df["twitter_n_texts"] = 0
    df["news_n_texts"] = 0

    # Lagi i rolling dla sentiment_score_combined (schemat TFT)
    for lag in [1, 6, 24, 42]:
        df[f"sentiment_score_combined_lag{lag}"] = df["sentiment_score_combined"].shift(lag)
    for window, label in [(6, "1d"), (42, "7d"), (180, "30d")]:
        df[f"sentiment_score_combined_roll_{label}"] = (
            df["sentiment_score_combined"].rolling(window, min_periods=1).mean()
            )

    return df


def main() -> None:
    args = parse_args()
    CACHE_DIR.mkdir(parents=True, 
                    exist_ok=True
                    )
    PROCESSED_DIR.mkdir(parents=True, 
                        exist_ok=True
                        )

    if args.dry_run:
        print("=== DRY RUN — szacowanie okien i tekstów ===")
        dry_run(top_k=args.top_k)
        return

    # Załaduj model CryptoBERT na GPU
    from src.sentiment.cryptobert_embedder import CryptoBERTEmbedder
    embedder = CryptoBERTEmbedder(batch_size=args.batch_size)

    source_frames: dict[str, pd.DataFrame] = {}

    for filename, text_col, title_col, _, label in REDDIT_SOURCES:
        raw_df = load_reddit_source(filename, 
                                    text_col, 
                                    title_col
                                    )
        if raw_df is None:
            continue
        emb_df = encode_source(label, 
                               raw_df, 
                               embedder, 
                               args.top_k, 
                               args.force
                               )
        if emb_df is not None:
            source_frames[label] = emb_df

    if not source_frames:
        print("[Error] Brak wyników — sprawdź pliki źródłowe")
        return

    print("\n[Merge] Łączę 4 źródła Reddit w jeden sygnał...")
    reddit_df = merge_reddit_sources(source_frames)
    reddit_df = add_lags_and_rolling(reddit_df)

    print("[Build] Buduję finalny DataFrame zgodny ze schematem TFT...")
    combined = build_combined_output(reddit_df)

    out_path = PROCESSED_DIR / "sentiment_combined_4h.parquet"
    combined.to_parquet(out_path)
    print(f"\n[Done] {len(combined):,} wierszy × {len(combined.columns)} kolumn -> {out_path}")
    print(f"[Done] Zakres: {combined.index.min().date()} -> {combined.index.max().date()}")
    print(f"[Done] Kolumny sentymentu: {[c for c in combined.columns if 'sentiment' in c or 'n_texts' in c]}")


if __name__ == "__main__":
    main()
