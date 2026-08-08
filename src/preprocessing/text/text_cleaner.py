"""
Text Cleaner: czyszczenie surowych tekstów przed użyciem modelu CryptoBERT.

CryptoBERT (ElKulako/cryptobert) był trenowany na tweetach o kryptowalutach.
Optymalnym wejściem jest tekst po podstawowym czyszczeniu, ale z zachowaniem
$TICKER, #hashtag i sygnałów sentymentu (np. "🚀", "💀").
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

# Wzorce do usunięcia / normalizacji
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_TICKER_RE = re.compile(r"\$([A-Z]{2,6})")
_HASHTAG_RE = re.compile(r"#(\w+)")
_WHITESPACE_RE = re.compile(r"\s+")
_NON_ASCII_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str, 
               keep_emojis: bool = True
               ) -> str:
    """
    Funkcja czyści pojedynczy tekst przed podaniem do CryptoBERTa.
    Kroki:
      1. Usuń URL-e
      2. Usuń @mentions
      3. Zachowaj $TICKER jako słowo (usuń tylko $)
      4. Zachowaj #hashtag jako słowo (usuń tylko #)
      5. Znormalizuj whitespace
      6. Usuń znaki kontrolne
    """
    if not isinstance(text, 
                      str
                      ) or not text.strip():
        return ""

    text = _URL_RE.sub(" ", 
                       text
                       )
    text = _MENTION_RE.sub(" ", 
                           text
                           )
    text = _TICKER_RE.sub(r"\1", 
                          text
                          )      # $BTC -> BTC
    text = _HASHTAG_RE.sub(r"\1", 
                           text
                           )
    text = _NON_ASCII_CONTROL_RE.sub(" ", 
                                     text
                                     )

    if not keep_emojis:
        text = _remove_emojis(text)

    text = _WHITESPACE_RE.sub(" ", 
                              text
                              ).strip()
    return text


def clean_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    title_col: str | None = "title",
    min_length: int = 10,
    keep_emojis: bool = True
    ) -> pd.DataFrame:
    """
    Funkcja czyści kolumnę lub kolumny tekstowe w DataFrame.
    Jeśli dostępny jest zarówno title jak i text (np. Reddit), łączy je.
    Usuwa wiersze z tekstem krótszym niż min_length znaków po czyszczeniu.
    """
    df = df.copy()

    if title_col and title_col in df.columns and text_col in df.columns:
        df["combined_text"] = (
            df[title_col].fillna("").apply(clean_text, 
                                           keep_emojis=keep_emojis
                                           )
            + " "
            + df[text_col].fillna("").apply(clean_text, 
                                            keep_emojis=keep_emojis
                                            )).str.strip()
    elif text_col in df.columns:
        df["combined_text"] = df[text_col].fillna("").apply(
            lambda t: clean_text(t, 
                                 keep_emojis=keep_emojis
                                 ))
    else:
        raise ValueError(f"Brak kolumny '{text_col}' w DataFrame")

    # Odfiltruj zbyt krótkie teksty
    df = df[df["combined_text"].str.len() >= min_length].copy()
    df = df.reset_index(drop=True)
    return df


def _remove_emojis(text: str) -> str:
    """Funkcja usuwa emoji i znaki spoza Basic Multilingual Plane"""
    return "".join(
        c for c in text
        if unicodedata.category(c) not in ("So", "Cs") and ord(c) <= 0xFFFF
    )
