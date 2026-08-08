"""

CryptoBERT Embedder: generuje embeddingi sentymentu z tekstów krypto.

Model: ElKulako/cryptobert (HuggingFace)
  - Fine-tuned BERT na ~3.2M tweetach o kryptowalutach
  - Klasy: Bearish (0), Neutral (1), Bullish (2)
  - Wyjście modelu: logity 3-klasowe + hidden states (768-dim)

"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "ElKulako/cryptobert"
PROCESSED_DIR = Path("data/processed/sentiment_embeddings")
OutputMode = Literal["scores", 
                     "embeddings", 
                     "both"
                     ]

LABEL_MAP = {0: "bearish", 
             1: "neutral", 
             2: "bullish"
             }


class CryptoBERTEmbedder:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 128,
        output_dir: Path = PROCESSED_DIR,
        ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, 
                              exist_ok=True
                              )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            output_hidden_states=True
            )
        self.model.to(self.device)
        self.model.eval()
        print(f"[CryptoBERT] Model załadowany na {self.device}")



    def encode_texts(
        self,
        texts: list[str],
        mode: OutputMode = "both",
    ) -> dict[str, np.ndarray]:
        """
        Metoda koduje listę tekstów przez CryptoBERT.
        """
        all_scores: list[np.ndarray] = []
        all_embeddings: list[np.ndarray] = []

        for i in range(0, 
                       len(texts), 
                       self.batch_size
                       ):
            batch = texts[i : i + self.batch_size]
            scores, embeddings = self._encode_batch(batch)
            all_scores.append(scores)
            if mode in ("embeddings", "both"):
                all_embeddings.append(embeddings)

        result: dict[str, np.ndarray] = {}

        if all_scores:
            scores_arr = np.vstack(all_scores)
            result["scores"] = scores_arr
            result["sentiment"] = scores_arr[:, 2] - scores_arr[:, 0]

        if all_embeddings:
            result["embeddings"] = np.vstack(all_embeddings)

        return result

    def encode_window_dataframe(
        self,
        windows_df: pd.DataFrame,
        texts_col: str = "texts",
        mode: OutputMode = "both",
        aggregation: str = "mean"
        ) -> pd.DataFrame:
        """
        Metoda koduje DataFrame z okienkami tekstów (output text_aggregator.py).
        Każde okno (lista tekstów) -> zagregowany wektor sentymentu.

        """
        records: list[dict] = []

        for timestamp, row in windows_df.iterrows():
            texts = row[texts_col]
            if not texts:
                continue

            encoded = self.encode_texts(texts, 
                                        mode=mode
                                        )
            record: dict = {"timestamp": timestamp}

            if "scores" in encoded:
                scores = encoded["scores"]  # (N, 3)
                record["sentiment_bearish_mean"] = float(scores[:, 0].mean())
                record["sentiment_neutral_mean"] = float(scores[:, 1].mean())
                record["sentiment_bullish_mean"] = float(scores[:, 2].mean())
                record["sentiment_score"] = float(encoded["sentiment"].mean())
                record["sentiment_score_std"] = float(encoded["sentiment"].std())
                record["sentiment_dispersion"] = float(
                    (scores[:, 0] > 0.5).mean() + (scores[:, 2] > 0.5).mean()
                    )

            if "embeddings" in encoded:
                embs = encoded["embeddings"]  # (N, 768)
                if aggregation == "mean":
                    agg_emb = embs.mean(axis=0)
                elif aggregation == "max":
                    agg_emb = embs.max(axis=0)
                else:
                    agg_emb = embs.mean(axis=0)

                for j, val in enumerate(agg_emb):
                    record[f"emb_{j}"] = float(val)

            records.append(record)

        result = pd.DataFrame(records).set_index("timestamp")
        result.index = pd.to_datetime(result.index, 
                                      utc=True
                                      )
        return result




    @torch.no_grad()
    def _encode_batch(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Metoda koduje jeden batch tekstów"""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
            )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        scores = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        cls_emb = outputs.hidden_states[-1][:, 0, :].cpu().numpy()

        return scores, cls_emb
