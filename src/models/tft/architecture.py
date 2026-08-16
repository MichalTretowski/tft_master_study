"""

TFT Architecture: Temporal Fusion Transformer z głowicą klasyfikacyjną.

Modyfikacje względem oryginalnego TFT:
  1. Głowica: classification head (sigmoid) zamiast quantile regression
  2. Loss: Differentiable Sharpe Ratio
  3. Output: pozycja tradingowa ∈ [-1, 1] przez Tanh (opcja DSR)
             lub P(up) ∈ [0, 1] przez Sigmoid (opcja BCE)

Architektura:
  [Static inputs] ──────────────────────────────────┐
                                                    ▼
  [Known future]  ──► VSN ──► LSTM encoder  ──► Attention ──► GRN ──► head
                                                    ▲
  [Observed past] ──► VSN ──► LSTM encoder ─────────┘

  """

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.tft.components import (
    GateAddNorm,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    VariableSelectionNetwork
    )
from src.models.tft.input_schema import TFTInputSchema


class TemporalFusionTransformer(nn.Module):
    def __init__(
        self,
        schema: TFTInputSchema,
        hidden_size: int = 128,
        lstm_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        output_mode: str = "position",
        embedding_dim_per_categorical: int = 8
        ) -> None:
        super().__init__()
        self.schema = schema
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.embedding_dim_per_categorical = embedding_dim_per_categorical

        n_coins = 2
        self.coin_embedding = nn.Embedding(n_coins, 
                                           embedding_dim_per_categorical
                                           )
        static_size = embedding_dim_per_categorical * max(1, len(schema.static_categoricals))

        n_known = len(schema.known_reals)
        n_observed = len(schema.observed_reals)

        self.known_projections = nn.ModuleList([
            nn.Linear(1, hidden_size) for _ in range(n_known)
        ])
        self.observed_projections = nn.ModuleList([
            nn.Linear(1, hidden_size) for _ in range(n_observed)
        ])


        self.static_encoder = GatedResidualNetwork(
            input_size=static_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout
            )

        self.static_context_variable_selection = nn.Linear(hidden_size, 
                                                           hidden_size
                                                           )
        self.static_context_enrichment = nn.Linear(hidden_size, 
                                                   hidden_size
                                                   )
        self.static_context_state_h = nn.Linear(hidden_size, 
                                                hidden_size
                                                )
        self.static_context_state_c = nn.Linear(hidden_size, 
                                                hidden_size
                                                )


        self.encoder_vsn = VariableSelectionNetwork(
            input_sizes=[hidden_size] * n_observed,
            hidden_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size
            )
        self.decoder_vsn = VariableSelectionNetwork(
            input_sizes=[hidden_size] * n_known,
            hidden_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size
            )


        self.encoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
            )
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
            )
        self.lstm_gate_add_norm = GateAddNorm(hidden_size, 
                                              hidden_size, 
                                              dropout
                                              )


        self.static_enrichment = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size
            )


        self.attention = InterpretableMultiHeadAttention(
            d_model=hidden_size,
            n_heads=n_heads,
            dropout=dropout
            )
        self.attention_gate_add_norm = GateAddNorm(hidden_size, 
                                                   hidden_size, 
                                                   dropout
                                                   )


        self.positionwise_grn = GatedResidualNetwork(hidden_size, 
                                                     hidden_size, 
                                                     dropout=dropout
                                                     )
        self.final_gate_add_norm = GateAddNorm(hidden_size, 
                                               hidden_size, 
                                               dropout
                                               )


        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
            )


    def forward(
        self,
        observed: torch.Tensor,
        known_future: torch.Tensor,
        static_cat: torch.Tensor
        ) -> dict[str, torch.Tensor]:
        """
        Parametry:
            observed: historyczne observed covariates (encoder input)
            known_future: future known covariates (decoder input)
            static_cat: kategoryczne static covariates (coin_id)

        Zwraca słownik:
            'output': (B, 1) - pozycja tradingowa lub P(up)
            'encoder_weights': (B, encoder_len, n_observed) - VSN weights
            'attention_weights': (B, decoder_len, encoder_len+decoder_len)
        """
        B = observed.size(0)


        coin_emb = self.coin_embedding(static_cat)
        static_context = self.static_encoder(coin_emb)

        ctx_vs = self.static_context_variable_selection(static_context)
        ctx_enrich = self.static_context_enrichment(static_context)
        ctx_h = self.static_context_state_h(static_context)
        ctx_c = self.static_context_state_c(static_context)


        obs_projected = self._project_inputs(observed, 
                                             self.observed_projections
                                             )
        known_projected = self._project_inputs(known_future, 
                                               self.known_projections
                                               )


        enc_selected, enc_weights = self.encoder_vsn(
            [obs_projected[:, :, i, :] for i in range(obs_projected.size(2))],
            context=ctx_vs.unsqueeze(1).expand(-1, obs_projected.size(1), -1)
            )
        dec_selected, _ = self.decoder_vsn(
            [known_projected[:, :, i, :] for i in range(known_projected.size(2))],
            context=ctx_vs.unsqueeze(1).expand(-1, known_projected.size(1), -1)
            )


        h0 = ctx_h.unsqueeze(0).expand(self.encoder_lstm.num_layers, 
                                       -1, 
                                       -1
                                       ).contiguous()
        c0 = ctx_c.unsqueeze(0).expand(self.encoder_lstm.num_layers, 
                                       -1, 
                                       -1
                                       ).contiguous()

        enc_out, (hn, cn) = self.encoder_lstm(enc_selected, 
                                              (h0, c0)
                                              )
        dec_out, _ = self.decoder_lstm(dec_selected, 
                                       (hn, cn)
                                       )

        lstm_out = torch.cat([enc_out, dec_out], 
                             dim=1
                             )
        vsn_out = torch.cat([enc_selected, dec_selected], 
                            dim=1
                            )
        lstm_out = self.lstm_gate_add_norm(lstm_out, 
                                           residual=vsn_out
                                           )


        enriched = self.static_enrichment(
            lstm_out,
            context=ctx_enrich.unsqueeze(1).expand(-1, lstm_out.size(1), -1)
            )

        seq_len = enriched.size(1)
        causal_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=enriched.device
                )
            )

        attn_out, attn_weights = self.attention(
            enriched,
            enriched,
            enriched,
            mask=causal_mask
            )
        attn_out = self.attention_gate_add_norm(attn_out, 
                                                residual=enriched
                                                )


        ff_out = self.positionwise_grn(attn_out)
        ff_out = self.final_gate_add_norm(ff_out, 
                                          residual=attn_out
                                          )

        enc_last = ff_out[:,
                          -self.schema.decoder_length - 1,
                          :
                          ]
        dec_last = ff_out[:,
                          -1,
                          :
                          ]

        final = torch.cat([enc_last, dec_last], dim=-1)
        logit = self.output_head(final)

        return {
            "logit": logit,
            "encoder_weights": enc_weights,
            "attention_weights": attn_weights
            }


    @staticmethod
    def _project_inputs(
        x: torch.Tensor,
        projections: nn.ModuleList
        ) -> torch.Tensor:
        """
        Projektuje każdą zmienną osobno do hidden_size.
        x: (B, T, n_vars) -> output: (B, T, n_vars, hidden_size)
        """
        projected = [proj(x[:, :, i].unsqueeze(-1)) for i, proj in enumerate(projections)]
        return torch.stack(projected, 
                           dim=2
                           )

    def get_variable_importance(
        self, encoder_weights: torch.Tensor
        ) -> dict[str, float]:
        """Zwraca średnie wagi VSN jako słownik {nazwa_cechy: waga}"""
        avg_weights = encoder_weights.mean(dim=(0, 1)).cpu().detach().numpy()
        return {
            col: float(avg_weights[i])
            for i, col in enumerate(self.schema.observed_reals)
            }
