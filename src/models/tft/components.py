"""
TFT Components: bloki składowe Temporal Fusion Transformer.

Implementacja na podstawie:
  Lim et al. (2021) "Temporal Fusion Transformers for Interpretable
  Multi-horizon Time Series Forecasting"
  https://arxiv.org/abs/1912.09363

Komponenty:
  - GatedLinearUnit (GLU)
  - GateAddNorm (GLU + residual + LayerNorm)
  - GatedResidualNetwork (GRN)
  - VariableSelectionNetwork (VSN)
  - MultiHeadAttention (z maską kauzalną)
  - TemporalAttentionLayer
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLinearUnit(nn.Module):
    """GLU: split -> linear gate z sigmoid"""

    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 dropout: float = 0.0
                 ) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, 
                            hidden_size * 2
                            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                x: torch.Tensor
                ) -> torch.Tensor:
        x = self.dropout(x)
        x = self.fc(x)
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)


class GateAddNorm(nn.Module):
    """GLU + residual connection + LayerNorm"""

    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 dropout: float = 0.0
                 ) -> None:
        super().__init__()
        self.glu = GatedLinearUnit(input_size, 
                                   hidden_size, 
                                   dropout
                                   )
        self.norm = nn.LayerNorm(hidden_size)
        self.skip = nn.Linear(input_size, 
                              hidden_size
                              ) if input_size != hidden_size else nn.Identity()

    def forward(self, 
                x: torch.Tensor, 
                residual: torch.Tensor | None = None
                ) -> torch.Tensor:
        if residual is None:
            residual = self.skip(x)
        return self.norm(self.glu(x) + residual)


class GatedResidualNetwork(nn.Module):
    """ GRN """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int | None = None,
        dropout: float = 0.1,
        context_size: int | None = None
        ) -> None:
        super().__init__()
        output_size = output_size or input_size

        self.fc1 = nn.Linear(input_size, 
                             hidden_size
                             )
        self.fc2 = nn.Linear(hidden_size, 
                             hidden_size
                             )
        self.context_proj = nn.Linear(context_size, 
                                      hidden_size, 
                                      bias=False
                                      ) if context_size else None
        self.gate_add_norm = GateAddNorm(hidden_size, 
                                         output_size, 
                                         dropout
                                         )
        self.skip = nn.Linear(input_size, 
                              output_size
                              ) if input_size != output_size else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None
        ) -> torch.Tensor:
        h = F.elu(self.fc1(x))
        if context is not None and self.context_proj is not None:
            h = h + self.context_proj(context)
        h = self.fc2(h)
        return self.gate_add_norm(h, residual=self.skip(x))


class VariableSelectionNetwork(nn.Module):
    """ VSN """

    def __init__(
        self,
        input_sizes: list[int],
        hidden_size: int,
        dropout: float = 0.1,
        context_size: int | None = None
        ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_vars = len(input_sizes)

        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(size, 
                                 hidden_size, 
                                 hidden_size, 
                                 dropout
                                 )
            for size in input_sizes
        ])

        flat_size = sum(input_sizes)
        self.softmax_grn = GatedResidualNetwork(
            flat_size, 
            hidden_size, 
            self.n_vars, 
            dropout, 
            context_size=context_size
            )

    def forward(
        self,
        var_inputs: list[torch.Tensor],
        context: torch.Tensor | None = None
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Metoda zwraca: (output, weights)
          output: (batch, time, hidden_size)
          weights: (batch, time, n_vars)
        """

        var_encodings = [grn(x) for grn, 
                         x in zip(self.var_grns, 
                                  var_inputs
                                  )
                         ]

        flat = torch.cat(var_inputs, 
                         dim=-1
                         )
        weights = torch.softmax(self.softmax_grn(flat, context), 
                                dim=-1
                                )

        stacked = torch.stack(var_encodings, 
                              dim=-2
                              )
        output = (weights.unsqueeze(-1) * stacked).sum(dim=-2)

        return output, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretowalny Multi-Head Attention z TFT.
    Uśrednia głowy zamiast konkatenacji -> zachowuje interpretowalność wag.
    """

    def __init__(self, 
                 d_model: int, 
                 n_heads: int, 
                 dropout: float = 0.0
                 ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, 
                                d_model
                                )
        self.k_proj = nn.Linear(d_model, 
                                d_model
                                )
        self.v_proj = nn.Linear(d_model, 
                                d_model // n_heads
                                )
        self.out_proj = nn.Linear(d_model // n_heads, 
                                  d_model
                                  )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None
        ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = q.shape

        Q = self.q_proj(q).view(B, 
                                T, 
                                self.n_heads, 
                                self.d_k
                                ).transpose(1, 2)
        K = self.k_proj(k).view(B, 
                                T, 
                                self.n_heads, 
                                self.d_k
                                ).transpose(1, 2)
        V = self.v_proj(v)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        avg_weights = attn_weights.mean(dim=1)

        context = torch.matmul(avg_weights, V)
        output = self.out_proj(context)

        return output, avg_weights
