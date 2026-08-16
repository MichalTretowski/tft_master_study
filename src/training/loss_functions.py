"""
Loss Functions: Differentiable Sharpe Ratio (DSR) i funkcje pomocnicze.

DSR (Differentiable Sharpe Ratio):
  Pozwala bezpośrednio optymalizować Sharpe Ratio przez backpropagation.
  Model wyjściowy to pozycja tradingowa ∈ [-1, 1] (tanh).

Referencja:
  Sharpe (1994), Moody et al. (1998) "Performance Functions and
  Reinforcement Learning for Trading Systems and Portfolios"
  jest to wersja za paywallem. 

  Darmowa dostępna wersja "konferencyjna" tego journala jest dostępna
  pod tym linkiem: Reinforcement Learning for Trading Systems and Portfolios 
  (https://cdn.aaai.org/KDD/1998/KDD98-049.pdf) - Moody, Wu, Liao, Saffell.

Koszt transakcji:
  Bez kosztów model uczy się nadmiernie tradować.
  Binance spot taker fee: ~0.1% (0.001).
  Aby zmienić pozycję z -1 na +1 = koszt 2 × fee.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DifferentiableSharpeRatio(nn.Module):
    """
    DSR Loss: minimalizuj ujemny Sharpe Ratio.

    Parametry:
        annualization_factor: sqrt(liczba_kroków_na_rok), 
            dla 4h: sqrt(6 × 365) = sqrt(2190) ≈ 46.8
        transaction_cost: koszt zmiany pozycji
        epsilon: stabilizator mianownika (zapobiegający dzieleniu przez 0)
    """

    ANNUALIZATION = {
        "1h":  (8760 ** 0.5),
        "4h":  (2190 ** 0.5),
        "1d":  (365 ** 0.5)
        }

    def __init__(
        self,
        annualization_factor: float | None = None,
        timeframe: str = "4h",
        transaction_cost: float = 0.001,
        epsilon: float = 1e-8
        ) -> None:
        super().__init__()
        self.timeframe = timeframe
        self.ann_factor = annualization_factor or self.ANNUALIZATION[timeframe]
        self.transaction_cost = transaction_cost
        self.epsilon = epsilon

    def forward(
        self,
        positions: torch.Tensor,
        price_returns: torch.Tensor
        ) -> torch.Tensor:
        """
        Metoda oblicza ujemny Sharpe Ratio.
        """
        positions = positions.squeeze(-1) if positions.dim() > 1 else positions
        price_returns = price_returns.squeeze(-1) if price_returns.dim() > 1 else price_returns

        return self._sharpe(
            self.portfolio_returns(positions, price_returns)
            )

    def portfolio_returns(
        self,
        positions: torch.Tensor,
        price_returns: torch.Tensor
        ) -> torch.Tensor:
        """Zwroty portfela: pozycja i zwrot z tego samego momentu."""
        turnover = torch.cat([
            positions[:1].abs(),          # wejscie z pozycji zerowej
            (positions[1:] - positions[:-1]).abs()
            ])

        return (
            positions * price_returns
            - turnover * self.transaction_cost
            )

    def _sharpe(self, 
                returns: torch.Tensor
                ) -> torch.Tensor:
        """Metoda oblicza ujemny annualizowamy Sharpe z batcha zwrotów"""
        mean_r = returns.mean()
        std_r = returns.std() + self.epsilon
        sharpe = mean_r / std_r * self.ann_factor
        return -sharpe  # minimalizujemy -> maksymalizujemy Sharpe



def compute_sharpe_ratio(
    returns: torch.Tensor | list[float],
    timeframe: str = "4h",
    risk_free_rate: float = 0.0
    ) -> float:
    """
    Metoda potrzeba do ewaluacji
    Oblicza Sharpe Ratio z serii zwrotów 
    """
    import numpy as np

    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()
    else:
        returns = np.array(returns)

    ann_factor = DifferentiableSharpeRatio.ANNUALIZATION[timeframe]
    excess = returns - risk_free_rate / ann_factor ** 2

    if excess.std() < 1e-10:
        return 0.0

    return float((excess.mean() / excess.std()) * ann_factor)
