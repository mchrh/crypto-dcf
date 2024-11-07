from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class SimulationParameters:
    """Parameters for Monte Carlo simulation"""
    tam: Tuple[float, float]  # (min, max) for triangular distribution
    terminal_growth: Tuple[float, float]  # (min, max)
    fee_capture_rate: Tuple[float, float]  # (min, max)
    risk_free_rate: Tuple[float, float]  # (min, max)
    market_risk_premium: Tuple[float, float]  # (min, max)
    crypto_beta: Tuple[float, float]  # (min, max)
    market_share_ceiling: Tuple[float, float]  # (min, max)