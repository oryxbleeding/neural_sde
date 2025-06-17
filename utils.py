"""Gemeinsame Hilfs‑Utilities (Seeds, MMD, KS/CvM)."""
from __future__ import annotations
import math, random
import torch, numpy as np, scipy.stats as stats
from typing import Tuple

# ---------------------------------------------------------------------------
# Reproduzierbarkeit
# ---------------------------------------------------------------------------

def set_seeds(seed: int = 2025) -> None:
    """Setzt alle relevanten Zufall‑Seeds für Reproduzierbarkeit."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

# ---------------------------------------------------------------------------
# Statistische Distanzen / Tests
# ---------------------------------------------------------------------------

def mmd_gaussian(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Linear‑Time‑Schätzer der Gaussian‑MMD (unbiased)."""
    m = x.shape[0]
    idx = torch.randperm(m)  # Shuffle für Paare
    x_shuf, y_shuf = x[idx], y[idx]
    k_xx = torch.exp(-((x - x_shuf) ** 2) / (2 * sigma ** 2)).mean()
    k_yy = torch.exp(-((y - y_shuf) ** 2) / (2 * sigma ** 2)).mean()
    k_xy = torch.exp(-((x - y) ** 2) / (2 * sigma ** 2)).mean()
    return k_xx + k_yy - 2 * k_xy


def ks_cvm(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Kolmogorov–Smirnov & Cramér–von‑Mises Teststatistiken."""
    ks = stats.ks_2samp(x, y, method="asymp").statistic
    cvm = stats.cramervonmises_2samp(x, y).statistic
    return ks, cvm

def mmd_loss(x, y, sigma=5.0):
    """Berechnet Maximum Mean Discrepancy (MMD) zwischen zwei Stichproben."""
    x = x.view(-1, 1)
    y = y.view(-1, 1)

    xx = torch.cdist(x, x, p=2) ** 2
    yy = torch.cdist(y, y, p=2) ** 2
    xy = torch.cdist(x, y, p=2) ** 2

    k_xx = torch.exp(-xx / (2 * sigma ** 2)).mean()
    k_yy = torch.exp(-yy / (2 * sigma ** 2)).mean()
    k_xy = torch.exp(-xy / (2 * sigma ** 2)).mean()

    return k_xx + k_yy - 2 * k_xy