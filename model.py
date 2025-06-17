# model.py

import torch
import torch.nn as nn
import torchsde
from torch import Tensor

class Clamp(nn.Module):
    def __init__(self, min=0.0, max=1.0):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.clamp(input, min=self.min, max=self.max)

class DriftNet(nn.Sequential):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

class DiffNet(nn.Sequential):
    def __init__(self, input_size: int, hidden_size: int, sigma_max: float = 1.0):
        super().__init__(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            Clamp(0.0, sigma_max)
        )

class MortalitySDE(nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        input_dim = 2  # Zeit + Zustand
        self.drift = DriftNet(input_dim, hidden_size)
        self.diffusion = DiffNet(input_dim, hidden_size)

    def _concat(self, t: Tensor, y: Tensor) -> Tensor:
        if t.dim() == 0:
            t = t.expand(y.size(0), 1)
        elif t.dim() == 1:
            t = t[:, None]
        return torch.cat([t, y], dim=1)

    def f(self, t: Tensor, y: Tensor) -> Tensor:
        return self.drift(self._concat(t, y))

    def g(self, t: Tensor, y: Tensor) -> Tensor:
        return self.diffusion(self._concat(t, y))
