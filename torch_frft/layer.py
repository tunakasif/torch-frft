import torch
import torch.nn as nn

from torch_frft import dfrft, frft


class FrFTLayer(nn.Module):
    def __init__(self, order: float = 1.0, *, dim: int = -1) -> None:
        super().__init__()
        self.order = nn.Parameter(torch.tensor(order, dtype=torch.float32))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return frft(x, self.order, dim=self.dim)


class DFrFTLayer(nn.Module):
    def __init__(self, order: float = 1.0, *, dim: int = -1) -> None:
        super().__init__()
        self.order = nn.Parameter(torch.tensor(order, dtype=torch.float32))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return dfrft(x, self.order.item(), dim=self.dim)
