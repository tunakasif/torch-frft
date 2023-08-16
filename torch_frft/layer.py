import torch
import torch.nn as nn

from torch_frft.dfrft_module import dfrft
from torch_frft.frft_module import frft


class FrFTLayer(nn.Module):
    def __init__(self, order: float = 1.0, *, dim: int = -1, trainable: bool = True) -> None:
        super().__init__()
        self.order = nn.Parameter(
            torch.tensor(order, dtype=torch.float32),
            requires_grad=trainable,
        )
        self.dim = dim

    def __repr__(self) -> str:
        return f"FrFTLayer(order={self.order.item()}, dim={self.dim})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return frft(x, self.order, dim=self.dim)


class DFrFTLayer(nn.Module):
    def __init__(self, order: float = 1.0, *, dim: int = -1, trainable: bool = True) -> None:
        super().__init__()
        self.order = nn.Parameter(
            torch.tensor(order, dtype=torch.float32),
            requires_grad=trainable,
        )
        self.dim = dim

    def __repr__(self) -> str:
        return f"DFrFTLayer(order={self.order.item()}, dim={self.dim})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return dfrft(x, self.order, dim=self.dim)
