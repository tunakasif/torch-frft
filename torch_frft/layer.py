import torch
import torch.nn as nn

from torch_frft.frft import frft


class FrFTLayer(nn.Module):
    def __init__(self, order: float = 1.0) -> None:
        super().__init__()
        self.order = nn.Parameter(torch.tensor(order, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return frft(x, self.order)
