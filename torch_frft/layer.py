import torch
import torch.nn as nn

from torch_frft.fracf_torch import fracF


class FrFTLayer(nn.Module):
    def __init__(self, order: float = 1.0) -> None:
        super().__init__()
        self.order = nn.Parameter(torch.tensor(order, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fracF(x, self.order)
