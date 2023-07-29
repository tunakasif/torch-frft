import torch
import torch.nn as nn

from trainable_frft.fracf_torch import fracF_05_15


class FrFTLayer(nn.Module):
    def __init__(self, order: float = 1.0) -> None:
        super().__init__()
        self.order = nn.Parameter(torch.tensor(order, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fracF_05_15(x, self.order)
