import torch
import torch.nn as nn

from torch_frft.dfrft_module import _dfrft_index, _get_dfrft_einsum_str, _get_dfrft_evecs, dfrft
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


class DFrFTLayerFixEvecs(nn.Module):
    def __init__(self, order: float = 1.0, *, dim: int = -1, trainable: bool = True) -> None:
        super().__init__()
        self.order = nn.Parameter(
            torch.tensor(order, dtype=torch.float32),
            requires_grad=trainable,
        )
        self.dim = dim
        self.dtype: torch.dtype | None = None
        self.size: int | None = None
        self.evecs: torch.Tensor | None = None

    def __repr__(self) -> str:
        return f"DFrFTLayerFixEvecs(order={self.order.item()}, dim={self.dim})"

    def _get_dfrft_matrix_from_fixed_evecs(self, x: torch.Tensor) -> torch.Tensor:
        if self.size is None:
            self.size = x.size(self.dim)
        if self.evecs is None:
            self.evecs = _get_dfrft_evecs(self.size, device=x.device).type(torch.complex64)

        idx = _dfrft_index(self.size, device=x.device)
        evals = torch.exp(-1j * self.order * (torch.pi / 2) * idx).type(torch.complex64)
        return torch.einsum("ij,j,kj->ik", self.evecs, evals, self.evecs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dfrft_matrix = self._get_dfrft_matrix_from_fixed_evecs(x)
        if self.dtype is None:
            self.dtype = torch.promote_types(dfrft_matrix.dtype, x.dtype)

        return torch.einsum(
            _get_dfrft_einsum_str(len(x.shape), self.dim),
            dfrft_matrix.type(self.dtype),
            x.type(self.dtype),
        )
