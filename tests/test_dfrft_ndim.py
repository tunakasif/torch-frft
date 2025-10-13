import torch

from torch_frft.dfrft_module import dfrft, dfrftmtx


def test_dfrft2D() -> None:
    N = 128
    a = 0.5
    X = torch.rand(N, N, dtype=torch.complex64)
    mtx = dfrftmtx(N, a)
    tol = 1e-6

    assert torch.allclose(dfrft(X, a, dim=0), mtx @ X, atol=tol)
    assert torch.allclose(dfrft(X, a, dim=1), X @ mtx.T, atol=tol)
