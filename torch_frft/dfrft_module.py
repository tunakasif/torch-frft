from math import ceil

import torch


def idfrft(x: torch.Tensor, a: float | torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    return dfrft(x, -a, dim=dim)


def dfrft(x: torch.Tensor, a: float | torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    dfrft_matrix = dfrftmtx(x.size(dim), a, device=x.device)
    dtype = torch.promote_types(dfrft_matrix.dtype, x.dtype)
    return torch.einsum(
        _get_dfrft_einsum_str(len(x.shape), dim),
        dfrft_matrix.type(dtype),
        x.type(dtype),
    )


def _get_dfrft_einsum_str(dim_count: int, req_dim: int) -> str:
    if req_dim < -dim_count or req_dim >= dim_count:
        raise ValueError("Dimension size error.")
    dim = torch.remainder(req_dim, torch.tensor(dim_count))
    diff = dim_count - dim
    remaining_str = "".join([chr(num) for num in range(98, 98 + diff)])
    return f"ab,...{remaining_str}->...{remaining_str.replace('b', 'a', 1)}"


def idfrftmtx(
    N: int,
    a: float | torch.Tensor,
    *,
    approx_order: int = 2,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    return dfrftmtx(N, -a, approx_order=approx_order, device=device)


def dfrftmtx(
    N: int,
    a: float | torch.Tensor,
    *,
    approx_order: int = 2,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    if N < 1 or approx_order < 2:
        raise ValueError("Necessary conditions for integers: N > 1 and approx_order >= 2.")
    evecs = _get_dfrft_evecs(N, approx_order=approx_order, device=device).type(torch.complex64)
    idx = _dfrft_index(N, device=device)
    evals = torch.exp(-1j * a * (torch.pi / 2) * idx).type(torch.complex64)
    dfrft_matrix = torch.einsum("ij,j,kj->ik", evecs, evals, evecs)
    return dfrft_matrix


def _get_dfrft_evecs(
    N: int,
    *,
    approx_order: int = 2,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    if N < 1 or approx_order < 2:
        raise ValueError("Necessary conditions for integers: N > 1 and approx_order >= 2.")
    S = _create_hamiltonian(N, approx_order=approx_order, device=device)
    P = _create_odd_even_decomp_matrix(N, device=device)

    CS = torch.einsum("ij,jk,lk->il", P, S, P)
    C2 = CS[: N // 2 + 1, : N // 2 + 1]
    S2 = CS[N // 2 + 1 :, N // 2 + 1 :]

    _, VC = torch.linalg.eigh(C2)  # ascending order
    _, VS = torch.linalg.eigh(S2)  # ascending order

    N0, N1 = ceil(N / 2 - 1), N // 2 + 1
    qvc = torch.cat((VC, torch.zeros((N0, N1), device=device)))
    qvs = torch.cat((torch.zeros((N1, N0), device=device), VS))

    SC2 = torch.matmul(P, qvc).flip(-1)  # descending order
    SS2 = torch.matmul(P, qvs).flip(-1)  # descending order

    if N % 2 == 0:
        evecs = torch.zeros(N, N + 1, device=device)
        SS2_new = torch.hstack(
            (SS2, torch.zeros((SS2.size(0), 1), dtype=SS2.dtype, device=SS2.device))
        )
        evecs[:, : N + 1 : 2] = SC2
        evecs[:, 1:N:2] = SS2_new
        evecs = torch.hstack((evecs[:, : N - 1], evecs[:, -1].unsqueeze(-1)))
    else:
        evecs = torch.zeros(N, N, device=device)
        evecs[:, : N + 1 : 2] = SC2
        evecs[:, 1:N:2] = SS2
    return evecs


def _dfrft_index(N: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    if N < 1:
        raise ValueError("N must be positive integer.")
    shift = 1 - (N % 2)  # 1 if N is even, 0 if N is odd
    last_entry = torch.tensor(N - 1 + shift, device=device)
    return torch.cat(
        (
            torch.arange(0, N - 1, dtype=torch.float32, device=device),
            last_entry.unsqueeze(-1),
        )
    )


def _circulant(vector: torch.Tensor) -> torch.Tensor:
    """
    Generate a circulant matrix based on the input vector.

    Parameters:
        vector (torch.Tensor): 1-dimensional PyTorch tensor representing
        the first row of the circulant matrix.

    Returns:
        torch.Tensor: The resulting circulant matrix.

    Example:
        >>> circulant(torch.tensor([1, 2, 3]))
        tensor([[1, 3, 2],
                [2, 1, 3],
                [3, 2, 1]])
    """
    vector = vector.flatten()
    size = vector.size(-1)
    idx = torch.arange(size, device=vector.device)
    indices = torch.remainder(idx[:, None] - idx, size)
    return vector[indices]


def _conv1d_full(vector: torch.Tensor, kernel1d: torch.Tensor) -> torch.Tensor:
    """
    Perform full 1-dimensional convolution on 1-dimensional input tensor and kernel.

    Parameters:
        input (torch.Tensor): Input 1-dimensional tensor.
        kernel (torch.Tensor): Convolution kernel (also 1-dimensional).

    Returns:
        torch.Tensor: Resulting 1-dimensional convolution with full padding.

    Example:
        >>> conv1d_full(torch.tensor([1, 2, 3, 4]), torch.tensor([1, -1, 2]))
        tensor([1, 1, 3, 5, 2, 8])
    """
    padding_size = kernel1d.size(0) - 1
    padded_input = torch.nn.functional.pad(
        vector, (padding_size, padding_size), mode="constant", value=0
    )
    conv_output = torch.conv1d(padded_input.view(1, 1, -1), kernel1d.view(1, 1, -1).flip(-1))
    return conv_output.reshape(-1)


def _create_hamiltonian(
    N: int,
    *,
    approx_order: int = 2,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    if N < 1 or approx_order < 2:
        raise ValueError("Necessary conditions for integers: N > 1 and approx_order >= 2.")

    order = approx_order // 2
    dum0 = torch.tensor([1.0, -2.0, 1.0], device=device)
    dum = dum0.clone()
    s = torch.zeros(1, device=device)

    for k in range(1, order + 1):
        coefficient = (
            2
            * (-1) ** (k - 1)
            * torch.prod(torch.arange(1, k, device=device)) ** 2
            / torch.prod(torch.arange(1, 2 * k + 1, device=device))
        )
        s = (
            coefficient
            * torch.cat(
                (
                    torch.zeros(1, device=device),
                    dum[k + 1 : 2 * k + 1],
                    torch.zeros(N - 1 - 2 * k, device=device),
                    dum[:k],
                )
            )
            + s
        )
        dum = _conv1d_full(dum, dum0)

    return _circulant(s) + torch.diag(torch.real(torch.fft.fft(s)))


def _create_odd_even_decomp_matrix(
    N: int, *, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    if N < 1:
        raise ValueError("N must be positive integer.")

    x1 = torch.ones(1 + N // 2, dtype=torch.float32, device=device)
    x2 = -torch.ones(N - N // 2 - 1, dtype=torch.float32, device=device)
    diagonal = torch.diag(torch.cat((x1, x2)))
    anti = torch.diag(torch.ones(N - 1, device=device), -1).rot90()
    P = (diagonal + anti) / torch.sqrt(torch.tensor(2.0))

    P[0, 0] = 1
    if N % 2 == 0:
        P[N // 2, N // 2] = 1
    return P
