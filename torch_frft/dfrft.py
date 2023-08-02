from math import ceil

import torch


def dFRT(
    N: int,
    a: float,
    *,
    approx_order: int = 2,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor | None:
    if N < 1 or approx_order < 2:
        raise ValueError("Necessary conditions for integers: N > 1 and approx_order >= 2.")
    return None


def dis_s(
    N: int,
    *,
    approx_order: int = 2,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    if N < 1 or approx_order < 2:
        raise ValueError("Necessary conditions for integers: N > 1 and approx_order >= 2.")
    S = creates(N, approx_order=approx_order, device=device)
    P = createp(N, device=device)

    CS = torch.einsum("ij,jk,lk->il", P, S, P)
    C2 = CS[: N // 2 + 1, : N // 2 + 1]
    S2 = CS[N // 2 + 1 :, N // 2 + 1 :]

    ec, VC = torch.linalg.eigh(C2)  # ascending order
    es, VS = torch.linalg.eigh(S2)  # ascending order

    N0, N1 = ceil(N / 2 - 1), N // 2 + 1
    qvc = torch.cat((VC, torch.zeros((N0, N1), device=device)))
    qvs = torch.cat((torch.zeros((N1, N0), device=device), VS))

    SC2 = torch.matmul(P, qvc).flip(-1)  # descending order
    SS2 = torch.matmul(P, qvs).flip(-1)  # descending order

    if N % 2 == 0:
        evec = torch.zeros(N, N + 1)
        SS2_new = torch.hstack(
            (SS2, torch.zeros((SS2.size(0), 1), dtype=SS2.dtype, device=SS2.device))
        )
        evec[:, : N + 1 : 2] = SC2
        evec[:, 1:N:2] = SS2_new
        evec = torch.hstack((evec[:, : N - 1], evec[:, -1].unsqueeze(-1)))
    else:
        evec = torch.zeros(N, N)
        evec[:, : N + 1 : 2] = SC2
        evec[:, 1:N:2] = SS2
    return evec


def dfrft_index(N: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
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


def cconvm(vector: torch.Tensor) -> torch.Tensor:
    """
    Generate a circulant matrix based on the input vector.

    Parameters:
        vector (torch.Tensor): 1-dimensional PyTorch tensor representing
        the first row of the circulant matrix.

    Returns:
        torch.Tensor: The resulting circulant matrix.
    """
    vector = vector.flatten()
    size = vector.size(-1)
    idx = torch.arange(size, device=vector.device)
    indices = torch.remainder(idx[:, None] - idx, size)
    return vector[indices]


def conv(input: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Perform full 1-dimensional convolution using PyTorch.

    Parameters:
        input (torch.Tensor): Input 1-dimensional tensor.
        kernel (torch.Tensor): Convolution kernel (also 1-dimensional).

    Returns:
        torch.Tensor: Resulting 1-dimensional convolution with full padding.
    """
    padding_size = kernel.size(0) - 1
    padded_input = torch.nn.functional.pad(
        input, (padding_size, padding_size), mode="constant", value=0
    )
    conv_output = torch.conv1d(padded_input.view(1, 1, -1), kernel.view(1, 1, -1).flip(-1))
    return conv_output.reshape(-1)


def creates(
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
        dum = conv(dum, dum0)

    return cconvm(s) + torch.diag(torch.real(torch.fft.fft(s)))


def createp(N: int, *, device: torch.device = torch.device("cpu")) -> torch.Tensor:
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
