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


def dfrft_index(N: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    if N < 1:
        raise ValueError("N must be positive integer.")
    shift = 1 if N % 2 == 0 else 0
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
