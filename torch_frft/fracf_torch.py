import torch
from torch.fft import fft, fftshift, ifft


def fracF(fc: torch.Tensor, a_param: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    N = fc.size(dim)
    if N % 2 == 1:
        raise ValueError("signal size must be even")

    # 4-modulation and shifting to [-2, 2] interval
    a = torch.fmod(a_param, 4)
    if a > 2:
        a -= 4
    elif a < -2:
        a += 4

    # special integer cases
    if a == 0.0:
        return fc
    elif a == 2.0 or a == -2.0:
        return dflip(fc, dim=dim)
    elif a == 1.0:
        return fftshift(fft(fftshift(fc), dim=dim, norm="ortho"))
    elif a == -1.0:
        return fftshift(ifft(fftshift(fc), dim=dim, norm="ortho"))

    biz = bizinter(fc, dim=dim)
    zeros = torch.zeros_like(biz).index_select(dim, torch.arange(0, N))
    fc = torch.cat([zeros, biz, zeros], dim=dim)

    res = fc
    if (0 < a < 0.5) or (1.5 < a < 2):
        res = corefrmod2(fc, torch.tensor(1.0), dim=dim)
        a -= 1

    if (-0.5 < a < 0) or (-2 < a < -1.5):
        res = corefrmod2(fc, torch.tensor(-1.0), dim=dim)
        a += 1

    res = corefrmod2(res, a, dim=dim)
    res = torch.index_select(res, dim=dim, index=torch.arange(N, 3 * N))
    res = bizdec(res, dim=dim)

    # Double the first entry of the vector in the given dimension,
    # res[0] *= 2 in n-dimensional case, i.e., Hadamard product with
    # [2, 1, 1, ..., 1] along n-th axis.
    first_entry_doubler_vec = torch.ones(res.size(dim))
    first_entry_doubler_vec[0] = 2
    res = vecmul_ndim(res, first_entry_doubler_vec, dim=dim)
    return res


def dflip(tensor: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    first, remaining = torch.tensor_split(tensor, (1,), dim=dim)
    return torch.concat((first, remaining.flip(dim)), dim=dim)


def bizdec(x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    k = torch.arange(0, x.size(dim), 2)
    return x.index_select(dim, k)


def bizinter(x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    if x.is_complex():
        return _bizinter_real(x.real, dim=dim) + 1j * _bizinter_real(x.imag, dim=dim)
    else:
        return _bizinter_real(x, dim=dim)


def _bizinter_real(x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    N = x.size(dim)
    N1 = N // 2 + (N % 2)
    N2 = 2 * N - (N // 2)

    upsampled = upsample2(x, dim=dim)
    xf = fft(upsampled, dim=dim)
    xf = torch.index_fill(xf, dim, torch.arange(N1, N2), 0)
    return 2 * torch.real(ifft(xf, dim=dim))


def upsample2(x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    upsampled = x.repeat_interleave(2, dim=dim)
    idx = torch.arange(1, upsampled.size(dim), 2)
    return torch.index_fill(upsampled, dim, idx, 0)


def corefrmod2(signal: torch.Tensor, a: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    # constants
    N = signal.size(dim)
    Nend = N // 2
    Nstart = -(N % 2 + Nend)
    deltax = torch.sqrt(torch.tensor(N))

    phi = a * torch.pi / 2
    alpha = -1j * torch.pi * torch.tan(phi / 2)
    beta = 1j * torch.pi / torch.sin(phi)

    Aphi_num = torch.exp(-1j * (torch.pi * torch.sign(torch.sin(phi)) / 4 - phi / 2))
    Aphi_denum = torch.sqrt(torch.abs(torch.sin(phi)))
    Aphi = Aphi_num / Aphi_denum

    # Chirp Multiplication
    x = torch.arange(Nstart, Nend) / deltax
    chirp = torch.exp(alpha * x**2)
    multip = vecmul_ndim(signal, chirp, dim=dim)

    # Chirp Convolution
    t = torch.arange(-N + 1, N) / deltax
    hlptc = torch.exp(beta * t**2)

    N2 = hlptc.size(0)
    next_power_two = 2 ** torch.ceil(torch.log2(torch.tensor(N2 + N - 1))).int()
    Hc = ifft(
        vecmul_ndim(
            fft(multip, n=next_power_two, dim=dim),
            fft(hlptc, n=next_power_two),
            dim=dim,
        ),
        dim=dim,
    )
    Hc = torch.index_select(Hc, dim, torch.arange(N - 1, 2 * N - 1))

    # Chirp Multiplication
    result = vecmul_ndim(Hc, Aphi * chirp, dim=dim) / deltax

    # Adjustment
    if N % 2 == 1:
        return torch.roll(result, -1, dims=(dim,))
    else:
        return result


def vecmul_ndim(
    tensor: torch.Tensor,
    vector: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """
    Multiply two tensors (`torch.mul()`) along a given dimension.
    """
    return torch.einsum(_get_mul_dim_einstr(len(tensor.shape), dim), tensor, vector)


def _get_mul_dim_einstr(dim_count: int, req_dim: int) -> str:
    if req_dim < -dim_count or req_dim >= dim_count:
        raise ValueError("Dimension size error.")
    dim = torch.remainder(req_dim, torch.tensor(dim_count))
    diff = dim_count - dim
    remaining_str = "".join([chr(num) for num in range(97, 97 + diff)])
    return f"...{remaining_str},a->...{remaining_str}"
