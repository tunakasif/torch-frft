import torch


def fracF(fc: torch.Tensor, a_param: torch.Tensor) -> torch.Tensor:
    N = fc.size(0)
    if N % 2 == 1:
        raise ValueError("signal size must be even")

    fc = torch.cat([torch.zeros(N), bizinter(fc.reshape(-1)), torch.zeros(N)])

    a = a_param.detach().clone()
    one = torch.tensor(1.0)

    flag = 0

    if 0 < a < 0.5:
        flag = 1
        a -= 1

    if -0.5 < a < 0:
        flag = 2
        a += 1

    if 1.5 < a < 2:
        flag = 3
        a -= a - 1

    if -2 < a < -1.5:
        flag = 4
        a += 1

    res = fc
    if (flag == 1) or (flag == 3):
        res = corefrmod2(fc, one)

    if (flag == 2) or (flag == 4):
        res = corefrmod2(fc, -one)

    if a == 0:
        res = fc
    else:
        if (a == 2) or (a == -2):
            res = torch.flipud(fc)
        else:
            res = corefrmod2(res, a)

    res = res[N : 3 * N]
    res = bizdec(res)
    res[0] *= 2
    return res


def bizdec(x: torch.Tensor) -> torch.Tensor:
    k = torch.arange(0, x.size(0), 2)
    return x[k].reshape(-1)


def bizinter(x: torch.Tensor) -> torch.Tensor:
    if x.is_complex():
        return _bizinter_real(x.real) + 1j * _bizinter_real(x.imag)
    else:
        return _bizinter_real(x)


def _bizinter_real(x: torch.Tensor) -> torch.Tensor:
    N = x.size(0)
    N1 = N // 2 + (N % 2)
    N2 = 2 * N - (N // 2)

    upsampled = torch.stack([x, torch.zeros(N)]).T.reshape(-1)
    xf = torch.fft.fft(upsampled)
    xf[N1:N2] = 0
    return 2 * torch.real(torch.fft.ifft(xf))


def corefrmod2(signal: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    # constants
    N = signal.size(0)
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
    multip = torch.mul(signal, chirp)

    # Chirp Convolution
    t = torch.arange(-N + 1, N) / deltax
    hlptc = torch.exp(beta * t**2)

    N2 = hlptc.size(0)
    next_power_two = 2 ** torch.ceil(torch.log2(torch.tensor(N2 + N - 1))).int()
    Hc = torch.fft.ifft(
        torch.mul(
            torch.fft.fft(hlptc, n=next_power_two),
            torch.fft.fft(multip, n=next_power_two),
        )
    )
    Hc = Hc[N - 1 : 2 * N - 1]

    # Chirp Multiplication
    result = torch.mul(Aphi * chirp, Hc) / deltax

    # Adjustment
    if N % 2 == 1:
        return torch.roll(result, -1)
    else:
        return result
