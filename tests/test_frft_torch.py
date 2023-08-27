import pytest
import torch

from torch_frft.frft_module import (
    _bizdec,
    _bizinter,
    _corefrmod2,
    _dflip,
    frft,
    frft_shifted,
    ifrft,
)


def test_dflip_1d() -> None:
    N = 1000
    torch.manual_seed(0)
    x = torch.rand(N)
    assert torch.allclose(_dflip(x), torch.concat((x[:1], x[1:].flip(0)), dim=0))
    assert torch.allclose(_dflip(_dflip(x)), x)


def test_bizdec() -> None:
    x1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x2 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y1 = _bizdec(x1)
    y2 = _bizdec(x2)
    assert torch.allclose(y1, torch.tensor([1.0, 3.0]))
    assert torch.allclose(y2, torch.tensor([1.0, 3.0, 5.0]))


def test_bizinter() -> None:
    x10 = torch.arange(0, 10)
    y10_expected = torch.tensor(
        [
            0,
            -0.813751514675042,
            1,
            2.14885899083011,
            2,
            2.14885899083011,
            3,
            3.65838444032454,
            4,
            4.5,
            5,
            5.34161555967546,
            6,
            6.85114100916989,
            7,
            6.85114100916989,
            8,
            9.81375151467504,
            9,
            4.5,
        ]
    )
    assert torch.allclose(_bizinter(x10), y10_expected)


def test_corefrmod2() -> None:
    x = torch.arange(0, 10)
    a10_expected = torch.tensor(
        [
            1.581138830084188 - 1j * 0.000000000000033,
            -1.581138830084181 - 1j * 0.513743148373015,
            1.581138830084184 + 1j * 1.148764602736809,
            -1.581138830084191 - 1j * 2.176250899482820,
            1.581138830084188 + 1j * 4.866244947338653,
            14.230249470757705 + 1j * 0.000000000000009,
            1.581138830084194 - 1j * 4.866244947338653,
            -1.581138830084195 + 1j * 2.176250899482820,
            1.581138830084190 - 1j * 1.148764602736801,
            -1.581138830084190 + 1j * 0.513743148373006,
        ]
    )
    a05_expected = torch.tensor(
        [
            4.300425665961087 - 1j * 7.863204974688656,
            4.027288320118928 - 1j * 3.346581752950329,
            2.806850172683342 - 1j * 1.070282213519745,
            2.942209152373217 - 1j * 0.129308758397126,
            3.103955191410604 + 1j * 1.658974800296961,
            4.773860317626514 + 1j * 0.538277189010558,
            9.234846570973911 + 1j * 1.870142992999104,
            3.501865444145145 - 1j * 8.541826363887390,
            -4.711778345708308 - 1j * 2.680135885340919,
            -1.572836646294373 + 1j * 2.216050384301396,
        ]
    )

    assert torch.allclose(_corefrmod2(x, torch.tensor(1.0)), a10_expected)
    assert torch.allclose(_corefrmod2(x, torch.tensor(0.5)), a05_expected)


def test_frft_arange() -> None:
    x = torch.arange(0, 10)
    a03_expected = torch.tensor(
        [
            0.157038300373207 - 1j * 1.233161184370146,
            -0.120837791229858 - 1j * 0.617715275835068,
            0.454703029732211 - 1j * 1.328337613535109,
            2.995608064781706 - 1j * 1.355960059533741,
            3.566160367663963 + 1j * 0.047349308848695,
            5.356102246627707 + 1j * 1.711068519765365,
            5.998962164301052 - 1j * 0.143503625126124,
            8.564173364433401 - 1j * 2.731368789218994,
            0.625481536671613 - 1j * 8.578203656279815,
            -3.931322814222189 - 1j * 3.298947780646836,
        ]
    )
    a05_expected = torch.tensor(
        [
            -1.061155531706124 + 1j * 0.664730131557422,
            -0.521494223725904 + 1j * 0.050473107354931,
            -0.312636785195539 - 1j * 0.629218673157758,
            1.525273067968619 - 1j * 1.877365678003593,
            4.296159585058391 + 1j * 0.980441925410502,
            4.546230329926177 + 1j * 1.090949230959734,
            9.332162716373247 + 1j * 2.024891825636435,
            3.809160143713090 - 1j * 9.011645324382824,
            -5.151162613070951 - 1j * 3.035430499225090,
            -2.288952795093801 + 1j * 2.329893310043556,
        ]
    )
    a07_expected = torch.tensor(
        [
            0.468337222624645 - 1j * 0.162938998659897,
            -0.088048737691936 + 1j * 0.359006461320178,
            -0.025795199784097 - 1j * 0.649890622444920,
            -0.047776563235135 - 1j * 0.245422698682353,
            3.737952098102939 - 1j * 1.790823124528120,
            7.007441403407102 + 1j * 6.127640862768134,
            9.698477613862426 - 1j * 4.505676586678055,
            -3.002547217099880 - 1j * 5.494625252614294,
            -2.591569353154538 + 1j * 2.213383418979645,
            1.908451818548160 + 1j * 1.315287479563901,
        ]
    )
    a10_expected = torch.tensor(
        [
            1.581138830084168 - 1j * 0.000000000000031,
            -1.581138830084188 - 1j * 0.513743148373010,
            1.581138830084184 + 1j * 1.148764602736811,
            -1.581138830084191 - 1j * 2.176250899482818,
            1.581138830084191 + 1j * 4.866244947338655,
            14.230249470757711 + 1j * 0.000000000000005,
            1.581138830084194 - 1j * 4.866244947338653,
            -1.581138830084194 + 1j * 2.176250899482822,
            1.581138830084192 - 1j * 1.148764602736803,
            -1.581138830084189 + 1j * 0.513743148373007,
        ]
    )
    a25_expected = torch.tensor(
        [
            0.468139593169412 + 1j * 4.463099456027352,
            -2.288952795093806 + 1j * 2.329893310043559,
            -5.151162613070946 - 1j * 3.035430499225087,
            3.809160143713099 - 1j * 9.011645324382821,
            9.332162716373237 + 1j * 2.024891825636439,
            4.546230329926177 + 1j * 1.090949230959732,
            4.296159585058382 + 1j * 0.980441925410500,
            1.525273067968621 - 1j * 1.877365678003587,
            -0.312636785195532 - 1j * 0.629218673157754,
            -0.521494223725902 + 1j * 0.050473107354936,
        ]
    )

    a_neg25_expected = torch.tensor(
        [
            0.468139593169411 - 1j * 4.463099456027352,
            -2.288952795093804 - 1j * 2.329893310043560,
            -5.151162613070946 + 1j * 3.035430499225087,
            3.809160143713098 + 1j * 9.011645324382821,
            9.332162716373238 - 1j * 2.024891825636438,
            4.546230329926177 - 1j * 1.090949230959732,
            4.296159585058382 - 1j * 0.980441925410501,
            1.525273067968621 + 1j * 1.877365678003589,
            -0.312636785195532 + 1j * 0.629218673157753,
            -0.521494223725901 - 1j * 0.050473107354935,
        ]
    )

    tol = 1e-4
    assert torch.allclose(frft(x, 0.3), a03_expected, atol=tol)
    assert torch.allclose(frft(x, 0.5), a05_expected, atol=tol)
    assert torch.allclose(frft(x, 0.7), a07_expected, atol=tol)
    assert torch.allclose(frft(x, 1.0), a10_expected, atol=tol)
    assert torch.allclose(frft(x, 2.5), a25_expected, atol=tol)
    assert torch.allclose(frft(x, -2.5), a_neg25_expected, atol=tol)


def test_frft_integer() -> None:
    from torch.fft import fft, fftshift, ifft

    tol = 1e-3
    N = 1000
    torch.manual_seed(0)
    x = torch.rand(N)
    sqrtN = torch.sqrt(torch.tensor(N))

    assert torch.allclose(frft(x, torch.tensor(0.0)), x, atol=tol)
    assert torch.allclose(
        frft(x, 1.0),
        fftshift(fft(fftshift(x))) / sqrtN,
        atol=tol,
    )
    assert torch.allclose(
        frft(x, -1.0),
        fftshift(ifft(fftshift(x))) * sqrtN,
        atol=tol,
    )
    assert torch.allclose(
        frft(x, 2.0).to(torch.complex64),
        fftshift(fft(fft(fftshift(x)))) / torch.tensor(N),
        atol=tol,
    )
    assert torch.allclose(
        frft(x, -2.0).to(torch.complex64),
        fftshift(ifft(ifft(fftshift(x)))) * torch.tensor(N),
        atol=tol,
    )


def test_frft_shifted() -> None:
    from torch.fft import fft, ifft

    tol = 1e-3
    N = 1000
    torch.manual_seed(0)
    x = torch.rand(N)

    assert torch.allclose(frft(x, torch.tensor(0.0)), x, atol=tol)
    assert torch.allclose(
        frft_shifted(x, 1.0),
        fft(x, norm="ortho"),
        atol=tol,
    )
    assert torch.allclose(
        frft_shifted(x, -1.0),
        ifft(x, norm="ortho"),
        atol=tol,
    )
    assert torch.allclose(
        frft(x, 2.0).to(torch.complex64),
        fft(fft(x, norm="ortho"), norm="ortho"),
        atol=tol,
    )
    assert torch.allclose(
        frft(x, -2.0).to(torch.complex64),
        ifft(ifft(x, norm="ortho"), norm="ortho"),
        atol=tol,
    )


def test_ifrft() -> None:
    tol = 1e-5
    N = 1000
    torch.manual_seed(0)
    x = torch.rand(N)

    a_values = torch.rand(100)
    for a in a_values:
        assert torch.allclose(
            frft(x, -a),
            ifrft(x, a),
            atol=tol,
        )


def test_odd_size_error() -> None:
    X = torch.ones(5)
    a = torch.rand(1)
    with pytest.raises(ValueError):
        frft(X, a)
