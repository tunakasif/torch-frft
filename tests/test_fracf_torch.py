import torch

from trainable_frft.fracf_torch import bizdec, bizinter, corefrmod2


def test_bizdec() -> None:
    x1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x2 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y1 = bizdec(x1)
    y2 = bizdec(x2)
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
    assert torch.allclose(bizinter(x10), y10_expected)


def test_corefrmod2_a1() -> None:
    x = torch.arange(0, 10)
    expected = torch.tensor(
        [
            1.58113883008419 - 1j * 3.29390504690326e-14,
            -1.58113883008418 - 1j * 0.513743148373015,
            1.58113883008418 + 1j * 1.14876460273681,
            -1.58113883008419 - 1j * 2.17625089948282,
            1.58113883008419 + 1j * 4.86624494733865,
            14.2302494707577 + 1j * 8.70686700207022e-15,
            1.58113883008419 - 1j * 4.86624494733865,
            -1.58113883008419 + 1j * 2.17625089948282,
            1.58113883008419 - 1j * 1.1487646027368,
            -1.58113883008419 + 1j * 0.513743148373006,
        ]
    )

    assert torch.allclose(corefrmod2(x, torch.tensor(1.0)), expected)
