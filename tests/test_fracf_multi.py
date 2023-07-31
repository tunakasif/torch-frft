import pytest
import torch

from trainable_frft.fracf_torch import _get_mul_dim_einstr, bizdec, bizinter, corefrmod2, upsample2

X = torch.tensor(
    [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
    ],
    dtype=torch.float32,
)


def test_bizdec_multi() -> None:
    global X
    expected_dim0 = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [11, 12, 13, 14, 15],
        ],
        dtype=torch.float32,
    )
    expected_dim1 = torch.tensor(
        [
            [1, 3, 5],
            [6, 8, 10],
            [11, 13, 15],
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(bizdec(X, dim=0), expected_dim0)
    assert torch.allclose(bizdec(X, dim=1), expected_dim1)


def test_upsample2_multi() -> None:
    global X
    expected_dim0 = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [0, 0, 0, 0, 0],
            [6, 7, 8, 9, 10],
            [0, 0, 0, 0, 0],
            [11, 12, 13, 14, 15],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    expected_dim1 = torch.tensor(
        [
            [1, 0, 2, 0, 3, 0, 4, 0, 5, 0],
            [6, 0, 7, 0, 8, 0, 9, 0, 10, 0],
            [11, 0, 12, 0, 13, 0, 14, 0, 15, 0],
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(upsample2(X, dim=0), expected_dim0)
    assert torch.allclose(upsample2(X, dim=1), expected_dim1)


def test_bizinter_multi() -> None:
    global X
    expected_dim0 = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
        ],
        dtype=torch.float32,
    )
    expected_dim1 = torch.tensor(
        [
            [1.0, 0.76393202250021, 2.0, 3.0, 3.0, 3.0, 4.0, 5.236067977499789, 5.0, 3.0],
            [6.0, 5.763932022500211, 7.0, 8.0, 8.0, 8.0, 9.0, 10.23606797749979, 10.0, 8.0],
            [11.0, 10.76393202250021, 12.0, 13.0, 13.0, 13.0, 14.0, 15.23606797749979, 15.0, 13.0],
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(bizinter(X, dim=0), expected_dim0)
    assert torch.allclose(bizinter(X, dim=1), expected_dim1)


def test_corefrmod2() -> None:
    global X
    a = torch.tensor(0.75)
    expected_dim1 = torch.tensor(
        [
            [
                -1.36184521845186 - 1j * 0.64900545553216,
                2.21553699686605 - 1j * 0.163089343860798,
                5.90466252162159 + 1j * 0.960869485102052,
                2.6190623113216 - 1j * 0.989587939424265,
                1.40920987087631 + 1j * 1.07955232856755,
            ],
            [
                -3.25308504420768 + 1j * 0.478749621223554,
                5.11513894391822 - 1j * 4.34264276352112,
                13.108680237035 + 1j * 4.49576846520802,
                6.33757909783394 - 1j * 1.53034701844948,
                3.01736129818456 + 1j * 1.1205972830012,
            ],
            [
                -5.14432486996349 + 1j * 1.60650469797927,
                8.01474089097039 - 1j * 8.52219618318144,
                20.3126979524484 + 1j * 8.03066744531398,
                10.0560958843463 - 1j * 2.0711060974747,
                4.6255127254928 + 1j * 1.16164223743486,
            ],
        ],
    )

    expected_dim0 = torch.tensor(
        [
            [
                5.82955200701331 - 1j * 2.37814084730481,
                5.89597726918259 - 1j * 3.16659930781413,
                5.96240253135188 - 1j * 3.95505776832344,
                6.02882779352117 - 1j * 4.74351622883276,
                6.09525305569046 - 1j * 5.53197468934207,
            ],
            [
                10.0025122796746 + 1j * 0.158954120796182,
                11.1947232913311 + 1j * 0.783395311083628,
                12.3869343029875 + 1j * 1.40783650137108,
                13.5791453146439 + 1j * 2.03227769165852,
                14.7713563263003 + 1j * 2.65671888194597,
            ],
            [
                -2.27206766863141 + 1j * 4.91061123035583,
                -2.2249437518112 + 1j * 5.46393340409701,
                -2.17781983499099 + 1j * 6.01725557783819,
                -2.13069591817077 + 1j * 6.57057775157936,
                -2.08357200135056 + 1j * 7.12389992532054,
            ],
        ]
    )

    assert torch.allclose(corefrmod2(X, a, dim=0), expected_dim0)
    assert torch.allclose(corefrmod2(X, a, dim=1), expected_dim1)


def test_get_mul_dim_einstr() -> None:
    assert _get_mul_dim_einstr(3, 0) == "...abc,a->...abc"
    assert _get_mul_dim_einstr(3, -3) == "...abc,a->...abc"

    assert _get_mul_dim_einstr(3, 1) == "...ab,a->...ab"
    assert _get_mul_dim_einstr(3, -2) == "...ab,a->...ab"

    assert _get_mul_dim_einstr(3, 2) == "...a,a->...a"
    assert _get_mul_dim_einstr(3, -1) == "...a,a->...a"


def test_get_mul_dim_einstr_error() -> None:
    with pytest.raises(ValueError):
        _get_mul_dim_einstr(3, 3)
    with pytest.raises(ValueError):
        _get_mul_dim_einstr(3, 4)
    with pytest.raises(ValueError):
        _get_mul_dim_einstr(3, -4)
