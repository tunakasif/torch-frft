import torch

from trainable_frft.fracf_torch import bizdec, bizinter, upsample2

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
