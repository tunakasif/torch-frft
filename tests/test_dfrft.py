from pathlib import Path

import pytest
import scipy
import torch

from torch_frft.dfrft import (
    _circulant,
    _conv1d_full,
    _create_hamiltonian,
    _create_odd_even_decomp_matrix,
    _dfrft_index,
    _get_dfrft_evecs,
    dfrftmtx,
)

test_data_path = Path(__file__).parent.joinpath("data")


def test_dfrft() -> None:
    with pytest.raises(ValueError):
        dfrftmtx(-5, 1.0)
    with pytest.raises(ValueError):
        dfrftmtx(0, 1.0)
    with pytest.raises(ValueError):
        dfrftmtx(4, 1.0, approx_order=1)

    tol = 1e-4
    assert torch.allclose(
        dfrftmtx(4, 0.5),
        torch.tensor(
            [
                [
                    0.707106781186547 - 1j * 0.25,
                    0.353553390593274 + 1j * 0.25,
                    -1.75515849680556e-16 + 1j * 0.25,
                    0.353553390593274 + 1j * 0.25,
                ],
                [
                    0.353553390593274 + 1j * 0.25,
                    0.353553390593274 - 1j * 0.603553390593274,
                    0.353553390593274 - 1j * 0.25,
                    -0.353553390593274 + 1j * 0.103553390593274,
                ],
                [
                    -1.75515849680556e-16 + 1j * 0.25,
                    0.353553390593274 - 1j * 0.25,
                    -0.707106781186548 - 1j * 0.25,
                    0.353553390593274 - 1j * 0.25,
                ],
                [
                    0.353553390593274 + 1j * 0.25,
                    -0.353553390593274 + 1j * 0.103553390593274,
                    0.353553390593274 - 1j * 0.25,
                    0.353553390593274 - 1j * 0.603553390593274,
                ],
            ],
            dtype=torch.complex64,
        ),
        atol=tol,
    )
    assert torch.allclose(
        dfrftmtx(5, 1.5),
        torch.tensor(
            [
                [
                    0.606748024640971 + 1j * 0.276393202250021,
                    0.419252573723482 - 1j * 0.223606797749979,
                    -0.0442616718885028 - 1j * 0.223606797749979,
                    -0.0442616718885028 - 1j * 0.223606797749979,
                    0.419252573723482 - 1j * 0.223606797749979,
                ],
                [
                    0.419252573723482 - 1j * 0.223606797749979,
                    -0.309203704635029 - 1j * 0.172651691155779,
                    0.0816915503331466 + 1j * 0.180901699437495,
                    0.453439584793331 + 1j * 0.180901699437495,
                    0.292297250372517 + 1j * 0.534455090030769,
                ],
                [
                    -0.0442616718885028 - 1j * 0.223606797749979,
                    0.0816915503331466 + 1j * 0.180901699437495,
                    0.00582969231454343 - 1j * 0.172651691155779,
                    -0.595671262693003 + 1j * 0.534455090030768,
                    0.453439584793331 + 1j * 0.180901699437495,
                ],
                [
                    -0.0442616718885028 - 1j * 0.223606797749979,
                    0.453439584793331 + 1j * 0.180901699437495,
                    -0.595671262693003 + 1j * 0.534455090030768,
                    0.00582969231454343 - 1j * 0.172651691155779,
                    0.0816915503331466 + 1j * 0.180901699437495,
                ],
                [
                    0.419252573723482 - 1j * 0.223606797749979,
                    0.292297250372517 + 1j * 0.534455090030769,
                    0.453439584793331 + 1j * 0.180901699437495,
                    0.0816915503331466 + 1j * 0.180901699437495,
                    -0.309203704635029 - 1j * 0.172651691155779,
                ],
            ],
            dtype=torch.complex64,
        ),
        atol=tol,
    )


def test_large_dfrft() -> None:
    global test_data_path

    tol = 1e-5
    for dataname in ("dFRT_N128_a125.mat", "dFRT_N256_a075.mat"):
        mat_data = scipy.io.loadmat(
            test_data_path.joinpath(dataname),
            variable_names=[
                "matrix",
                "a",
                "N",
            ],
            squeeze_me=True,
        )
        N = int(mat_data["N"])
        a = float(mat_data["a"])
        X = torch.tensor(mat_data["matrix"], dtype=torch.complex64)
        assert torch.allclose(dfrftmtx(N, a), X, atol=tol)


def test_dfrft_index() -> None:
    with pytest.raises(ValueError):
        _dfrft_index(-2)
    with pytest.raises(ValueError):
        _dfrft_index(0)

    assert _dfrft_index(1).tolist() == [0]
    assert _dfrft_index(2).tolist() == [0, 2]
    assert _dfrft_index(3).tolist() == [0, 1, 2]
    assert _dfrft_index(4).tolist() == [0, 1, 2, 4]
    assert _dfrft_index(100).tolist() == list(range(99)) + [100]
    assert _dfrft_index(101).tolist() == list(range(101))


def test_circulant() -> None:
    random_vector = torch.rand(100, dtype=torch.float32)
    vector = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    excepted = torch.tensor(
        [
            [1, 4, 3, 2],
            [2, 1, 4, 3],
            [3, 2, 1, 4],
            [4, 3, 2, 1],
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(_circulant(vector), excepted)
    assert torch.allclose(
        _circulant(random_vector),
        torch.tensor(scipy.linalg.circulant(random_vector), dtype=torch.float32),
    )


def test_creates() -> None:
    excepted_N4o2 = torch.tensor(
        [
            [2, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, -2, 1],
            [1, 0, 1, 0],
        ],
        dtype=torch.float32,
    )
    excepted_N6o2 = torch.tensor(
        [
            [2, 1, 0, 0, 0, 1],
            [1, 1, 1, 0, 0, 0],
            [0, 1, -1, 1, 0, 0],
            [0, 0, 1, -2, 1, 0],
            [0, 0, 0, 1, -1, 1],
            [1, 0, 0, 0, 1, 1],
        ],
        dtype=torch.float32,
    )
    excepted_N6o4 = torch.tensor(
        [
            [2.5, 1.33333333333333, -0.0833333333333333, 0, -0.0833333333333333, 1.33333333333333],
            [
                1.33333333333333,
                1.41666666666667,
                1.33333333333333,
                -0.0833333333333333,
                0,
                -0.0833333333333333,
            ],
            [
                -0.0833333333333333,
                1.33333333333333,
                -1.25,
                1.33333333333333,
                -0.0833333333333333,
                0,
            ],
            [
                0,
                -0.0833333333333333,
                1.33333333333333,
                -2.83333333333333,
                1.33333333333333,
                -0.0833333333333333,
            ],
            [
                -0.0833333333333333,
                0,
                -0.0833333333333333,
                1.33333333333333,
                -1.25,
                1.33333333333333,
            ],
            [
                1.33333333333333,
                -0.0833333333333333,
                0,
                -0.0833333333333333,
                1.33333333333333,
                1.41666666666667,
            ],
        ],
        dtype=torch.float32,
    )

    tol = 1e-5
    with pytest.raises(ValueError):
        _create_hamiltonian(0)
    with pytest.raises(ValueError):
        _create_hamiltonian(4, approx_order=1)
    assert torch.allclose(_create_hamiltonian(4, approx_order=2), excepted_N4o2, atol=tol)
    assert torch.allclose(_create_hamiltonian(6, approx_order=2), excepted_N6o2, atol=tol)
    assert torch.allclose(_create_hamiltonian(6, approx_order=4), excepted_N6o4, atol=tol)


def test_createp() -> None:
    with pytest.raises(ValueError):
        _create_odd_even_decomp_matrix(0)
    assert torch.allclose(_create_odd_even_decomp_matrix(1), torch.tensor([1.0]))
    assert torch.allclose(_create_odd_even_decomp_matrix(2), torch.eye(2, dtype=torch.float32))
    assert torch.allclose(
        _create_odd_even_decomp_matrix(3),
        torch.tensor(
            [
                [1.0, 0, 0],
                [0, 0.707106781186547, 0.707106781186547],
                [0, 0.707106781186547, -0.707106781186547],
            ]
        ),
        atol=1e-5,
    )
    assert torch.allclose(
        _create_odd_even_decomp_matrix(4),
        torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 0.707106781186547, 0, 0.707106781186547],
                [0, 0, 1, 0],
                [0, 0.707106781186547, 0, -0.707106781186547],
            ]
        ),
        atol=1e-5,
    )


def test_conv() -> None:
    assert torch.allclose(
        _conv1d_full(
            torch.tensor([1, 2, 3, 4], dtype=torch.float32),
            torch.tensor([1, -1, 2], dtype=torch.float32),
        ),
        torch.tensor([1, 1, 3, 5, 2, 8], dtype=torch.float32),
    )

    assert torch.allclose(
        _conv1d_full(
            torch.tensor([1, 1, 1, 1], dtype=torch.float32),
            torch.tensor([-2, 0, 5], dtype=torch.float32),
        ),
        torch.tensor([-2, -2, 3, 3, 5, 5], dtype=torch.float32),
    )


def test_dis_s() -> None:
    tol = 1e-5
    assert torch.allclose(
        _get_dfrft_evecs(3),
        torch.tensor(
            [
                [-0.888073833977115, 0, 0.459700843380983],
                [-0.325057583671868, 0.707106781186547, -0.627963030199554],
                [-0.325057583671868, -0.707106781186547, -0.627963030199554],
            ]
        ),
        atol=tol,
    )
    assert torch.allclose(
        _get_dfrft_evecs(4),
        torch.tensor(
            [
                [0.853553390593274, 0, 0.5, 0.146446609406726],
                [0.353553390593274, 0.707106781186547, -0.5, -0.353553390593274],
                [0.146446609406726, 0, -0.5, 0.853553390593274],
                [0.353553390593274, -0.707106781186547, -0.5, -0.353553390593274],
            ]
        ),
        atol=tol,
    )
    assert torch.allclose(
        _get_dfrft_evecs(8),
        torch.tensor(
            [
                [
                    0.720381832052063,
                    0,
                    -0.544895106775819,
                    0,
                    0.397052243880409,
                    0,
                    0.162211674410729,
                    0.0132750508655158,
                ],
                [
                    0.464883125942374,
                    0.603553390593274,
                    0.25,
                    0.353553390593274,
                    -0.397052243880409,
                    0.103553390593274,
                    -0.25,
                    -0.035116874057626,
                ],
                [
                    0.151945315516416,
                    0.353553390593274,
                    0.461939766255643,
                    -0.5,
                    0.164464424385935,
                    -0.353553390593274,
                    0.461939766255644,
                    0.151945315516416,
                ],
                [
                    0.0351168740576261,
                    0.103553390593274,
                    0.25,
                    -0.353553390593274,
                    0.397052243880409,
                    0.603553390593274,
                    -0.25,
                    -0.464883125942374,
                ],
                [
                    0.0132750508655158,
                    0,
                    0.162211674410729,
                    0,
                    0.397052243880409,
                    0,
                    -0.544895106775819,
                    0.720381832052063,
                ],
                [
                    0.0351168740576261,
                    -0.103553390593274,
                    0.25,
                    0.353553390593274,
                    0.397052243880409,
                    -0.603553390593274,
                    -0.25,
                    -0.464883125942374,
                ],
                [
                    0.151945315516416,
                    -0.353553390593274,
                    0.461939766255643,
                    0.5,
                    0.164464424385935,
                    0.353553390593274,
                    0.461939766255644,
                    0.151945315516416,
                ],
                [
                    0.464883125942374,
                    -0.603553390593274,
                    0.25,
                    -0.353553390593274,
                    -0.397052243880409,
                    -0.103553390593274,
                    -0.25,
                    -0.035116874057626,
                ],
            ]
        ),
        atol=tol,
    )
