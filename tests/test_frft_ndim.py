from pathlib import Path

import scipy
import torch

from torch_frft.frft_module import frft, frft_shifted

test_data_path = Path(__file__).parent.joinpath("data")
X = torch.tensor(
    [
        [0, 0, 0, 0],
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20],
    ]
)


def test_frft_along_dims() -> None:
    global X
    a = 0.75
    expected_dim0 = torch.tensor(
        [
            [
                -1.639937532870064 - 1j * 1.260544830999431,
                -2.053600005144979 - 1j * 1.047276319302379,
                -2.467262477419891 - 1j * 0.834007807605332,
                -2.880924949694803 - 1j * 0.620739295908285,
            ],
            [
                2.692285882089642 - 1j * 1.831901311430273,
                2.563548771595040 - 1j * 2.193833902635730,
                2.434811661100438 - 1j * 2.555766493841186,
                2.306074550605834 - 1j * 2.917699085046646,
            ],
            [
                -0.279554616944487 + 1j * 3.275718988216769,
                0.191658793730746 + 1j * 2.955390997071200,
                0.662872204405978 + 1j * 2.635063005925632,
                1.134085615081208 + 1j * 2.314735014780063,
            ],
            [
                17.154400630655218 + 1j * 4.172078412430205,
                19.060445145172462 + 1j * 4.635642680478006,
                20.966489659689707 + 1j * 5.099206948525807,
                22.872534174206958 + 1j * 5.562771216573609,
            ],
            [
                8.761396009098664 - 1j * 9.041622828837008,
                9.232609419773897 - 1j * 9.361950819982576,
                9.703822830449129 - 1j * 9.682278811128146,
                10.175036241124364 - 1j * 10.002606802273716,
            ],
            [
                -5.009553870992480 - 1j * 4.682885330267955,
                -5.138290981487083 - 1j * 5.044817921473413,
                -5.267028091981683 - 1j * 5.406750512678871,
                -5.395765202476286 - 1j * 5.768683103884329,
            ],
        ]
    )
    expected_dim1 = torch.tensor(
        [
            [0.0 + 1j * 0.0, 0.0 + 1j * 0.0, 0.0 + 1j * 0.0, 0.0 + 1j * 0.0],
            [
                0.012969158656193 - 1j * 0.401447437522288,
                0.624145873079738 + 1j * 0.282305389191933,
                4.889045010474662 + 1j * 0.565406277873251,
                1.635121448621091 - 1j * 1.471191310147956,
            ],
            [
                -0.881766803308353 + 1j * 0.473105941060268,
                1.227022272584351 - 1j * 1.683852318466376,
                12.296603344534061 + 1j * 2.382750970285592,
                2.765343152312703 - 1j * 3.511950202280931,
            ],
            [
                -1.776502765272898 + 1j * 1.347659319642825,
                1.829898672088962 - 1j * 3.650010026124682,
                19.704161678593458 + 1j * 4.200095662697935,
                3.895564856004320 - 1j * 5.552709094413909,
            ],
            [
                -2.671238727237449 + 1j * 2.222212698225386,
                2.432775071593575 - 1j * 5.616167733782992,
                27.111720012652853 + 1j * 6.017440355110280,
                5.025786559695932 - 1j * 7.593467986546881,
            ],
            [
                -3.565974689201997 + 1j * 3.096766076807937,
                3.035651471098188 - 1j * 7.582325441441302,
                34.519278346712255 + 1j * 7.834785047522617,
                6.156008263387545 - 1j * 9.634226878679856,
            ],
        ]
    )

    assert torch.allclose(frft(X, a, dim=0), expected_dim0)
    assert torch.allclose(frft(X, a, dim=1), expected_dim1)
    assert torch.allclose(frft(X, a), expected_dim1)


def test_frftn() -> None:
    global X
    a0 = 0.80
    a1 = 1.25
    tol = 1e-5
    expected = torch.tensor(
        [
            [
                0.688842550742503 - 1j * 1.05025022587593,
                -1.5232285106647 - 1j * 0.743334167643746,
                -4.58605878637312 + 1j * 2.84544820399633,
                -0.349084616201554 - 1j * 0.616835515516242,
            ],
            [
                -0.0999660396698176 + 1j * 1.81524234357208,
                2.13908086021153 - 1j * 0.481447770834742,
                1.68345545085984 - 1j * 7.43616563323154,
                1.4051660436296 + 1j * 0.522319644772944,
            ],
            [
                -0.56304360405674 - 1j * 1.98256969443392,
                -1.56516443709101 + 1j * 1.46044276194116,
                3.57851342184798 + 1j * 7.94031937080238,
                -2.29875957970397 + 1j * 0.288031570470201,
            ],
            [
                -11.6066292756661 + 1j * 3.72969786803663,
                5.5754363968664 + 1j * 12.3183955439532,
                40.1508076837946 - 1j * 0.934258295678831,
                2.23345435548267 + 1j * 7.82687317811882,
            ],
            [
                -2.73051982783143 + 1j * 7.19439803347227,
                7.7240001751366 + 1j * 1.51060037126872,
                11.6968058241686 - 1j * 21.3261032159491,
                5.70405160346758 + 1j * 2.02432220126325,
            ],
            [
                4.24890047743273 + 1j * 0.683676796634407,
                0.090562509169084 - 1j * 4.40583541439558,
                -12.454278311989 - 1j * 4.33550012916856,
                0.353561401416305 - 1j * 3.23584505823459,
            ],
        ]
    )

    assert torch.allclose(frft(frft(X, a0, dim=0), a1, dim=1), expected, atol=tol)


def test_base_case() -> None:
    X = torch.rand(100, 100)
    tol = 1e-4

    assert torch.allclose(
        frft_shifted(X, 1.0, dim=-1),
        torch.fft.fft(X, norm="ortho", dim=-1),
        atol=tol,
    )

    assert torch.allclose(
        frft_shifted(X, 1.0, dim=0),
        torch.fft.fft(X, norm="ortho", dim=0),
        atol=tol,
    )

    assert torch.allclose(
        frft_shifted(frft_shifted(X, 1.0, dim=0), 1.0, dim=1),
        torch.fft.fft2(X, norm="ortho"),
        atol=tol,
    )
    assert torch.allclose(
        frft_shifted(frft_shifted(X, -1.0, dim=0), -1.0, dim=1),
        torch.fft.ifftn(X, norm="ortho"),
        atol=tol,
    )
    assert torch.allclose(
        frft_shifted(frft_shifted(X, 2.0, dim=0), 2.0, dim=1).to(torch.complex64),
        torch.fft.fftn(torch.fft.fftn(X, norm="ortho"), norm="ortho"),
        atol=tol,
    )


def test_3D() -> None:
    global test_data_path

    tol = 1e-5
    for dataname in ("rand3d_a090.mat", "ones3d_a065.mat"):
        mat_data = scipy.io.loadmat(
            test_data_path.joinpath(dataname),
            variable_names=[
                "input",
                "a",
                "expected_dim0",
                "expected_dim1",
                "expected_dim2",
            ],
            squeeze_me=True,
        )
        a = torch.tensor(mat_data["a"], dtype=torch.float32)
        X = torch.tensor(mat_data["input"], dtype=torch.float32)
        expected_dim0 = torch.tensor(mat_data["expected_dim0"], dtype=torch.complex64)
        expected_dim1 = torch.tensor(mat_data["expected_dim1"], dtype=torch.complex64)
        expected_dim2 = torch.tensor(mat_data["expected_dim2"], dtype=torch.complex64)

        assert torch.allclose(frft(X, a, dim=0), expected_dim0, atol=tol)
        assert torch.allclose(frft(X, a, dim=1), expected_dim1, atol=tol)
        assert torch.allclose(frft(X, a, dim=2), expected_dim2, atol=tol)
