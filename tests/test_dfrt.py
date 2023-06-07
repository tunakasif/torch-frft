import jax.numpy as jnp

from trainable_frft.dfrt import cconvm


def test_cconvm() -> None:
    x = jnp.array([1.0, 2, 3, 4, 5])  # first entry 1.0 to make it float
    expected_toeplitz = jnp.array(
        [
            [1.0, 5, 4, 3, 2],
            [2, 1, 5, 4, 3],
            [3, 2, 1, 5, 4],
            [4, 3, 2, 1, 5],
            [5, 4, 3, 2, 1],
        ]
    )
    assert jnp.allclose(cconvm(x), expected_toeplitz)
