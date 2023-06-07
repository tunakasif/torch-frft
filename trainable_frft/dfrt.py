import jax.numpy as jnp
from jax import Array, jit
from jax.scipy.linalg import toeplitz


@jit
def cconvm(c: Array) -> Array:
    r = jnp.concatenate([jnp.array([c[0]]), jnp.flip(jnp.array(c[1:]))])
    return toeplitz(c, r)
