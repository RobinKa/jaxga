import jax
import jax.numpy as jnp
from ..jaxga import dual_blade_index
from functools import cache


@cache
def get_mv_dual(a_blade_indices, dims):
    out_blade_indices = []
    signs = jnp.empty([len(a_blade_indices)], dtype=jnp.float32)
    for i, blade_index in enumerate(a_blade_indices):
        sign, dual_index = dual_blade_index(blade_index, dims)
        out_blade_indices.append(dual_index)
        signs = signs.at[i].set(sign)

    def _values_mv_dual(a_values):
        return a_values * signs

    _values_mv_dual_jit = jax.jit(_values_mv_dual)

    return _values_mv_dual_jit, tuple(out_blade_indices)
