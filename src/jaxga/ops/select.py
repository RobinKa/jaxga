import jax
import jax.numpy as jnp
from functools import cache


@cache
def get_mv_select(a_blade_indices, select_index):
    out_blade_indices = []
    out_a_indices = []
    for i, a_index in enumerate(a_blade_indices):
        if select_index(a_index):
            out_blade_indices.append(a_index)
            out_a_indices.append(i)
    out_a_indices = jnp.array(out_a_indices, dtype=jnp.int32)

    def _values_mv_select(a_values):
        return a_values[out_a_indices]

    _values_mv_select_jit = jax.jit(_values_mv_select)

    return _values_mv_select_jit, tuple(out_blade_indices)
