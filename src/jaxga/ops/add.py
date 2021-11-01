import jax
import jax.numpy as jnp
from .reduce_same import get_mv_reduce_same
from functools import cache


@cache
def get_mv_add(a_blade_indices, b_blade_indices):
    out_blade_indices = a_blade_indices + b_blade_indices
    mv_reduce_same, out_blade_indices = get_mv_reduce_same(out_blade_indices)

    def _values_mv_reduce_same(a_values, b_values):
        return mv_reduce_same(jnp.concatenate([a_values, b_values], axis=0))
    _values_mv_add_jit = jax.jit(_values_mv_reduce_same)

    return _values_mv_add_jit, out_blade_indices
