import jax
import jax.numpy as jnp
from functools import cache


@cache
def get_mv_reduce_same(a_blade_indices):
    blade_to_index = {}
    out_indices = []

    indices = []
    unique_count = len(set(a_blade_indices))
    out_indices = [[] for _ in range(unique_count)]

    for i, blade_index in enumerate(a_blade_indices):
        if blade_index in blade_to_index:
            index = blade_to_index[blade_index]
        else:
            index = len(blade_to_index)
            blade_to_index[blade_index] = index
            indices.append(blade_index)
        out_indices[index].append(i)

    def _values_mv_reduce_same(a_values):
        out_batch_shape = a_values.shape[1:]
        result = jnp.empty([len(out_indices), *out_batch_shape], dtype=jnp.float32)
        for i, mm in enumerate(out_indices):
            for j, m in enumerate(mm):
                if j == 0:
                    result = result.at[i].set(a_values[m])
                else:
                    result = result.at[i].add(a_values[m])
        return result
    _values_mv_reduce_same_jit = jax.jit(_values_mv_reduce_same)

    return _values_mv_reduce_same_jit, tuple(indices)
