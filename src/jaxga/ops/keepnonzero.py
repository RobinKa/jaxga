import jax
import jax.numpy as jnp


def get_mv_keep_nonzero(a_blade_indices, a_values):
    out_blade_indices = []
    out_a_indices = []
    for i, a_index in enumerate(a_blade_indices):
        if not jnp.allclose(a_values[i], 0):
            out_blade_indices.append(a_index)
            out_a_indices.append(i)
    out_a_indices = jnp.array(out_a_indices, dtype=jnp.int32)

    def _values_mv_keep_nonzero(a_values):
        return a_values[out_a_indices]

    _values_mv_keep_nonzero_jit = jax.jit(_values_mv_keep_nonzero)

    return _values_mv_keep_nonzero_jit, tuple(out_blade_indices)
