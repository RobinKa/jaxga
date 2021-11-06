import itertools
import jax
import jax.numpy as jnp
from ..jaxga import reduce_bases
from functools import cache


@cache
def get_mv_multiply(a_blade_indices, b_blade_indices, signature, prod="gp"):
    out_indices = []
    out_blade_indices = []
    out_signs = []
    out_indices = []
    indices_a = []
    indices_b = []

    blade_to_index = {}

    for (i_a, index_a), (i_b, index_b) in itertools.product(
        enumerate(a_blade_indices),
        enumerate(b_blade_indices)
    ):
        out_sign, out_index = reduce_bases(index_a, index_b, signature)
        if out_sign != 0 and (
            prod == "gp" or
            (prod == "op" and len(out_index) == abs(len(index_a) + len(index_b))) or
            (prod == "ip" and len(out_index) == abs(len(index_a) - len(index_b)))
        ):
            out_signs.append(out_sign)
            indices_a.append(i_a)
            indices_b.append(i_b)

            if out_index in blade_to_index:
                out_indices.append(blade_to_index[out_index])
            else:
                blade_to_index[out_index] = len(blade_to_index)
                out_indices.append(blade_to_index[out_index])
                out_blade_indices.append(out_index)

    if len(out_indices) == 0:
        def _values_mv_mul(a_values, b_values):
            return jnp.zeros((), dtype=jnp.float32)
    else:
        out_size = max(out_indices) + 1

        indices_a = jnp.array(indices_a)
        indices_b = jnp.array(indices_b)
        out_signs = jnp.array(out_signs, dtype=jnp.float32)
        out_indices = jnp.array(out_indices)

        def _values_mv_mul(a_values, b_values):
            num_extra_dims = max(len(a_values), len(b_values)) - 1

            segment_out_values = (
                jnp.reshape(out_signs, (-1,) + (1,) * num_extra_dims) * a_values[indices_a] * b_values[indices_b]
            )

            out_values = jax.ops.segment_sum(
                segment_out_values, out_indices, num_segments=out_size
            )

            return out_values

    _values_mv_mul_jit = jax.jit(_values_mv_mul)

    return _values_mv_mul_jit, tuple(out_blade_indices)
