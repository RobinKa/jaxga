import itertools
import jax
import jax.numpy as jnp
from ..jaxga import reduce_bases, reverse_indices
from functools import cache


@cache
def get_mv_sandwich(a_blade_indices, b_blade_indices, signature, prod="gp"):
    """a b ~a"""
    out_indices = []
    out_blade_indices = []
    out_signs = []
    out_indices = []
    indices_a = []
    indices_b = []
    indices_a_r = []

    blade_to_index = {}

    for (i_a, index_a), (i_b, index_b), (i_a_r, index_a_r) in itertools.product(
        enumerate(a_blade_indices),
        enumerate(b_blade_indices),
        enumerate(reverse_indices(a_blade_indices))
    ):
        out_sign_1, out_index_1 = reduce_bases(index_a, index_b, signature)
        out_sign_2, out_index = reduce_bases(out_index_1, index_a_r, signature)
        out_sign = out_sign_1 * out_sign_2

        if out_sign != 0 and (
            prod == "gp" or
            (prod == "op" and len(out_index) == abs(len(index_a) + len(index_b))) or
            (prod == "ip" and len(out_index) == abs(len(index_a) - len(index_b)))
        ):
            out_signs.append(out_sign)
            indices_a.append(i_a)
            indices_b.append(i_b)
            indices_a_r.append(i_a_r)

            if out_index in blade_to_index:
                out_indices.append(blade_to_index[out_index])
            else:
                blade_to_index[out_index] = len(blade_to_index)
                out_indices.append(blade_to_index[out_index])
                out_blade_indices.append(out_index)

    if len(out_indices) == 0:
        def _values_mv_sandwich(a_values, b_values):
            return jnp.zeros((), dtype=jnp.float32)
    else:
        out_size = max(out_indices) + 1

        def _values_mv_sandwich(a_values, b_values):
            out_batch_shape = jnp.broadcast_shapes(
                a_values.shape[1:], b_values.shape[1:]
            )
            out_values = jnp.zeros(
                [out_size, *out_batch_shape], dtype=jnp.float32
            )

            for index_a, index_b, index_a_r, out_sign, out_index in zip(indices_a, indices_b, indices_a_r, out_signs, out_indices):
                out_values = out_values.at[out_index].add(
                    out_sign * a_values[index_a] * b_values[index_b] * a_values[index_a_r]
                )

            return out_values

    _values_mv_sandwich_jit = jax.jit(_values_mv_sandwich)

    return _values_mv_sandwich_jit, tuple(out_blade_indices)
