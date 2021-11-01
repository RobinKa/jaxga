import itertools
import jax
import jax.numpy as jnp
from ..jaxga import _reduce_bases
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
        zip(range(len(a_blade_indices)), a_blade_indices),
        zip(range(len(b_blade_indices)), b_blade_indices)
    ):
        out_sign, out_index = _reduce_bases(index_a, index_b, signature)
        if out_sign != 0 and prod == "gp" or (prod == "op" and len(out_index) == abs(len(index_a) + len(index_b))) or (prod == "ip" and len(out_index) == abs(len(index_a) - len(index_b))):
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

        def _values_mv_mul(a_values, b_values):
            out_batch_shape = jnp.broadcast_shapes(a_values.shape[1:], b_values.shape[1:])
            out_values = jnp.zeros([out_size, *out_batch_shape], dtype=jnp.float32)

            for index_a, index_b, out_sign, out_index in zip(indices_a, indices_b, out_signs, out_indices):
                out_values = out_values.at[out_index].add(
                    out_sign * a_values[index_a] * b_values[index_b]
                )

            return out_values

    _values_mv_mul_jit = jax.jit(_values_mv_mul, static_argnames=["out_size"])

    return _values_mv_mul_jit, tuple(out_blade_indices)
