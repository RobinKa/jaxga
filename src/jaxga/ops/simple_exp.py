import jax
import jax.numpy as jnp
from .multiply import get_mv_multiply
from .add import get_mv_add
from .select import get_mv_select
from ..jaxga import is_scalar_index
from functools import cache


@cache
def get_mv_simple_exp(a_blade_indices, signature):
    mv_multiply, a_sq_indices = get_mv_multiply(
        a_blade_indices, a_blade_indices, signature
    )
    mv_select_scalar, scalar_indices = get_mv_select(
        a_sq_indices, is_scalar_index
    )
    mv_add, out_indices = get_mv_add(scalar_indices, a_blade_indices)

    def _values_mv_simple_exp(a_values):
        a_sq_values = mv_select_scalar(mv_multiply(a_values, a_values))
        a_sq_sqrt = jnp.sign(a_sq_values) * jnp.sqrt(jnp.abs(a_sq_values))

        out_scalar = jnp.where(
            a_sq_sqrt < 0,
            jnp.cos(a_sq_sqrt),
            jnp.where(
                a_sq_sqrt > 0,
                jnp.cosh(a_sq_sqrt),
                jnp.ones_like(a_sq_sqrt)
            )
        )

        out_blade = jnp.where(
            a_sq_sqrt < 0,
            a_values / a_sq_sqrt * jnp.sin(a_sq_sqrt),
            jnp.where(
                a_sq_sqrt > 0,
                a_values / a_sq_sqrt * jnp.sinh(a_sq_sqrt),
                a_values
            )
        )

        return mv_add(out_scalar, out_blade)

    _values_mv_simple_exp_jit = jax.jit(_values_mv_simple_exp)

    return _values_mv_simple_exp_jit, out_indices
