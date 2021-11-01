import jax
from functools import cache
from .multiply import get_mv_multiply
from .select import get_mv_select
from .add import get_mv_add
from ..jaxga import is_scalar_index


@cache
def get_mv_inverse(a_blade_indices, signature):
    dims = len(set(a_blade_indices))
    n = 2 ** ((dims + 1) // 2)

    last_ind = a_blade_indices

    selects = []
    adds = []
    mults = []
    for k in range(1, n):
        select, select_ind = get_mv_select(last_ind, is_scalar_index)
        add, add_ind = get_mv_add(last_ind, select_ind)
        mult, mult_ind = get_mv_multiply(a_blade_indices, add_ind, signature)

        selects.append(select)
        adds.append(add)
        mults.append(mult)

        last_ind = mult_ind

    select_last_scalar, _ = get_mv_select(last_ind, is_scalar_index)

    def _values_mv_inverse(a_values):
        u = a_values
        for k, (select, add, mult) in enumerate(zip(selects, adds, mults)):
            c = n / (k + 1) * select(u)
            u_minus_c = add(u, -c)
            u = mult(a_values, u_minus_c)

        return u_minus_c / select_last_scalar(u)
    _values_mv_inverse_jit = jax.jit(_values_mv_inverse)

    return _values_mv_inverse_jit, add_ind
