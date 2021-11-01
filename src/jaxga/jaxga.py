from functools import cache
from .signatures import positive_signature


def blade_name(blade_indices):
    return "e_{%s}" % ", ".join(str(ind) for ind in blade_indices)


def mv_repr(indices, values):
    return "Multivector(%s)" % " + ".join(
        "%s %s" % (val, blade_name(ind))
        for (ind, val) in zip(indices, values)
    )


def _normal_swap(x):
    for i in range(len(x) - 1):
        a, b = x[i], x[i + 1]
        if a > b:
            x[i], x[i+1] = b, a
            return False, x
    return True, x


def get_normal_ordered(blade_name):
    blade_name = list(blade_name)
    sign = -1
    done = False
    while not done:
        sign *= -1
        done, blade_name = _normal_swap(blade_name)
    return sign, tuple(blade_name)


def _collapse_same(x):
    for i in range(len(x) - 1):
        a, b = x[i], x[i + 1]
        if a == b:
            return False, x[:i] + x[i+2:], a
    return True, x, None


@cache
def _reduce_bases(a, b, metric):
    combined = list(a + b)

    # Bring into normal order:
    sign, combined = get_normal_ordered(combined)

    done = False
    while not done:
        done, combined, combined_elem = _collapse_same(combined)
        if not done:
            sign *= metric(combined_elem)

    return sign, tuple(combined)


def is_scalar_index(b):
    return b == ()


def pseudoscalar_index(dims):
    return tuple(range(dims))


def dual_blade_index(blade_index, dims):
    pss = pseudoscalar_index(dims)
    blade_index_reverse = tuple(reversed(blade_index))
    return _reduce_bases(blade_index_reverse, pss, positive_signature)


def reverse_indices(blade_indices):
    return [tuple(reversed(blade_index)) for blade_index in blade_indices]
