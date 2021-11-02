import clifford as cf
from clifford import MVArray
import numpy as np
import pytest


def _clifford_mul(a, b):
    return a * b


def _mv_ones(D, num_elements, num_bases):
    return MVArray([D.MultiVector(value=np.ones(2**num_bases, dtype=np.float32)) for i in range(num_elements)])


@pytest.mark.parametrize("num_bases", list(range(1, 10)))
def test_clifford_mul_mv_mv(num_bases, benchmark):
    layout, blades = cf.Cl(num_bases)
    a = _mv_ones(layout, 100, num_bases)
    b = _mv_ones(layout, 100, num_bases)
    benchmark(_clifford_mul, a, b)


if __name__ == '__main__':
    pytest.main()
