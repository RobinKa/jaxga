import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pytest
import jax.numpy as jnp
from jaxga.mv import MultiVector
from jaxga.signatures import positive_signature


def _jaxga_mul(a, b):
    return a * b


def _mv_ones(num_elements, num_bases):
    return MultiVector(
        values=jnp.ones([num_bases, num_elements], dtype=jnp.float32),
        indices=tuple((i,) for i in range(num_bases)),
        signature=positive_signature
    )


@pytest.mark.parametrize("num_bases", list(range(1, 10)))
def test_jaxga_mul_mv_mv(num_bases, benchmark):
    a = _mv_ones(100, num_bases)
    b = _mv_ones(100, num_bases)
    _jaxga_mul(a, b)
    benchmark(_jaxga_mul, a, b)


if __name__ == '__main__':
    pytest.main()
