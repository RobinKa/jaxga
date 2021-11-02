import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tfga import GeometricAlgebra
import tensorflow as tf
import pytest

def _tfga_mul(ga, a, b):
    return ga.geom_prod(a, b)


@pytest.mark.parametrize("num_bases", list(range(1, 10)))
def test_tfga_mul_mv_mv(num_bases, benchmark):
    ga = GeometricAlgebra([1] * num_bases)
    a = tf.ones([100, ga.num_blades])
    b = tf.ones([100, ga.num_blades])
    _tfga_mul(ga, a, b)
    benchmark(_tfga_mul, ga, a, b)
