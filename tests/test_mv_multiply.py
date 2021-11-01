from jaxga.mv import MultiVector as MV
from jaxga.signatures import pga_signature
import pytest


def test_mv_multiply_with_e1_e1_result_1():
    a = MV.e(1)
    result = a * a
    assert result.values.shape == (1,)
    assert result.values[0] == 1
    assert result.indices == ((),)


def test_mv_multiply_with_e1_e2_result_e_12():
    a = MV.e(1)
    b = MV.e(2)
    result = a * b
    assert result.values.shape == (1,)
    assert result.values[0] == 1
    assert result.indices == ((1, 2),)


def test_mv_multiply_with_e12_e12_result_minus_1():
    a = MV.e(1, 2)
    result = a * a
    assert result.values.shape == (1,)
    assert result.values[0] == -1
    assert result.indices == ((),)


def test_mv_multiply_with_pga_e0_e0_result_():
    a = MV.e(0, signature=pga_signature)
    result = a * a
    assert result.values.shape == ()
    assert result.values == 0
    assert result.indices == ()


def test_mv_multiply_with_pga_e01_e02_result_0():
    a = MV.e(0, 1, signature=pga_signature)
    b = MV.e(0, 2, signature=pga_signature)
    result = a * b
    assert result.values.shape == ()
    assert result.values == 0
    assert result.indices == ()


if __name__ == '__main__':
    pytest.main()
