from jaxga.mv import MultiVector as MV
import pytest


def test_mv_dual_with_3d_e0_result_e12():
    a = MV.e(0)
    result = a.dual(3)
    assert result.values.shape == (1,)
    assert result.values[0] == 1
    assert result.indices == ((1, 2),)

def test_mv_dual_with_3d_e12_result_e0():
    a = MV.e(1, 2)
    result = a.dual(3)
    assert result.values.shape == (1,)
    assert result.values[0] == 1
    assert result.indices == ((0,),)

def test_mv_dual_with_10d_e5_result_minus_e012346789():
    a = MV.e(5)
    result = a.dual(10)
    assert result.values.shape == (1,)
    assert result.values[0] == -1
    assert result.indices == ((0, 1, 2, 3, 4, 6, 7, 8, 9,),)

if __name__ == '__main__':
    pytest.main()
