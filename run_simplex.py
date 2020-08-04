from simplex import *
from verification import *


def test_ex74():
    # exercise 7.4
    A = Matrix([[-3,1],[-2,1],[-1,1],[6,1],[0,-1]])
    b = Matrix([0, 1, 3, 24, 0])
    c = Matrix([1,1])
    res, v_star, opt_val = simplex_full(A, b, c)
    assert res == SimplexResult.OPTIMAL
    assert v_star == Matrix([3, 6])
