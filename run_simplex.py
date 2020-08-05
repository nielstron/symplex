from sympy import Matrix, Float
from simplex import *
from verification import *


def test_ex74():
    # exercise 7.4
    A = Matrix([[-3,1],[-2,1],[-1,1],[6,1],[0,-1]])
    m, n = A.shape
    b = Matrix([0, 1, 3, 24, 0])
    c = Matrix([1,1])
    print(is_generic(A, b))
    eps = Float(2**-2)
    e = Matrix([eps**(i+1) for i in range(m)])
    #print(is_generic(A, b+e)) does not find a basis, prob due to rounding errors
    res, v_star, opt_val = simplex_full(A, b, c)
    assert res == SimplexResult.OPTIMAL
    assert v_star == Matrix([3, 6])

if __name__ == '__main__':
    test_ex74()