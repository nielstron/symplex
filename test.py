from sympy import Matrix, Rational
from simplex import *
from verification import *
from perturb import *


def test_ex74():
    # exercise 7.4
    A = Matrix([[-3,1],[-2,1],[-1,1],[6,1],[0,-1]])
    m, n = A.shape
    b = Matrix([0, 1, 3, 24, 0])
    c = Matrix([1,1])
    assert is_generic(A, b)
    res, v_star, opt_val = simplex_full(A, b, c)
    assert res == SimplexResult.OPTIMAL
    assert v_star == Matrix([3, 6])


def test_ex82():
    A = Matrix([
        [0, 1, 0],
        [0, 0, 1],
        [1, -1, -1],
        [3, 2, 2],
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ])
    b = Matrix(
        [6, 9, 3, 24, 0, 0, 0]
    )
    x_0 = Matrix([8,0,0])
    c = Matrix([1,1,-1])
    I_x_0 = active_constraints(x_0, A, b)
    assert I_x_0 == [3,5,6]
    assert not is_contained(x_0, A, b)

    v_feasible = determine_feasible_vertex(A, b, I_x_0, PivotRule.MAXIMAL)
    assert v_feasible == Matrix([6, 0, 3])

    x_p = Matrix([0,0,9])
    assert is_contained(x_p, A, b)
    B = next(iter(bases(x_p, A, b)))
    res, v_star, opt_val = simplex(A, b, c, x_p, set(B), PivotRule.MAXIMAL)
    assert v_star == Matrix([4, 6, 0])


def test_ex81():
    A = Matrix([
        [0, 1, 0],
        [0, 0, 1],
        [1, -1, -1],
        [3, 2, 2],
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ])
    b = Matrix(
        [6, 9, 3, 24, 0, 0, 0]
    )
    v_feasible = determine_feasible_vertex2(A, b, PivotRule.MAXIMAL)
    assert is_contained(v_feasible, A, b)
    assert is_vertex(v_feasible, A, b)


def test_example3428():
    A = Matrix([
        [1, 2, 1],
        [-2, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    m, n = A.shape
    b = Matrix([3, 0, 1, 1, 1, 0, 0, 0])
    c = Matrix([1,1,1])
    B = {2,6,7}
    v_3 = Matrix([1,0,0])
    simplex(A, b, c, v_3, B)
    e = pertubation_vector(range(m), Rational(1,64))
    v_3_e = v_3 + (sub_matrix(A, B)**-1*sub_matrix(e, B))
    b_e = b + e
    simplex(A, b_e, c, v_3_e, B)
    e = pertubation_vector([5,4,3,1,0,7,6,2], Rational(1,64))
    v_3_e = v_3 + (sub_matrix(A, B)**-1*sub_matrix(e, B))
    b_e = b + e
    simplex(A, b_e, c, v_3_e, B)

if __name__ == '__main__':
    test_example3428()
    test_ex74()
    test_ex81()
    test_ex82()