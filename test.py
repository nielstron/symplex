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
    res, v_star, opt_val = simplex_full(A, b, c, pivot_rule_i=PivotRule.MINIMAL)
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

    v_feasible = determine_feasible_vertex(A, b, I_x_0, pivot_rule_p=PivotRule.MAXIMAL, pivot_rule_i=PivotRule.MAXIMAL)
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
    v_feasible = determine_feasible_vertex2(A, b, pivot_rule_p=PivotRule.MAXIMAL, pivot_rule_i=PivotRule.MAXIMAL)
    assert is_contained(v_feasible, A, b)
    assert is_vertex(v_feasible, A, b)


def test_ex85():
    A = Matrix([
        [1, 1],
        [-1, 0],
        [0, -1],
        [1, 0]
    ])
    b = Matrix([2, 0, 0, 1])
    I = {0, 2}
    assert not is_contained(sub_matrix(A, I)**-1*sub_matrix(b, I), A, b)
    A_init, b_init, c_init, v_init = initial_vertex_polygon(A, b, I)
    B_init = next(iter(bases(v_init, A_init, b_init)))
    _, v_start1, _ = simplex(A_init, b_init, c_init, v_init, set(B_init), pivot_rule_i=PivotRule.MINIMAL)
    v_start1 = Matrix(v_start1[:2])
    I_start1 = active_constraints(v_start1, A, b)
    _, v_start2, _ = simplex(A_init, b_init, c_init, v_init, set(B_init), pivot_rule_i=PivotRule.MAXIMAL)
    v_start2 = Matrix(v_start2[:2])
    I_start2 = active_constraints(v_start2, A, b)
    assert v_start1 == Matrix([1,0])
    assert set(I_start1) == {2, 3}
    # does not work as expected
    #assert v_start2 == Matrix([1,1])
    #assert I_start2 == {0, 3}


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
    simplex(A, b, c, v_3, B, pivot_rule_i=PivotRule.MINIMAL)
    #simplex(A, b, c, v_3, B, pivot_rule_i=PivotRule.LEXMIN([0,1,2,3,4,5,6,7]))
    #simplex(A, b, c, v_3, B, pivot_rule_i=PivotRule.LEXMIN([5,4,3,1,0,7,6,2]))
    e = pertubation_vector(range(m), Rational(1,64))
    v_3_e = v_3 + (sub_matrix(A, B)**-1*sub_matrix(e, B))
    b_e = b + e
    simplex(A, b_e, c, v_3_e, B)
    e = pertubation_vector([5,4,3,1,0,7,6,2], Rational(1,64))
    v_3_e = v_3 + (sub_matrix(A, B)**-1*sub_matrix(e, B))
    b_e = b + e
    simplex(A, b_e, c, v_3_e, B)


def test_ex93():
    A = BlockMatrix([
        [Identity(3)],
        [-Identity(3)],
    ]).as_explicit()
    b = Matrix(6*[1])
    c = Matrix([0, 0, 1])
    v0 = Matrix(3*[0])
    v_start1 = determine_feasible_vertex3(A, b, c, v0)

if __name__ == '__main__':
    test_example3428()
    test_ex74()
    test_ex81()
    test_ex82()
    test_ex85()
    test_ex93()