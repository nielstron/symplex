from sympy import Matrix, Rational, Identity, BlockMatrix
from simplex import *
from utils import *
from perturb import *


def test_ex74():
    # exercise 7.4
    A = Matrix([[-3,1],[-2,1],[-1,1],[6,1],[0,-1]])
    m, n = A.shape
    b = Matrix([0, 1, 3, 24, 0])
    c = Matrix([1,1])
    assert is_generic(A, b)
    res, v_star, opt_val = simplex_full(A, b, c, pivot_rule_p=PivotRule.MINIMAL, pivot_rule_i=PivotRule.MINIMAL)
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
    res, v_star, opt_val = simplex(A, b, c, x_p, set(B), pivot_rule_p=PivotRule.MAXIMAL, pivot_rule_i=PivotRule.MAXIMAL)
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
    assert v_start1 == Matrix([1,0])
    assert set(I_start1) == {2, 3}
    # does not work as expected, gives same result as above
    #_, v_start2, _ = simplex(A_init, b_init, c_init, v_init, set(B_init), pivot_rule_i=PivotRule.MAXIMAL)
    #v_start2 = Matrix(v_start2[:2])
    #I_start2 = active_constraints(v_start2, A, b)
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

    # two perturbations are applied
    r1 = range(m)
    r2 = [5, 4, 3, 1, 0, 7, 6, 2]

    # once with explicit pertubation
    e = pertubation_vector(r1, Rational(1,64))
    v_3_e = v_3 + (sub_matrix(A, B)**-1*sub_matrix(e, B))
    b_e = b + e
    res, v_r1_star_expl_pert, _ = simplex(A, b_e, c, v_3_e, B, pivot_rule_i=PivotRule.MINIMAL)
    v_r1_star_expl = v_star_from_perturbed_polygon(A, b, b_e, v_r1_star_expl_pert)

    e = pertubation_vector(r2, Rational(1,64))
    v_3_e = v_3 + (sub_matrix(A, B)**-1*sub_matrix(e, B))
    b_e = b + e
    res, v_r2_star_expl_pert, _ = simplex(A, b_e, c, v_3_e, B, pivot_rule_i=PivotRule.MINIMAL)
    v_r2_star_expl = v_star_from_perturbed_polygon(A, b, b_e, v_r2_star_expl_pert)

    # and once with our fancy lexmin rule
    res, v_r1_star_lexmin, _ = simplex(A, b, c, v_3, B, pivot_rule_i=PivotRule.LEXMIN(r1))
    res, v_r2_star_lexmin, _ = simplex(A, b, c, v_3, B, pivot_rule_i=PivotRule.LEXMIN(r2))
    assert v_r1_star_expl == v_r1_star_lexmin
    assert v_r2_star_expl == v_r2_star_lexmin



def test_ex93():
    A = BlockMatrix([
        [Identity(3)],
        [-Identity(3)],
    ]).as_explicit()
    b = Matrix(6*[1])
    c = Matrix([0, 0, 1])
    v0 = Matrix(3*[0])
    v_start1 = determine_feasible_vertex3(A, b, c, v0)


def test_ex112():
    A = Matrix([
        [1, 1, 1],
        [1, -1, 0]
    ])
    b = Matrix([1, 0])
    c = Matrix([0, 0, -1])
    B = {0,1}
    # Note the primal LP now is max b^Tx for A^Tx <= c which we hence input below
    res, v_star, opt_val, _ = simplex_tableau(A.transpose(), c, b, B, pivot_rule=PivotRule.MAXIMAL)
    assert v_star == Matrix([0, 0, 1])
    assert opt_val == -1


def test_ex113():
    # note the dual is already given in the exercise, so we note down the primal instead
    b = Matrix([15, 0, 6, 17])
    A = Matrix([
        [3, 3, 2, 4],
        [5, -5, 1, 5],
    ]).transpose()
    c = Matrix([5, 6])
    A_init = Matrix([
        [3, 3, 2, 4, 1, 0],
        [5, -5, 1, 5, 0, 1],
    ]).transpose()
    b_init = Matrix([0, 0, 0, 0, 1, 1])
    c_init = c
    B_init = {4, 5}
    _, v_init, _, B = simplex_tableau(A_init, b_init, c_init, B_init)
    res, v_star, opt_val, _ = simplex_tableau(A, b, c, B)
    assert v_star == Matrix([0, 0, Rational(1, 6), Rational(7,6)])
    assert opt_val == Rational(125, 6)



def test_ex121():
    c = Matrix([-1, -1, 1])
    b = Matrix([6+Rational(4,3), 4+Rational(2,3), 6, 4, 0, 0, 0])
    A = Matrix([
        [1, 2, 0],
        [1, 1, 1],
        [3, 0, 1],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    x_bar_0 = Matrix([2, 2+Rational(2,3), 0])
    assert is_contained(x_bar_0, A, b)
    assert set(active_constraints(x_bar_0, A, b)) == {0, 1, 2, 6}
    B = list(bases(x_bar_0,A,b))[-1]
    res, x_star, opt_val = simplex(A, b, c, x_bar_0, set(B), pivot_rule_i=PivotRule.MAXIMAL, pivot_rule_p=PivotRule.MAXIMAL)
    assert x_star == Matrix([0, 0, 4])


if __name__ == '__main__':
    test_example3428()
    test_ex74()
    test_ex81()
    test_ex82()
    test_ex85()
    test_ex93()
    test_ex112()
    test_ex113()
    test_ex121()