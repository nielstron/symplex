from typing import *
from sympy import Matrix, Rational, symbols, RealNumber
from verification import *


def pertubation_vector(perm: Iterable[int], eps):
    """
    Provided a sufficiently small epsilon returns a pertubation vector
    s.t. (A, b+e(\epsilon)) is in general position
    :param: permutation of m constraints, usually `range(m)`
    :param eps:
    :return: e(\epsilon)
    """
    return Matrix([eps**(i+1) for i in perm])


def perturbed_polygon(A: Matrix, b: Matrix, eps=Rational(1, 128)):
    m = A.shape[0]
    return A, b+pertubation_vector(range(m), eps)


def v_star_from_perturbed_polygon(A: Matrix, b: Matrix, A_pert: Matrix, b_pert: Matrix, v_pert: Matrix):
    I_pert = active_constraints(v_pert, A_pert, b_pert)
    v_star = (sub_matrix(A, I_pert)**-1)*sub_matrix(b, I_pert)
    return v_star

