from sympy import Matrix
from itertools import combinations
from typing import *


def active_constraints(v: Matrix, A: Matrix, b: Matrix):
    I = []
    m = b.shape[0]
    for i in range(m):
        if (A[i,:]*v)[0] == b[i]:
            I.append(i)
    return I


def bases(v: Matrix, A: Matrix, b: Matrix):
    n = v.shape[0]
    Bs = []
    for potential_basis in combinations(active_constraints(v, A, b), n):
        if sub_matrix(A, potential_basis).rank() == n:
            Bs.append(potential_basis)
    return Bs


def sub_matrix(A: Matrix, I: Iterable[int]):
    return Matrix([A[i, :] for i in I])
