from sympy import Matrix, BlockMatrix
from sympy.matrices.common import NonInvertibleMatrixError
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


def vertices(A: Matrix, b: Matrix):
    m, n = A.shape
    vs = set()
    for potential_basis in combinations(range(m), n):
        A_B = sub_matrix(A, potential_basis)
        if A_B.rank() == n:
            v: Matrix = (A_B**-1)*sub_matrix(b, potential_basis)
            if is_contained(v, A, b):
                vs.add(v.as_immutable())
    return vs


def is_basis(v: Matrix, A: Matrix, b: Matrix, B: Set[int]):
    A_B = Matrix([A[i,:] for i in B])
    b_B = Matrix([b[i] for i in B])
    if A_B.rank() != v.shape[0]:
        print("Rank of A is too low")
        return False
    if (A_B**-1)*b_B != v:
        print("A is not active in all of v")
        return False
    return True


def is_contained(v: Matrix, A: Matrix, b:Matrix):
    m = b.shape[0]
    for i in range(m):
        if (A[i,:]*v)[0] > b[i]:
            print(f"Point not in Polygon, constraint {i} violated")
            return False
    return True


def is_vertex(v: Matrix, A:Matrix, b: Matrix):
    return len(bases(v, A, b)) > 0 and is_contained(v, A, b)


def is_feasible_eq(A: Matrix, b: Matrix):
    """
    Checks feasibility of Ax = b by comparing ranks of A and (A,b)
    """
    A_b: Matrix = BlockMatrix([A, b]).as_explicit()
    A_r = A.rank()
    A_b_r = A_b.rank()
    if A_r == A_b_r:
        return True
    return False


def is_generic(A: Matrix, b: Matrix):
    """
    Check genericity by rank comparison
    cf Theorem 3.4.5
    """
    m, n = A.shape
    for I in combinations(range(m), n+1):
        A_I = sub_matrix(A, I)
        b_I = sub_matrix(b, I)
        if is_feasible_eq(A_I, b_I):
            print(f"Ax <= b is not generic, A_Ix = b_I is feasible for I={I}")
            return False
    print("Ax <= b is generic by rank comparison")
    return True
