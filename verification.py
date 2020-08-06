from sympy.matrices import *
from typing import Set

from utils import *


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
    return len(bases(v, A, b)) > 0


def is_feasible_eq(A: Matrix, b: Matrix):
    A_b: Matrix = BlockMatrix([A, b]).as_explicit()
    A_r = A.rank()
    A_b_r = A_b.rank()
    if A_r == A_b_r:
        print(f"Ax = b is feasible, A, (A,b) have same ranks {A_r} and {A_b_r}")
        return True
    print(f"Ax = b infeasible, A, (A,b) have different ranks {A_r} and {A_b_r}")
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
            print("Ax <= b is not generic")
            return False
    print("Ax <= b is generic")
    return True
