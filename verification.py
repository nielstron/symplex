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

