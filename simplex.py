from sympy import Matrix, pretty
from typing import *
from enum import Enum
from math import inf

from verification import *
from utils import *


class SimplexResult(Enum):
    OPTIMAL = "optimal"
    UNBOUNDED = "unbounded"
    INFEASIBLE = "infeasible"
    INVALID = "invalid"

class PivotRule(Enum):
    MINIMAL = lambda x: x[0]
    MAXIMAL = lambda x: x[-1]


def simplex(A: Matrix, b: Matrix, c: Matrix, v: Matrix, B: Set[int], pivot_rule=PivotRule.MINIMAL):
    """
    :param A:
    :param b:
    :param c:
    :param v:
    :param B: Basis (!) 0 indexed
    :return:
    """
    res = None
    opt_val = None
    v_star = None
    m, n = A.shape
    if not is_contained(v, A, b):
        res = SimplexResult.INVALID
    if not is_basis(v, A, b, B):
        print(f"{B} is not a valid Basis of {v}")
        res = SimplexResult.INVALID
    iteration = -1
    visited_bases = {frozenset(B)}
    while res is None:
        iteration += 1
        print(f"Iteration {iteration}")
        print(f"v_{iteration}:")
        print(pretty(v))
        print(f"B = {B}")

        N = set(range(m)) - B
        print(f"N = {N}")
        AB = sub_matrix(A, sorted(list(B)))
        print("A_B:")
        print(pretty(AB))
        mABm1 = -AB**-1
        print("-A_B^-1:")
        print(pretty(mABm1))
        s = [mABm1[:,i] for i in range(n)]
        print("s:")
        print(pretty(s))

        mABm1_mulc = mABm1.transpose() * c
        print("-A_B^-1*c:")
        print(pretty(mABm1_mulc))
        if all(e <= 0 for e in mABm1_mulc[:]): # equivalent: all(c.transpose()*s[j] <= 0 for j in range(n)):
            print("v optimal")
            # print some properties
            if all(e < 0 for e in mABm1_mulc[:]):
                print(f"v is the unique optimum")
            res = SimplexResult.OPTIMAL
            v_star = v
            opt_val = (c.transpose()*v)[0]
        else:
            valid_p = [p for p in range(n) if (c.transpose()*s[p])[0] > 0]
            print(f"valid_p = {valid_p}")
            p = pivot_rule(valid_p)
            print(f"p = {p}")
            R = [i for i in N if (A[i,:]*s[p])[0] > 0]
            print(f"R = {R}")
            if len(R) == 0:
                print("\phi unbounded from above on P")
                res = SimplexResult.UNBOUNDED
                opt_val = inf
            else:
                step_sizes = [(b[i] - (A[i,:]*v)[0])/(A[i,:]*s[p])[0] for i in R]
                print(f"step_sizes = {step_sizes}")
                lam = min(step_sizes)
                print(f"lam = {lam}")
                i_in_candidates = [i for i, s in zip(R, step_sizes) if s == lam]
                print(f"i_in candidates = {i_in_candidates}")
                i_in = pivot_rule(i_in_candidates)
                print(f"i_in = {i_in}")
                i_out = sorted(list(B))[p]
                print(f"i_out = {i_out}")
                B = B - {i_out} | {i_in}
                v = v + lam*s[p]
                if B in visited_bases:
                    print("Basis visited second time, detecting cycle and abort")
                    res = SimplexResult.INVALID
                visited_bases.add(frozenset(B))
    return res, v_star, opt_val


def initial_vertex_polygon(A: Matrix, b: Matrix, I: Set[int]):
    """
    cf Lemma 3.2.5
    """
    m, n = A.shape
    assert len(I) == n
    A_I = sub_matrix(A, I)
    b_I = sub_matrix(b, I)
    v = (A_I**-1)*b_I
    J = set(i for i in range(m) if (A[i,:]*v)[0] > b[i])
    k = len(J)
    A_entries = []
    k_zeroes = k*[0]
    n_zeroes = n*[0]
    for i in set(range(m)) - J:
        A_i = list(A[i,:]) + k_zeroes
        A_entries.append(A_i)
    for i,j in enumerate(J):
        k_spec = k_zeroes.copy()
        k_spec[i] = 1
        A_i = list(A[j,:]) + k_spec
        A_entries.append(A_i)
    for i in range(len(J)):
        k_spec = k_zeroes.copy()
        k_spec[i] = 1
        A_i = n_zeroes.copy() + k_spec
        A_entries.append(A_i)
    A_p = Matrix(A_entries)

    b_entries = list(b[i] for i in set(range(m)) - J)
    b_entries.extend(list(b[i] for i in J))
    b_entries.extend(k_zeroes)
    b_p = Matrix(b_entries)

    z_0 = list(v)
    z_0.extend(b[i] - (A[i,:]*v)[0] for i in J)
    z_0 = Matrix(z_0)

    c = n_zeroes + k*[1]
    c = Matrix(c)

    return A_p, b_p, c, z_0


def determine_feasible_vertex(A: Matrix, b: Matrix, I: Set[int], pivot_rule=PivotRule.MINIMAL):
    n = A.shape[1]
    A_init, b_init, c_init, v_init = initial_vertex_polygon(A, b, I)
    B_init = next(iter(bases(v_init, A_init, b_init)))
    r_init, v, opt_val = simplex(A_init, b_init, c_init, v_init, set(B_init), pivot_rule=pivot_rule)
    if opt_val is None or opt_val < 0:
        print("Problem is infeasible")
        return None
    return Matrix(v[:n])


def initial_vertex_polygon2(A: Matrix, b: Matrix):
    """
    cf ex 8.1
    In addition, we restrict xn+1 to be smeq 1
    s.t. we directly obtain a feasoble solution of P as optimal vertex of P'
    """
    m, n = A.shape
    A_entries = []
    for i in range(m):
        A_i = list(A[i,:]) + [-b[i]]
        A_entries.append(A_i)
    A_entries.append(n*[0]+[-1])
    A_entries.append(n*[0]+[1])
    A_p = Matrix(A_entries)

    b_p = Matrix((m+1)*[0] + [1])

    c_p = Matrix(n*[0] + [1])
    v_0 = Matrix((n+1)*[0])
    return A_p, b_p, c_p, v_0


def determine_feasible_vertex2(A: Matrix, b: Matrix, pivot_rule: PivotRule.MINIMAL):
    m, n = A.shape
    A_init, b_init, c_init, v_0 = initial_vertex_polygon2(A, b)
    B_init = next(iter(bases(v_0, A_init, b_init)))
    res, v_init, opt_val = simplex(A_init, b_init, c_init, v_0, set(B_init), pivot_rule=pivot_rule)
    if opt_val is None or opt_val < 1:
        print("Polygon is infeasible")
        return None
    return Matrix(v_init[:n])


def simplex_full(A: Matrix, b: Matrix, c: Matrix, pivot_rule = PivotRule.MINIMAL):
    n = A.shape[1]
    I = set(range(n))
    v = determine_feasible_vertex(A, b, I, pivot_rule=pivot_rule)
    if v is None:
        return SimplexResult.INFEASIBLE, None, None
    B = next(iter(bases(v, A, b)))
    return simplex(A, b, c, v, set(B))


def simplex_tableau(A: Matrix, b: Matrix, c: Matrix, B: Set[int]):
    m, n = A.shape
    c_B = sub_matrix(c, B)
    A_B = sub_matrix(A, B)
    A_Bm1 = A_B**-1
    A_Bm1A = A_Bm1*A
    A_0 = list(-c_B.transpose()*A_Bm1*b)
    A_0.extend(list(c.transpose() - c_B.transpose()*A_Bm1*A))
    A_rem = []
    for i in range(m):
        A_i = [(A_Bm1[i, :]*b)[0]]
        A_i.extend(list(A_Bm1A[i, :]))
        A_rem.append(A_i)
    tableau = Matrix([A_0, *A_rem])
    N = set(range(n)) - B
    if all(tableau[0, j] >= 0 for j in N):
        print("v is optimal")
        return
    if any(tableau[0,j] < 0 and all(tableau[i,j] <= 0 for i in range(m)) for j in N):
        print("Problem is unbounded")
        return


def is_generic2(A: Matrix, b: Matrix):
    """
    Check genericity by checking for feasibility of sub polygons
    following the definition directly
    """
    m, n = A.shape
    for I in combinations(range(m), n+1):
        # check if A_I*x = b_I is infeasible
        # note the equality, making the use of block matrices useful
        res, _, _ = simplex_full(
            BlockMatrix([[sub_matrix(A, I)], [-sub_matrix(A, I)]]).as_explicit(),
            BlockMatrix([[sub_matrix(b, I)], [-sub_matrix(b, I)]]).as_explicit(),
            Matrix(n*[0])
        )
        if res != SimplexResult.INFEASIBLE:
            print(f"Matrix A is not generic, index set {I} has result {res}")
            return False
    print("Matrix A is generic")
    return True
