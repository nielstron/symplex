from sympy import Matrix, pretty
from typing import *
from enum import Enum
from math import inf

from verfication import *


class SimplexResult(Enum):
    OPTIMAL = "optimal"
    UNBOUNDED = "unbounded"
    INVALID = "invalid"


def simplex(A: Matrix, b: Matrix, c: Matrix, v: Matrix, B: Set[int]):
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
    m = b.shape[0]
    if not is_contained(v, A, b):
        res = SimplexResult.INVALID
    if not is_basis(v, A, b, B):
        print(f"{B} is not a valid Basis of {v}")
        res = SimplexResult.INVALID
    n = c.shape[0]
    iteration = -1
    while res is None:
        iteration += 1
        print(f"Iteration {iteration}")
        print(f"v_{iteration}:")
        print(pretty(v))
        print(f"B = {B}")
        N = set(range(m)) - B
        print(f"N = {N}")
        AB = Matrix([A[i,:] for i in sorted(list(B))])
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
            # minimal index rule
            p = valid_p[0]
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
                # again minimal index rule
                i_in = i_in_candidates[0]
                print(f"i_in = {i_in}")
                i_out = sorted(list(B))[p]
                print(f"i_out = {i_out}")
                B = B - {i_out} | {i_in}
                v = v + lam*s[p]
    return res, v_star, opt_val


def initial_vertex_polygon(A: Matrix, b: Matrix, I: Set[int]):
    n = A.shape[1]
    m = A.shape[0]
    assert len(I) == n
    A_I = Matrix([A[i,:] for i in I])
    b_I = Matrix([b[i] for i in I])
    v = (A_I**-1)*b_I
    J = set(i for i in range(m) if (A[i,:]*v)[0] > b[i])
    k = len(J)
    A_entries = []
    k_zeroes = k*[0]
    n_zeroes = n*[0]
    for i in set(range(m)) - J:
        A_i = list(A[i,:])
        A_i.extend(k_zeroes)
        A_entries.append(A_i)
    for i,j in enumerate(J):
        k_spec = k_zeroes.copy()
        k_spec[i] = 1
        A_i = list(A[j,:])
        A_i.extend(k_spec)
        A_entries.append(A_i)
    for i in range(len(J)):
        k_spec = k_zeroes.copy()
        k_spec[i] = 1
        A_i = n_zeroes.copy()
        A_i.extend(k_spec)
        A_entries.append(A_i)
    A_p = Matrix(A_entries)
    b_entries = list(b[i] for i in set(range(m)) - J)
    b_entries.extend(list(b[i] for i in J))
    b_entries.extend(k_zeroes)
    b_p = Matrix(b_entries)

    z_0 = list(v)
    z_0.extend(b[i] - (A[i,:]*v)[0] for i in J)
    z_0 = Matrix(z_0)

    c = n_zeroes
    c.extend(k*[1])
    c = Matrix(c)

    return A_p, b_p, c, z_0


def simplex_full(A: Matrix, b: Matrix, c: Matrix):
    n = A.shape[1]
    I = set(range(n))
    A_init, b_init, c_init, v_init = initial_vertex_polygon(A, b, I)
    B_init = next(iter(bases(v_init, A_init, b_init)))
    r_init, v, opt_val = simplex(A_init, b_init, c_init, v_init, set(B_init))
    if opt_val is None or opt_val < 0:
        print("Problem is infeasible")
    B = next(iter(bases(v, A, b)))
    return simplex(A, b, c, v, set(B))
