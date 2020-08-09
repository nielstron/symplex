from sympy import pprint
import sympy as sy
from random import randint

# naming convention as in HE 11.2
# this also implies, that the positivity constraints are not reflected in the matrix A, nor in the
# vector c
# only difference is that we start counting our basis and constraints from 0

def unbounded(d0, d_lower):
    for i in range(d_lower.shape[1]):
        if (d0[i] < 0 and all([x <= 0 for x in d_lower[:, i]])):
            return True

    return False

def get_table(A, B: list, c):
    # filter entries
    A_B = A[:, B]
    A_B_inv = A_B.inv()
    c_B = c[B, :]

    # lower part of table
    d_lower = A_B_inv*A

    # first row without the first entry
    d0 = c.transpose() - c_B.transpose()*d_lower

    v = A_B_inv * b

    # upper left corner
    ulc = -c_B.transpose()*A_B_inv*b

    return ulc, v, d0, d_lower

def get_pivot(l, val, pivot_lowest = True):
    if (pivot_lowest):
        return l.index(val)
    else:
        return len(l) - 1 - list(reversed(l)).index(val)

def do_simplex_tableau(A, b, c, B: set, print_steps = True, pivot_lowest = True, initial = False):
    # returns the basis of the optimal vertex and the optimal vertex if any
    # we assume that we are always feasible, which is guarantied if we are called like below
    # we have the parameter 'initial' because it can happen, that we have an artificial variable in
    # a basis at the end with value 0 and we cannot construct a basis in the original problem from
    # this

    # the whole set - list thing is a bit ugly, but it works. The reason to use lists is to fix an order
    # the reason for sets are the set operations

    I = set(range(A.shape[1]))

    N = I - B

    B_list = sorted(list(B))
    N_list = sorted(list(N))

    cur, v, d0, d_lower = get_table(A, B_list, c)

    if (print_steps):
        print("Basis:", B_list)
        print("Table:")
        pprint(cur.row_join(d0).col_join(v.row_join(d_lower)))
        print("")

    # optimality criteria
    while (any([x < 0 for x in d0[:, list(N)]])):
        if (unbounded(d0, d_lower)):
            # we are unbounded
            return None, None
        # columns
        C = [j for j in N_list if (d0[j] < 0 and any([x > 0 for x in d_lower[:, j]]))]
        #print("C", C)

        # choose either lowest or highest index
        j = min(C) if pivot_lowest else max(C)

        # rows
        # we need R to be sorted
        R = [i for i in range(A.shape[0]) if d_lower[i, j] > 0]
        #print("R", R)
        lambdas = [v[i]/d_lower[i, j] for i in R]
        lamb = min(lambdas)

        # choose either lowest or highest index
        r = get_pivot(lambdas, lamb, pivot_lowest)

        # convert this to the actual index in the basis
        r = B_list[R[r]]

        # form new basis
        B = (B - {r}) | {j}
        N = I - B

        B_list = sorted(list(B))
        N_list = sorted(list(N))

        cur, v, d0, d_lower = get_table(A, B_list, c)

        if (print_steps):
            print("Out index:", r)
            print("In index:", j)
            print("New basis:", B_list)
            print("Table:")
            pprint(cur.row_join(d0).col_join(v.row_join(d_lower)))
            print("")

    # if we are in the initial phase, we have m variables more that should not be in a base with
    # value 0. This can happen, if the value of the artificial variable is zero.
    if (initial):
        artificial_in_base_zero = [B_list[i] for i in range(len(B_list))
                                                 if (v[i] == 0 and B_list[i] > n-m)]
        while (len(artificial_in_base_zero) > 0):
            if (print_steps):
                print("Artificial variable in optimal base of initial value problem")
            # we fix this by swapping with another variable that then also is set to zero
            r = artificial_in_base_zero[0]

            non_zero = [d_lower[B_list.index(r), i] != 0 for i in N_list]
            assert(any(non_zero)), "Redundancy detected, should not be possible?"
            j = get_pivot(non_zero, True, pivot_lowest)
            B = (B - {r}) | {j}
            N = I - B

            B_list = sorted(list(B))
            N_list = sorted(list(N))

            cur, v, d0, d_lower = get_table(A, B_list, c)
            if (print_steps):
                print("Out index:", r)
                print("In index:", j)
                print("New basis:", B_list)
                print("Table:")
                pprint(cur.row_join(d0).col_join(v.row_join(d_lower)))
                print("")
            artificial_in_base_zero = [B_list[i] for i in range(len(B_list))
                                                     if (v[i] == 0 and B_list[i] > n-m)]

    ret_v = sy.Matrix([0 if (i not in B) else v[B_list.index(i)] for i in range(A.shape[1])])
    return ret_v, B

def solve_with_tableau(A, b, c, n, m, print_steps = True, pivot_lowest = True):
    # construct first phase problem matrix/vectors
    # insert artificial variables
    A_ = A.row_join(sy.eye(m))

    # we try to minimize the sum of our artificial variables to get them to zero
    c_ = sy.zeros(n + m, 1)
    c_[-m:, :] = sy.ones(m, 1)

    # b does not change
    b_ = b

    # trivially feasible solution, not really necessary as we work in the basis indices
    v_ = sy.zeros(n + m, 1)
    v_[-m:, :] = b

    # basis for trivially feasible solution
    B_ = set(range(n, n + m))

    assert(A_ * v_ == b_), "Trivially feasible solution is not feasible."
    #assert(A[:, B_].inv()*b == v_ ), "Basis does not correspond to solution."

    if (print_steps):
        print("Solving initial value problem")

    # get starting vertex as solution of initial value problem
    v, B = do_simplex_tableau(A_, b_, c_, B_, print_steps, pivot_lowest, True)

    if (any([x != 0 for x in v[n:]])):
        if (print_steps):
            print("Unable to find starting vertex, problem is unfeasible.")
        return None
    else:
        if (print_steps):
            print("Found initial vertex with basis:", B)

        # use initial vertex to compute solution
        v, B = do_simplex_tableau(A, b, c, B, print_steps, pivot_lowest, False)

        if (v == None):
            if (print_steps):
                print("Problem is unbounded.")
            return None
        else:
            if (print_steps):
                print("Optimal vertex found:")
                pprint(v)
                print("Corresponding basis:", list(B))
                print("Objective function value:")
                pprint(sy.flatten(c.transpose()*v)[0])
            return sy.flatten(c.transpose()*v)[0]

# test values
c = sy.Matrix([15, 0, 6, 17])
n = c.shape[0]

b = sy.Matrix([5, 6])
m = b.shape[0]

A = sy.Matrix([[3, 3, 2, 4],
               [5, -5, 1, 5]])

assert(A.shape == (m, n)), "Internal matrix construction failed unexpectedly."
assert(A.rank() == m), ("The given constraint matrix cannot include linearly dependent constraints - "
                      "remove them to use this method")
assert(m <= n), "Cannot have more constraints than variables if they are linearly independent."

solve_with_tableau(A, b, c, n, m, True, True)

# the rest was used for testing/finding bugs
#for i in range(10000):
#    n = 2
#    m = 1
#
#    c = [randint(-10, 10) for j in range(n)]
#
#    c = sy.Matrix(c)
#
#    b = [randint(-10, 10) for j in range(m)]
#
#    b = sy.Matrix(b)
#
#    A = [[randint(-10, 10) for k in range(n)] for j in range(m)]
#
#    A = sy.Matrix(A)
#
#    assert(A.shape == (m, n))
#
#    if (A.rank() != m):# or all([x == 0 for x in c])):
#        print("continue")
#        continue
#    pprint(A)
#    pprint(b)
#    pprint(c)
#
#    low = solve_with_tableau(A, b, c, n, m, False, True)
#    high = solve_with_tableau(A, b, c, n, m, False, False)
#
#    if (low != high):
#        print("Found diverging solutions!")
#        pprint(A)
#        pprint(b)
#        pprint(c)
#        input("Lowest")
#        solve_with_tableau(A, b, c, n, m, True, True)
#        input("Highest")
#        solve_with_tableau(A, b, c, n, m, True, False)
#        input()
#
