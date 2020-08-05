from sympy import Matrix, Rational, symbols
from simplex import is_generic


def make_generic_explicit_perturb(A: Matrix, b: Matrix, force=False):
    m, n = A.shape
    if not force and is_generic(A, b):
        return A, b, m*[0]
    eps = symbols("\epsilon")
    eps_sub = Rational(1, 64)
    e = Matrix([eps**(i+1) for i in range(m)])
    while not is_generic(A, b+(e.subs(eps, eps_sub))):
        eps_sub /= 2
    return A, b + (e.subs(eps, eps_sub)), e.subs(eps, eps_sub)
