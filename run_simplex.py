from simplex import *
from verfication import *

# exercise 7.4
A = Matrix([[-3,1],[-2,1],[-1,1],[6,1],[0,-1]])
b = Matrix([0, 1, 3, 24, 0])
c = Matrix([1,1])
simplex_full(A, b, c)
