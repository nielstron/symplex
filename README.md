Symplex
-------

> Symbolic simplex for educational purposes

This repo contains a number of functions and scripts that
implement the simplex algorithm and various helper methods.

All computations are done symbolically and intermediate results in each step
are stored explicitely to facilitate investigating the procedure i.e. via the debug view.
The focus of this project is to be educational, not to be fast or efficient.

The content is heavily based on the course "Linear and Convex Optimization" held by Prof. Dr. Peter Gritzmann in the summer term 2020 at the Technical University Munich.
References to course content may occur.

#### Usage

Note that the only way to specify a linear problem is in natural form,
that is supplying a matrix _A_ and vectors _b_ and _c_ such that
the argument to _min c^T\*x , A*x <= b_ is searched.

A number of example usages of the code is contained in `symples/test.py`.
It is at the same time a very small test suite for this package.
