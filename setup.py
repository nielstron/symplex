#!/usr/bin/env python
# coding=utf-8
from setuptools import setup, find_packages

from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

short_description = "{}".format(
    "Symbolic simplex for educational purposes"
)

setup(
    name="symplex",
    version="0.0.1",
    description=short_description,
    author="nielstron",
    author_email="n.muendler@web.de",
    url="https://github.com/nielstron/symplex/",
    py_modules=["symplex"],
    packages=find_packages(),
    install_requires=["sympy"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
    keywords="python symbolic simplex linear optimization",
    python_requires=">=3",
)
