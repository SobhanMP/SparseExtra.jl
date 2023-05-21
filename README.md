[![docs](https://img.shields.io/badge/docs-blue.svg)](https://sobhanmp.github.io/SparseExtra.jl/)
![tests badge](https://github.com/sobhanmp/SparseExtra.jl/actions/workflows/ci.yml/badge.svg)
[![codecov badge](https://codecov.io/gh/SobhanMP/SparseExtra.jl/branch/main/graph/badge.svg?token=MzXyDmfScn)](https://codecov.io/gh/SobhanMP/SparseExtra.jl)

Collections of mostly sparse stuff developed by me. See documentation. But big picture, there is a very fast dijkstra implementation that uses sparse matrices as graph representation, parallel LU solve (i.e., Ax=b), and iternz an API for iterating over sparse structures like SparseMatrixCSC, Diagonal, bidiagonal that is composable meaning that if you have a new vector type, implementing the iternz will make it work with other sparse object that already use it.

Most of this was developed in the context of this [paper](https://arxiv.org/abs/2210.14351) which is something to keep in mind when testing dense sparse matrices. Traffic graphs usually have less than 6 neighbors and are very balanced.

