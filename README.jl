# ![example workflow](https://github.com/sobhanmp/SparseExtra.jl/actions/workflows/ci.yml/badge.svg)
# [![codecov](https://codecov.io/gh/SobhanMP/SparseExtra.jl/branch/main/graph/badge.svg?token=MzXyDmfScn)](https://codecov.io/gh/SobhanMP/SparseExtra.jl)

# Collections of mostly sparse stuff developed by me.




# # parallel ldiv!
using SparseExtra, LinearAlgebra, SparseArrays, BenchmarkTools
const C = A + I
const B = Matrix(sprandn(size(C)..., 0.1));
const F = lu(C);
const X = similar(B);

# Standard:

@benchmark ldiv!($X, $F, $B)

# With FLoops.jl:

@benchmark par_solve!($X, $F, $B)

# with manual loops

@benchmark par_ldiv!_t($X, $F, $B)
