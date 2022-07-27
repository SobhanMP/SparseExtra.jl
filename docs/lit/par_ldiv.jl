# # parallel ldiv!

using SparseExtra, LinearAlgebra, SparseArrays, BenchmarkTools
const n = 10_000
const A = sprandn(n, n, 5 / n);
const C = A + I
const B = Matrix(sprandn(n, n, 1 / n));
const F = lu(C);
const X = similar(B);

# Standard:

@benchmark ldiv!($X, $F, $B)

# With FLoops.jl:

@benchmark par_solve!($X, $F, $B)

# with manual loops

@benchmark par_ldiv!_t($X, $F, $B)
