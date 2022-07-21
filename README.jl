# ![example workflow](https://github.com/sobhanmp/SparseExtra.jl/actions/workflows/ci.yml/badge.svg)
# [![codecov](https://codecov.io/gh/SobhanMP/SparseExtra.jl/branch/main/graph/badge.svg?token=MzXyDmfScn)](https://codecov.io/gh/SobhanMP/SparseExtra.jl)

# Collections of mostly sparse stuff developed by me.


# # The `iternz` API

# This returns an iterator over the structural non-zero elements of the array (elements that aren't zero due to the structure not zero elements) i.e.
# ```julia
# all(iternz(x)) do (v, k...)
#     x[k...] == v
# end
# ```

# The big idea is to abstract away all of the speciall loops needed to iterate over sparse containers. These include special Linear Algebra matrices like `Diagonal`, and `UpperTriangular` or `SparseMatrixCSC`. Furethemore it's possible to use this recursively i.e. An iteration over a `Diagonal{SparseVector}` will skip the zero elements (if they are not stored) of the SparseVector.



# For an example let's take the sum of the elements in a matrix such that `(i + j) % 7 == 0`. The most general way of writing it is

using BenchmarkTools, SparseArrays
const n = 10_000
const A = sprandn(n, n, max(1000, 0.1 * n*n) / n / n);

function general(x::AbstractMatrix)
    s = zero(eltype(x))
    @inbounds for j in axes(x, 2),
        i in axes(x, 1)
        if (i + j) % 7 == 0
            s += x[i, j]
        end
    end
    return s
end
@benchmark general($A)

# Now this is pretty bad, we can improve the performance by using the sparse structure of the problem

using SparseArrays: getcolptr, nonzeros, rowvals

function sparse_only(x::SparseMatrixCSC)
    s = zero(eltype(x))
    @inbounds for j in axes(x, 2),
        ind in getcolptr(x)[j]:getcolptr(x)[j + 1] - 1

        i = rowvals(x)[ind]
        if (i + j) % 7 == 0
            s += nonzeros(x)[ind]
        end
    end
    return s
end

# We can test for correctness

sparse_only(A) == general(A)

# and benchmark the function

@benchmark sparse_only($A)

# we can see that while writing the function requires understanding how CSC matrices are stored, the code is 600x faster. The thing is that this pattern gets repeated everywhere so we might try and abstract it away. My proposition is the iternz api. 

using SparseExtra

function iternz_only(x::AbstractMatrix)
    s = zero(eltype(x))
    for (v, i, j) in iternz(x)
        if (i + j) % 7 == 0
            s += v
        end
    end
    return s
end
iternz_only(A) == general(A)

#

@benchmark sparse_only($A)

# The speed is the same as the specialized version but there is no `@inbounds`, no need for ugly loops etc. As a bonus point it works on all of the specialized matrices

using LinearAlgebra
all(iternz_only(i(A)) ≈ general(i(A)) for i in [Transpose, UpperTriangular, LowerTriangular, Diagonal, Symmetric]) # symmetric changes the order of exection.

# Since these interfaces are written using the iternz interface themselves, the codes generalize to the cases where these special matrices are combined, removing the need to do these tedious specialization.

# For instance the 3 argument dot can be written as 



function iternz_dot(x::AbstractVector, A::AbstractMatrix, y::AbstractVector)
    (length(x), length(y)) == size(A) || throw(ArgumentError("bad shape"))
    acc = zero(promote_type(eltype(x), eltype(A), eltype(y)))
    @inbounds for (v, i, j) in iternz(A)
        acc += x[i] * v * y[j]
    end
    acc
end


const (x, y) = randn(n), randn(n);
const SA = Symmetric(A);

# Correctness tests

dot(x, A, y) ≈ iternz_dot(x, A, y) && dot(x, SA, y) ≈ iternz_dot(x, SA, y)

# Benchmarks

@benchmark dot($x, $A, $y)

#

@benchmark iternz_dot($x, $A, $y)

#

@benchmark dot($x, $SA, $y)

#

@benchmark iternz_dot($x, $SA, $y)



# ## API:

# The Api is pretty simple, the `iternz(A)` should return an iteratable such that 
# ```julia
# all(A[ind...] == v for (v, ind...) in iternz(A))
# ```
#
# If the matrix is a container for a different type, the inner iteration should be done via iternz. This repo provides the `IterateNZ` container whose sole pupose is to hold the array to overload `Base.iterate`. Additionally matrices have the `skip_col` and `skip_row_to` functions defined. The idea that if meaningful, this should return a state such that iterating on that state will give the first element of the next column or in the case of `skip_row_to(cont, state, i)`, iterate should return `(i, j)` where j is the current column.



# ## TODO
# - test with non-one based indexing



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
