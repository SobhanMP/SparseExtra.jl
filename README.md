```@meta
EditURL = "<unknown>/README.jl"
```

![example workflow](https://github.com/sobhanmp/SparseExtra.jl/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/SobhanMP/SparseExtra.jl/branch/main/graph/badge.svg?token=MzXyDmfScn)](https://codecov.io/gh/SobhanMP/SparseExtra.jl)

Collections of mostly sparse stuff developed by me.

# The `iternz` API

This returns an iterator over the structural non-zero elements of the array (elements that aren't zero due to the structure not zero elements) i.e.
```julia
all(iternz(x)) do (v, k...)
    x[k...] == v
end
```

The big idea is to abstract away all of the speciall loops needed to iterate over sparse containers. These include special Linear Algebra matrices like `Diagonal`, and `UpperTriangular` or `SparseMatrixCSC`. Furethemore it's possible to use this recursively i.e. An iteration over a `Diagonal{SparseVector}` will skip the zero elements (if they are not stored) of the SparseVector.

For an example let's take the sum of the elements in a matrix such that `(i + j) % 7 == 0`. The most general way of writing it is

````julia
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
````

````
BenchmarkTools.Trial: 12 samples with 1 evaluation.
 Range (min … max):  449.649 ms … 457.199 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     450.697 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   451.578 ms ±   2.561 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

  █ ▁▁▁ ▁   ▁ ▁  █                                       ▁    ▁  
  █▁███▁█▁▁▁█▁█▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁█ ▁
  450 ms           Histogram: frequency by time          457 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.
````

Now this is pretty bad, we can improve the performance by using the sparse structure of the problem

````julia
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
````

````
sparse_only (generic function with 1 method)
````

We can test for correctness

````julia
sparse_only(A) == general(A)
````

````
true
````

and benchmark the function

````julia
@benchmark sparse_only($A)
````

````
BenchmarkTools.Trial: 327 samples with 1 evaluation.
 Range (min … max):  15.153 ms …  23.301 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     15.200 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   15.282 ms ± 640.526 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

  █▅                                                            
  ██▇▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▅ ▆
  15.2 ms       Histogram: log(frequency) by time      19.3 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.
````

we can see that while writing the function requires understanding how CSC matrices are stored, the code is 600x faster. The thing is that this pattern gets repeated everywhere so we might try and abstract it away. My proposition is the iternz api.

````julia
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
````

````
true
````

````julia
@benchmark sparse_only($A)
````

````
BenchmarkTools.Trial: 329 samples with 1 evaluation.
 Range (min … max):  15.168 ms … 15.392 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     15.210 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   15.212 ms ± 27.239 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

     ▂ ▄ ▂  ▆▂▁█▃▅▄▆▆▄▁▇▄▂▄▇ ▁ ▃                               
  ▆▆▇█▆████▆████████████████▆█▅█▄▆▁▆▇▆▄▆▃▄▅▃▁▁▄▁▁▁▁▁▃▁▁▃▁▁▁▁▃ ▅
  15.2 ms         Histogram: frequency by time        15.3 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.
````

The speed is the same as the specialized version but there is no `@inbounds`, no need for ugly loops etc. As a bonus point it works on all of the specialized matrices

````julia
using LinearAlgebra
all(iternz_only(i(A)) ≈ general(i(A)) for i in [Transpose, UpperTriangular, LowerTriangular, Diagonal, Symmetric]) # symmetric changes the order of exection.
````

````
true
````

Since these interfaces are written using the iternz interface themselves, the codes generalize to the cases where these special matrices are combined, removing the need to do these tedious specialization.

For instance the 3 argument dot can be written as

````julia
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
````

Correctness tests

````julia
dot(x, A, y) ≈ iternz_dot(x, A, y) && dot(x, SA, y) ≈ iternz_dot(x, SA, y)
````

````
true
````

Benchmarks

````julia
@benchmark dot($x, $A, $y)
````

````
BenchmarkTools.Trial: 573 samples with 1 evaluation.
 Range (min … max):  8.687 ms …  8.880 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     8.724 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   8.724 ms ± 19.783 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

            ▃  ▃▂▂▁ ▃ ▅▂▆▃▅▂█▂▅▂ ▃                            
  ▄▃▃▄▅▅▆▇▇▇██▇███████████████████▇▅▅▆█▃▄▃▅▁▄▃▂▂▁▃▁▃▂▂▁▂▁▃▂▂ ▄
  8.69 ms        Histogram: frequency by time        8.78 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.
````

````julia
@benchmark iternz_dot($x, $A, $y)
````

````
BenchmarkTools.Trial: 487 samples with 1 evaluation.
 Range (min … max):  10.232 ms … 10.365 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     10.267 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   10.270 ms ± 17.653 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

             ▁  ▅▃▂██▆▅▂▃▃▂▁                                   
  ▃▃▂▃▂▃▅▅▅▄██▇█████████████▅▇▆▇▆▇▄▃▅▃▅▃▄▂▃▁▁▃▃▂▁▃▂▁▂▁▁▁▁▁▂▁▂ ▄
  10.2 ms         Histogram: frequency by time        10.3 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.
````

````julia
@benchmark dot($x, $SA, $y)
````

````
BenchmarkTools.Trial: 9 samples with 1 evaluation.
 Range (min … max):  574.014 ms … 587.621 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     574.127 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   575.833 ms ±   4.445 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

  █                                                              
  █▅▅▁▁▁▅▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▅ ▁
  574 ms           Histogram: frequency by time          588 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.
````

````julia
@benchmark iternz_dot($x, $SA, $y)
````

````
BenchmarkTools.Trial: 486 samples with 1 evaluation.
 Range (min … max):  10.249 ms … 10.934 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     10.288 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   10.294 ms ± 51.773 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

    ▁ ▁ ▂▃▄▃█▅▄                                                
  ▄▃█▇█████████▇▅▆▄▄▅▃▄▃▃▂▃▃▃▂▁▃▁▁▁▂▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▁▂ ▃
  10.2 ms         Histogram: frequency by time        10.5 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.
````

## API:

The Api is pretty simple, the `iternz(A)` should return an iteratable such that
```julia
all(A[ind...] == v for (v, ind...) in iternz(A))
```

If the matrix is a container for a different type, the inner iteration should be done via iternz. This repo provides the `IterateNZ` container whose sole pupose is to hold the array to overload `Base.iterate`. Additionally matrices have the `skip_col` and `skip_row_to` functions defined. The idea that if meaningful, this should return a state such that iterating on that state will give the first element of the next column or in the case of `skip_row_to(cont, state, i)`, iterate should return `(i, j)` where j is the current column.

## TODO
- test with non-one based indexing

# parallel ldiv!

````julia
using SparseExtra, LinearAlgebra, SparseArrays, BenchmarkTools
const C = A + I
const B = Matrix(sprandn(size(C)..., 0.1));
const F = lu(C);
const X = similar(B);
````

Standard:

````julia
@benchmark ldiv!($X, $F, $B)
````

````
BenchmarkTools.Trial: 1 sample with 1 evaluation.
 Single result which took 717.932 s (0.00% GC) to evaluate,
 with a memory estimate of 0 bytes, over 0 allocations.
````

With FLoops.jl:

````julia
@benchmark par_solve!($X, $F, $B)
````

````
BenchmarkTools.Trial: 1 sample with 1 evaluation.
 Single result which took 134.198 s (0.00% GC) to evaluate,
 with a memory estimate of 1.24 MiB, over 163 allocations.
````

with manual loops

````julia
@benchmark par_ldiv!_t($X, $F, $B)
````

````
BenchmarkTools.Trial: 1 sample with 1 evaluation.
 Single result which took 130.826 s (0.00% GC) to evaluate,
 with a memory estimate of 1.24 MiB, over 130 allocations.
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

