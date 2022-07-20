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
const A = sprandn(n, n, min(1000, 0.1 * n*n) / n / n);

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
BenchmarkTools.Trial: 97 samples with 1 evaluation.
 Range (min … max):  49.724 ms …  54.943 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     51.749 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   51.740 ms ± 780.160 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

                              ▂        █    ▄  ▂   ▂   ▂        
  ▄▁▄▁▄▁▁▄▁▁▁▁▄▁▆▁▁▁▄▄▆▄▁▆▄█▆████▁▆█▄▆▆█▄█▆▆█▆▄█▄█▄█▄▆███▆▁▁▄▄ ▁
  49.7 ms         Histogram: frequency by time           53 ms <

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
BenchmarkTools.Trial: 10000 samples with 6 evaluations.
 Range (min … max):  5.410 μs …  12.438 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     5.441 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   5.491 μs ± 254.516 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

  ▅█▆▃▂▂▂▁▁▁▁▁▁  ▁                                            ▁
  ██████████████████▇█▇▇▇▇█▇▆▆▇▇▆▅▆▆▅▄▅▅▄▅▄▅▄▅▄▅▄▃▁▃▄▄▃▄▁▁▄▁▅ █
  5.41 μs      Histogram: log(frequency) by time      6.41 μs <

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
BenchmarkTools.Trial: 10000 samples with 6 evaluations.
 Range (min … max):  5.402 μs …  27.686 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     5.461 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   5.641 μs ± 630.061 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

  ██▅▅▄▄▄▃▃▂▂▁▁▁ ▁                                            ▂
  ██████████████████▇▇▇▆▆▆▇▆▆▆▆▅▅▆▅▆▆▅▄▅▆▆▅▆▅▄▅▅▄▅▅▅▄▅▅▅▅▅▄▅▅ █
  5.4 μs       Histogram: log(frequency) by time      7.99 μs <

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
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  13.935 μs … 152.533 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     14.697 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   15.067 μs ±   2.208 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

     ▆█▄                                                        
  ▂▃▇███▇▅▅▄▄▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▂ ▃
  13.9 μs         Histogram: frequency by time         23.1 μs <

 Memory estimate: 0 bytes, allocs estimate: 0.
````

````julia
@benchmark iternz_dot($x, $A, $y)
````

````
BenchmarkTools.Trial: 10000 samples with 7 evaluations.
 Range (min … max):  4.220 μs … 20.442 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     5.513 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   6.605 μs ±  2.228 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

         ▄▇█▅▁                                           ▂▃   
  ▂▂▂▂▃▅▇█████▇▆▄▄▄▄▄▄▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▂▂▃▂▂▂▂▂▂▂▂▂▂██▅ ▃
  4.22 μs        Histogram: frequency by time        11.2 μs <

 Memory estimate: 0 bytes, allocs estimate: 0.
````

````julia
@benchmark dot($x, $SA, $y)
````

````
BenchmarkTools.Trial: 111 samples with 1 evaluation.
 Range (min … max):  44.615 ms …  48.224 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     44.947 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   45.301 ms ± 717.616 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

    ▅▆█                                                         
  █████▇▇▅▄▃▄▁▄▄▄▄▁▇▃▁▄▄▁▄▅▄▅▃▃▄▃▃▄▃▁▃▁▃▁▃▁▁▁▄▁▁▁▁▁▁▃▁▃▃▁▁▁▁▁▃ ▃
  44.6 ms         Histogram: frequency by time         47.5 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.
````

````julia
@benchmark iternz_dot($x, $SA, $y)
````

````
BenchmarkTools.Trial: 10000 samples with 7 evaluations.
 Range (min … max):  4.361 μs …  22.740 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     4.421 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   4.475 μs ± 431.840 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

   ▃██▇▅▄▃▂▂▁▂▁▂▃▂▁                                           ▂
  █████████████████████▇▇▇▇▆▇▅▆▅▆▄▅▅▅▅▄▄▁▄▃▄▃▄▄▃▃▁▁▄▄▃▃▅▄▄▁▄▄ █
  4.36 μs      Histogram: log(frequency) by time      5.34 μs <

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

#With FLoops.jl:

@benchmark par_solve!($X, $F, $B)
````

````
BenchmarkTools.Trial: 4 samples with 1 evaluation.
 Range (min … max):  1.269 s …    1.497 s  ┊ GC (min … max): 0.35% … 0.35%
 Time  (median):     1.393 s               ┊ GC (median):    0.35%
 Time  (mean ± σ):   1.388 s ± 111.948 ms  ┊ GC (mean ± σ):  0.33% ± 0.10%

  █           █                                     █      █  
  █▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁█ ▁
  1.27 s         Histogram: frequency by time          1.5 s <

 Memory estimate: 4.47 GiB, allocs estimate: 20001.
````

with manual loops

````julia
@benchmark par_ldiv!_t($X, $F, $B)
````

````
BenchmarkTools.Trial: 5 samples with 1 evaluation.
 Range (min … max):  1.171 s …   1.240 s  ┊ GC (min … max): 0.22% … 0.31%
 Time  (median):     1.199 s              ┊ GC (median):    0.29%
 Time  (mean ± σ):   1.200 s ± 29.915 ms  ┊ GC (mean ± σ):  0.28% ± 0.04%

  ██                     █               █                █  
  ██▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  1.17 s         Histogram: frequency by time        1.24 s <

 Memory estimate: 4.47 GiB, allocs estimate: 20006.
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

