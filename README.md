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
BenchmarkTools.Trial: 96 samples with 1 evaluation.
 Range (min … max):  51.085 ms …  53.858 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     51.938 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   52.085 ms ± 743.745 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

     ▆ █▂ ▂      ▂  ▂      ▂   ▂                                
  ██▄█▆██▆█▄▄▄▄▁██▄▄█▆▄▁▄▆▄█▁▆▄█▆▆▄▁█▁▄▄▆▄▄▆▆▁▆▁▆▁▄▄▄▁▁▄▁▁▄▁▁▄ ▁
  51.1 ms         Histogram: frequency by time         53.8 ms <

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
 Range (min … max):  5.263 μs … 14.149 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     5.394 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   5.702 μs ±  1.018 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

  █▇▄▃▅▃▃▃▃▂▂▂▁▁                                             ▂
  ███████████████▇▆▇▇▇▆▆▆▇▅▆▅▆▆▅▅▄▅▅▄▄▅▄▄▁▄▆▆▆▇▇▇▇█▇▇████▇▇▇ █
  5.26 μs      Histogram: log(frequency) by time     10.3 μs <

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
 Range (min … max):  5.360 μs …  12.114 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     5.478 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   5.685 μs ± 572.886 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

  █▅▄▃▃▅▄▄▃▃▃▃▂▂▂ ▁                                           ▁
  ████████████████████▇▇▇▇▆▆▆▆▆▅▆▆▅▅▆▅▅▅▇▅▆▆▆▆▅▅▆▆▅▅▆▅▅▆▄▄▄▄▄ █
  5.36 μs      Histogram: log(frequency) by time      8.28 μs <

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
 Range (min … max):  13.256 μs … 45.260 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     14.500 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   15.210 μs ±  2.405 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

   ▁▆███▇▆▅▅▄▃▂▂▂▁▁▁▁ ▁ ▁▁                                    ▂
  ▃████████████████████████▇▇▆▆▆▆▆▅▆▆▅▇▆▆▇▆▇▇▇▇▇▅▇▇▆▆▆▇▆▅▅▅▄▅ █
  13.3 μs      Histogram: log(frequency) by time      26.4 μs <

 Memory estimate: 0 bytes, allocs estimate: 0.
````

````julia
@benchmark iternz_dot($x, $A, $y)
````

````
BenchmarkTools.Trial: 10000 samples with 8 evaluations.
 Range (min … max):  2.916 μs … 11.866 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     3.336 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   3.717 μs ±  1.596 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

  ▅▅▆█▄▂▂▁▁▁                                              ▂▃ ▂
  ████████████▇▇▇▆▇▅▆▆▆▇▆▄▃▁▁▁▁▁▁▁▁▁▁▃▁▁▁▁▁▁▁▁▁▁▁▆▇▅▃▁▁▁▁▃██ █
  2.92 μs      Histogram: log(frequency) by time     10.7 μs <

 Memory estimate: 0 bytes, allocs estimate: 0.
````

````julia
@benchmark dot($x, $SA, $y)
````

````
BenchmarkTools.Trial: 109 samples with 1 evaluation.
 Range (min … max):  44.430 ms … 50.313 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     45.854 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   46.158 ms ±  1.296 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

  ▄▁           ▂█   ▅                                          
  ███▃▆▅█▁▃▆▅▆▃██▆▅▆█▅▃▃▃▃▆▆▃▅▅▁▆▁▁▁▆▆▅▁▆▃▁▃▁▅▁▁▃▁▁▁▅▁▃▁▁▁▁▁▃ ▃
  44.4 ms         Histogram: frequency by time        49.8 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.
````

````julia
@benchmark iternz_dot($x, $SA, $y)
````

````
BenchmarkTools.Trial: 10000 samples with 8 evaluations.
 Range (min … max):  3.964 μs …  10.421 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     3.993 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   4.042 μs ± 292.523 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

  █▇▄▃▃▂▂▂▁                                                   ▂
  ████████████▇▇▇▇▆▇▅▅▅▅▅▅▄▅▄▄▅▃▃▁▃▁▄▃▃▁▁▁▃▄▄▃▁▃▃▁▁▁▄▁▃▄▁▁▃▄▄ █
  3.96 μs      Histogram: log(frequency) by time      5.62 μs <

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
BenchmarkTools.Trial: 13 samples with 1 evaluation.
 Range (min … max):  388.609 ms … 394.147 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     390.498 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   390.701 ms ±   1.629 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

  ▁     ▁   ▁ ▁   ▁▁  █  ▁▁  ▁                              ▁ ▁  
  █▁▁▁▁▁█▁▁▁█▁█▁▁▁██▁▁█▁▁██▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁█ ▁
  389 ms           Histogram: frequency by time          394 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.
````

With FLoops.jl:

````julia
@benchmark par_solve!($X, $F, $B)
````

````
BenchmarkTools.Trial: 12 samples with 1 evaluation.
 Range (min … max):  384.391 ms … 481.361 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     435.816 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   426.211 ms ±  33.430 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

  █                                    ▃                         
  █▁▁▁▁▁▇▁▁▁▁▁▁▁▁▇▁▁▁▁▁▁▁▁▁▁▁▇▁▁▁▁▁▁▁▁▁█▁▁▇▁▇▁▁▁▇▁▁▁▁▁▁▁▁▁▁▁▁▁▇ ▁
  384 ms           Histogram: frequency by time          481 ms <

 Memory estimate: 157.64 KiB, allocs estimate: 13.
````

with manual loops

````julia
@benchmark par_ldiv!_t($X, $F, $B)
````

````
BenchmarkTools.Trial: 13 samples with 1 evaluation.
 Range (min … max):  387.524 ms … 424.265 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     390.232 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   395.993 ms ±  10.520 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

  ▁█ ██     ▁    ▁     ▁▁          ▁                          ▁  
  ██▁██▁▁▁▁▁█▁▁▁▁█▁▁▁▁▁██▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  388 ms           Histogram: frequency by time          424 ms <

 Memory estimate: 158.14 KiB, allocs estimate: 16.
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

