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
BenchmarkTools.Trial: 95 samples with 1 evaluation.
 Range (min … max):  52.295 ms …  53.141 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     52.626 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   52.649 ms ± 195.350 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

    ▂     ▂   ▅▂  ▂ ▅▂   █▂ ▂▅        ▅      ▂ █                
  ▅▅█▁▅▁▅▅██▁▅█████▅██▅▅███▅██████▅▁█▅██▁▅▅▅▅█▅█▁█▅▁▅▁▁▁▁▁▁▁▅█ ▁
  52.3 ms         Histogram: frequency by time         53.1 ms <

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
 Range (min … max):  5.388 μs …  27.941 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     5.543 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   5.668 μs ± 616.151 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

  █▇▇▇▄▄▄▃▂▂▃▂▂▁                                              ▂
  ████████████████▇▇▇▆▆▆▆▄▄▆▆▅▅▅▆▄▆▆▄▆▅▅▄▅▄▅▅▅▅▄▄▅▄▅▄▃▅▅▅▄▅▅▄ █
  5.39 μs      Histogram: log(frequency) by time      8.76 μs <

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
 Range (min … max):  5.351 μs …  26.629 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     5.425 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   5.586 μs ± 661.738 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

  ▆█▄▆▅▃▂▂▃▄▂▂▁▁                                              ▁
  ████████████████▇▇▇▆▆▅▄▄▄▅▃▄▂▄▄▅▄▃▅▃▅▅▄▅▅▅▅▄▃▅▄▅▃▄▄▄▄▄▄▄▅▅▅ █
  5.35 μs      Histogram: log(frequency) by time      7.96 μs <

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
 Range (min … max):  13.611 μs … 82.077 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     14.560 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   14.927 μs ±  1.848 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

     ▃█▇▄                                                      
  ▁▂▄█████▆▄▄▄▃▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
  13.6 μs         Histogram: frequency by time        23.1 μs <

 Memory estimate: 0 bytes, allocs estimate: 0.
````

````julia
@benchmark iternz_dot($x, $A, $y)
````

````
BenchmarkTools.Trial: 10000 samples with 8 evaluations.
 Range (min … max):  2.978 μs …  22.270 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     3.148 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   3.326 μs ± 525.230 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

   ██▂                                                         
  ████▆▄▃▃▃█▅▅▃▂▂▃▃▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
  2.98 μs         Histogram: frequency by time        5.47 μs <

 Memory estimate: 0 bytes, allocs estimate: 0.
````

````julia
@benchmark dot($x, $SA, $y)
````

````
BenchmarkTools.Trial: 111 samples with 1 evaluation.
 Range (min … max):  44.814 ms …  46.374 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     45.451 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   45.444 ms ± 351.582 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

    ▁     ▃ ▃   ▁   ▁  ▃ ▆   ▃▃▃█     ▁  ▁   ▁  ▁               
  ▇▁█▇▄▇▄▁█▄█▄▇▇█▇▄▄█▇▄█▄█▄▄▄████▁▇▇▇▇█▇▁█▁▇▇█▁▇█▇▇▁▇▁▄▇▇▄▁▁▁▄ ▄
  44.8 ms         Histogram: frequency by time         46.2 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.
````

````julia
@benchmark iternz_dot($x, $SA, $y)
````

````
BenchmarkTools.Trial: 10000 samples with 8 evaluations.
 Range (min … max):  3.478 μs …  16.866 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     3.581 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   3.677 μs ± 372.602 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

  ▆█ ▁                                                         
  ██▇█▆▄▂▄▅▄▃▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
  3.48 μs         Histogram: frequency by time        5.32 μs <

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

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

