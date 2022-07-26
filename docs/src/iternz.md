```@meta
EditURL = "../lit/iternz.jl"
```

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
BenchmarkTools.Trial: 11 samples with 1 evaluation.
 Range (min … max):  461.048 ms … 468.458 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     463.496 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   463.928 ms ±   2.200 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

                                       █                         
  ▇▁▇▁▁▁▁▁▁▁▇▁▁▁▇▇▁▁▁▁▇▁▁▇▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▇ ▁
  461 ms           Histogram: frequency by time          468 ms <

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
BenchmarkTools.Trial: 301 samples with 1 evaluation.
 Range (min … max):  15.765 ms …  19.260 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     16.511 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   16.646 ms ± 566.611 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

       ▁▃▅▄█▃▄ ▃ ▁▄   ▁                                         
  ▃▁▃▃▄█████████▇██▆▅▇██▅▆▅█▇▅▃▆▅▃▆▅▆▄▃▄▁▁▃▁▄▃▅▃▁▃▁▁▃▃▁▁▁▁▁▃▁▃ ▄
  15.8 ms         Histogram: frequency by time         18.6 ms <

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
BenchmarkTools.Trial: 266 samples with 1 evaluation.
 Range (min … max):  15.932 ms … 32.230 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     17.630 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   18.806 ms ±  2.971 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

    ▄█▅▂█▁▃                                                    
  ▅█████████▄▅▅▆▅▆▆▄▄▄▄▅▃▁▃▄▃▄▃▁▅▄▁▃▃▃▁▁▃▁▃▃▃▁▄▃▃▁▁▃▃▃▁▃▃▁▁▃▃ ▃
  15.9 ms         Histogram: frequency by time        28.3 ms <

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
BenchmarkTools.Trial: 500 samples with 1 evaluation.
 Range (min … max):  9.153 ms …  16.797 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     9.794 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   9.998 ms ± 744.517 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

    ▃    ▂▄█▃ ▁                                                
  ██████▇██████▆█▅▄▄▅▄▄▄▄▄▄▅▄▄▄▃▄▃▄▄▃▃▃▁▃▁▁▂▂▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▂ ▄
  9.15 ms         Histogram: frequency by time        12.9 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.
````

````julia
@benchmark iternz_dot($x, $A, $y)
````

````
BenchmarkTools.Trial: 426 samples with 1 evaluation.
 Range (min … max):  10.760 ms …  16.682 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     11.519 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   11.727 ms ± 741.895 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

         ▅█▄▁▂▂                                                 
  ▅▇▇▆█▇▇████████▆█▄▆██▆▄▆▄▆▅▆▃▄▄▃▄▂▄▃▃▂▂▁▃▁▁▁▂▃▁▁▃▁▃▁▁▁▃▃▁▁▁▂ ▄
  10.8 ms         Histogram: frequency by time         14.5 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.
````

````julia
@benchmark dot($x, $SA, $y)
````

````
BenchmarkTools.Trial: 9 samples with 1 evaluation.
 Range (min … max):  607.881 ms … 623.882 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     609.728 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   611.193 ms ±   4.900 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

  ▁  █ ▁ ▁█      ▁                                            ▁  
  █▁▁█▁█▁██▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  608 ms           Histogram: frequency by time          624 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.
````

````julia
@benchmark iternz_dot($x, $SA, $y)
````

````
BenchmarkTools.Trial: 440 samples with 1 evaluation.
 Range (min … max):  10.576 ms …  17.101 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     11.173 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   11.371 ms ± 740.788 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

   ▂▃   █▆▃▄▃                                                   
  ▆███████████▇▆▇▇▅▆▆▆▅▄▅▅▃▄▃▃▃▃▃▂▁▂▁▁▂▂▃▁▃▁▁▂▂▁▂▁▁▁▂▁▁▁▃▁▁▁▁▂ ▃
  10.6 ms         Histogram: frequency by time         14.4 ms <

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

