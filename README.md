![example workflow](https://github.com/sobhanmp/SparseExtra.jl/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/SobhanMP/SparseExtra.jl/branch/main/graph/badge.svg?token=MzXyDmfScn)](https://codecov.io/gh/SobhanMP/SparseExtra.jl)

Collections of stuff developed by me, mostly sparse stuff.



## `iternz(x)`

This is returns an iterator such that 
```julia
all(iternz(x)) do (v, k...)
    x[k...] == v
end
```

This is useful for iterating over all elements of a sparse container. On dense containers, it will incure a small overhead.

a few benchmarks can be seen [here](https://github.com/JuliaSparse/SparseArrays.jl/pull/167). Basically this removes the need to handcode fast iterations on sparse matrices. The goal is to add support for all special and dense matrices. This would allow fast and generic implementation of many algebraic operations like 3-argument dot.




