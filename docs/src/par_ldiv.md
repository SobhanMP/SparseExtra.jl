```@meta
EditURL = "<unknown>/docs/lit/par_ldiv.jl"
```

# parallel ldiv!

````julia
using SparseExtra, LinearAlgebra, SparseArrays, BenchmarkTools
const n = 10_000
const A = sprandn(n, n, 5 / n);
const C = A + I
const B = Matrix(sprandn(n, n, 1 / n));
const F = lu(C);
const X = similar(B);
````

Standard:

````julia
@benchmark ldiv!($X, $F, $B)
````

````
BenchmarkTools.Trial: 1 sample with 1 evaluation.
 Single result which took 96.309 s (0.00% GC) to evaluate,
 with a memory estimate of 0 bytes, over 0 allocations.
````

With FLoops.jl:

````julia
@benchmark par_solve!($X, $F, $B)
````

````
BenchmarkTools.Trial: 1 sample with 1 evaluation.
 Single result which took 25.195 s (0.00% GC) to evaluate,
 with a memory estimate of 1.24 MiB, over 163 allocations.
````

with manual loops

````julia
@benchmark par_ldiv!_t($X, $F, $B)
````

````
BenchmarkTools.Trial: 1 sample with 1 evaluation.
 Single result which took 25.395 s (0.00% GC) to evaluate,
 with a memory estimate of 1.24 MiB, over 130 allocations.
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

