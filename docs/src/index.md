# SparseExtra

```@meta
DocTestSetup = :(using SparseArrays, LinearAlgebra, SparseExtra)
```


**SparseExtra.jl** is a a series of helper function useful for dealing with sparse matrix, random walk, and shortest path on sparse graphs and some more. Most of this was developed for an unpublished paper which would help explain some of the design choises.




# [SparseExtra](@id sparse-extra)
```@docs
SparseExtra.par_solve_f! 
SparseExtra.skip_row_to 
SparseExtra.extract_path 
SparseExtra.skip_col 
SparseExtra.IterateNZ
SparseExtra.path_cost 
SparseExtra.GenericSparseMatrixCSC
SparseExtra.path_cost_nt 
SparseExtra.DijkstraState
SparseExtra.Path2Edge
SparseExtra.par_solve! 
SparseExtra.dijkstra 
SparseExtra.iternz 
SparseExtra.par_inv!
```




