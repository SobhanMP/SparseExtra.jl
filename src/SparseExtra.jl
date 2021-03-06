module SparseExtra

using SparseArrays
using SparseArrays: AbstractSparseMatrixCSC, AbstractSparseVector
using SparseArrays: getcolptr, getrowval, getnzval, nonzeroinds
using SparseArrays: rowvals, nonzeros
using SparseArrays: sparse_check_Ti, _goodbuffers
if isdefined(SparseArrays, :UMFPACK)
    using SparseArrays.UMFPACK: UmfpackLU
    using SparseArrays: UMFPACK
else
    using SuiteSparse.UMFPACK: UmfpackLU
    using SuiteSparse: UMFPACK
end

using VectorizationBase
using VectorizationBase: vsum, vprod, vmaximum, vminimum, bitselect

using LinearAlgebra, StaticArrays
using DataStructures
using DataStructures: FasterForward
using FLoops
include("iternz.jl")
export iternz
include("sparse_helpers.jl")
export unsafe_sum, colnorm!, colsum, sparse_like, getindex_
include("solve.jl")
export par_solve!, par_solve, par_inv!, par_inv
include("graph.jl")
export dijkstra, DijkstraState, sparse_feature_vec, sparse_feature, Path2Edge, path_cost, path_cost_nt, MkPair
include("genericmatrix.jl")
export GenericSparseMatrixCSC
include("simd.jl")
end # module SparseExtra
