module SparseExtra

using SparseArrays
using SparseArrays: AbstractSparseMatrixCSC, AbstractSparseVector
using SparseArrays: getcolptr, getrowval, getnzval, nonzeroinds

using LinearAlgebra, StaticArrays

include("iternz.jl")
export iternz
include("sparse_helpers.jl")
export unsafe_sum, colnorm!

end # module SparseExtra
