module SparseExtra

using SparseArrays
using SparseArrays: AbstractSparseMatrixCSC, AbstractSparseVector
using SparseArrays: getcolptr, getrowval, getnzval, nonzeroinds

using LinearAlgebra

include("iternz.jl")


export iternz

end # module SparseExtra
