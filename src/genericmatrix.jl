struct GenericSparseMatrixCSC{
    Tv, Ti, 
    Av<:AbstractVector{Tv}, 
    Aic<:AbstractVector{Ti},
    Air<:AbstractVector{Ti}} <: AbstractSparseMatrixCSC{Tv, Ti}

   m::Int
   n::Int
   colptr::Aic
   rowval::Air
   nzval::Av

   function GenericSparseMatrixCSC(
        m::Integer, n::Integer, 
        colptr::Aic,
        rowval::Air, 
        nzval::Av) where {Av, Aic, Air}
        Ti = eltype(Aic)
        Ti === eltype(Air) || error("eltype of GSM's col and row must be the same")
        sparse_check_Ti(m, n, Ti)
        _goodbuffers(Int(m), Int(n), colptr, rowval, nzval) || throw(ArgumentError("Invalid buffer"))
        
        new{eltype(Av), Ti, Av, Aic, Air}(Int(m), Int(n), colptr, rowval, nzval)
    end
end

GenericSparseMatrixCSC(a::SparseMatrixCSC) = GenericSparseMatrixCSC(a.m, a.n, a.colptr, a.rowval, a.nzval)

@inline SparseArrays.rowvals(x::GenericSparseMatrixCSC) = x.rowval
@inline SparseArrays.nonzeros(x::GenericSparseMatrixCSC) = x.nzval
@inline SparseArrays.getcolptr(x::GenericSparseMatrixCSC) = x.colptr
@inline SparseArrays.getrowval(x::GenericSparseMatrixCSC) = x.rowval
@inline SparseArrays.getnzval(x::GenericSparseMatrixCSC) = x.nzval


Base.size(S::GenericSparseMatrixCSC) = (getfield(S, :m), getfield(S, :n))
Base.size(S::GenericSparseMatrixCSC, i) = if i <= 0
    error()
elseif i == 1
    getfield(S, :m)
elseif i == 2
    getfield(S, :n)
else
    error()
end
SparseArrays._goodbuffers(S::AbstractSparseMatrixCSC) = _goodbuffers(size(S)..., getcolptr(S), getrowval(S), nonzeros(S))
SparseArrays._checkbuffers(S::AbstractSparseMatrixCSC) = (@assert _goodbuffers(S); S)
