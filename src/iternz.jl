abstract type SparseIndexIterate end
@inline SparseArrays.getcolptr(x::SparseIndexIterate) = getcolptr(x.m)
@inline SparseArrays.getrowval(x::SparseIndexIterate) = getrowval(x.m)
@inline SparseArrays.getnzval(x::SparseIndexIterate) = getnzval(x.m)
@inline SparseArrays.nonzeroinds(x::SparseIndexIterate) = nonzeroinds(x.m)
@inline SparseArrays.nonzeros(x::SparseIndexIterate) = nonzeros(x.m)
@inline SparseArrays.nnz(x::SparseIndexIterate) = nnz(x.m)
@inline Base.length(x::SparseIndexIterate) = nnz(x.m)
@inline Base.size(x::SparseIndexIterate) = size(x.m)
@inline Base.size(x::SparseIndexIterate, i) = size(x.m)[i]

struct IterateNZCSC{T<: AbstractSparseMatrixCSC} <: SparseIndexIterate
    m::T
end

Base.eltype(::IterateNZCSC{T}) where {Ti, Tv, T <: AbstractSparseMatrixCSC{Tv, Ti}} = Tuple{Ti, Ti, Tv}
Base.iterate(x::IterateNZCSC, state=(1, 0)) = @inbounds let (j, ind) = state
    ind += 1
    while (j < size(x, 2)) && (ind > getcolptr(x)[j + 1] - 1)
        j += 1
    end
    (j > size(x, 2) || ind > getcolptr(x)[end] - 1) && return nothing

    (getrowval(x)[ind], j, getnzval(x)[ind]), (j, ind)
end

iternz(S::AbstractSparseMatrixCSC) = IterateNZCSC(S)



struct IterateSparseVec{T<: AbstractSparseVector} <: SparseIndexIterate
    m::T
end

Base.eltype(::IterateSparseVec{T}) where {Ti, Tv, T <: AbstractSparseVector{Tv, Ti}} = Tuple{Ti, Tv}

Base.iterate(x::IterateSparseVec, state=0) = @inbounds begin
    state += 1
    if state > nnz(x)
        nothing
    else
        (nonzeroinds(x)[state], nonzeros(x)[state]), state
    end
end

iternz(S::AbstractSparseVector) = IterateSparseVec(S)


