abstract type SparseIndexIterate end





# AbstractSparseMatrixCSC
struct IterateNZCSC{T<: AbstractSparseMatrixCSC} <: SparseIndexIterate
    m::T
end
@inline Base.length(x::IterateNZCSC) = nnz(x.m)
@inline Base.eltype(::IterateNZCSC{T}) where {Ti, Tv, T <: AbstractSparseMatrixCSC{Tv, Ti}} = Tuple{Tv, Ti, Ti}
@inline Base.iterate(x::IterateNZCSC, state=(1, 0)) = @inbounds let (j, ind) = state,
    m = x.m
    ind += 1
    while (j < size(m, 2)) && (ind > getcolptr(m)[j + 1] - 1) # skip empty cols or one that have been exhusted
        j += 1
    end
    (j > size(m, 2) || ind > getcolptr(m)[end] - 1) && return nothing

    (getnzval(m)[ind], getrowval(m)[ind], j), (j, ind)
end

iternz(S::AbstractSparseMatrixCSC) = IterateNZCSC(S)


# AbstractSparseVector
struct IterateSparseVec{T<: AbstractSparseVector} <: SparseIndexIterate
    m::T
end
@inline Base.length(x::IterateSparseVec) = nnz(x.m)
@inline Base.eltype(::IterateSparseVec{T}) where {Ti, Tv, T <: AbstractSparseVector{Tv, Ti}} = Tuple{Tv, Ti}
@inline Base.iterate(x::IterateSparseVec, state=0) = @inbounds let m = x.m
    state += 1
    if state > nnz(m)
        nothing
    else
        (nonzeros(m)[state], nonzeroinds(m)[state]), state
    end
end
iternz(S::AbstractSparseVector) = IterateSparseVec(S)



# AbstractArray
struct IterateAbstractArray{T, N, A<:AbstractArray{T, N}}
    x::A
end


@inline second(x::Tuple) = x[2]
@inline Base.length(x::IterateAbstractArray) = length(x.x)
@inline Base.size(x::IterateAbstractArray) = (length(x),)
@inline Base.eltype(::IterateAbstractArray{T, N}) where {T, N} = Tuple{T, Vararg{Int, N}}


@inline Base.iterate(x::IterateAbstractArray) = begin
    c = CartesianIndices(x.x)
    i, s = iterate(c)
    (x.x[i], Tuple(i)...), (c, s)
end

@inline Base.iterate(x::IterateAbstractArray, state) = @inbounds begin
    res = iterate(state...)
    res === nothing && return nothing
    i, s = res
    (x.x[i], Tuple(i)...), (state[1], s)
end

iternz(x::A) where {T, N, A<:AbstractArray{T, N}} = IterateAbstractArray{T, N, A}(x)
