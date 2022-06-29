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


@inline Base.length(x::IterateAbstractArray) = length(x.x)
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


# Diagonal
struct IterateDiagonal{T, S}
    x::S
end
Base.length(x::IterateDiagonal) = length(x.x)
Base.iterate(x::IterateDiagonal) = let a = iterate(x.x)
    a === nothing && return nothing
    (v, i), s = a
    (v, i, i), s
end
Base.iterate(x::IterateDiagonal, state) = let a = iterate(x.x, state)
    a === nothing && return nothing
    (v, i), s = a
    (v, i, i), s
end
Base.eltype(::IterateDiagonal{T}) where T = Tuple{T, Int, Int}
iternz(x::Diagonal{T, <:AbstractVector{T}}) where T = let a = iternz(x.diag)
    IterateDiagonal{T, typeof(a)}(a)
end


# UpperTriangular
struct IterateUpperTriangular{T, S}
    x::S
end
Base.eltype(::IterateUpperTriangular{T}) where T = Tuple{T, Int, Int}
# can't know length

Base.iterate(x::IterateUpperTriangular) = let a = iterate(x.x)
    if a === nothing 
        nothing
    else let ((v, i, j), s) = a
            if i <= j
                (v, i, j)
            else
                iterate(x, s)
            end
        end
    end
end

Base.iterate(x::IterateUpperTriangular, state) = while true
    a = iterate(x.x, state)
    a === nothing && return nothing
    (v, i, j), s = a
    i <= j && return ((v, i, j), s)
end

iternz(x::UpperTriangular{T, <:AbstractMatrix{T}}) where T = let a = iternz(x.data)
    IterateDiagonal{T, typeof(a)}(a)
end

