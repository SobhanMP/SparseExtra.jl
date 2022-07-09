

# AbstractSparseMatrixCSC
struct IterateNZ{N, T}
    m::T
end
SparseArrays.nnz(x) = length(x)
iternz(x) = IterateNZ{ndims(x), typeof(x)}(x)
Base.length(x::IterateNZ) = length(x.m) 
Base.eltype(x::IterateNZ{N}) where N = Tuple{eltype(x.m), Vararg{Int, N}}
@inline skip_col(x::IterateNZ, s) = s

Base.length(x::IterateNZ{2, <:AbstractSparseMatrixCSC}) = nnz(x.m)
Base.iterate(x::IterateNZ{2, <:AbstractSparseMatrixCSC}, state=(1, 0)) = 
@inbounds let (j, ind) = state,
    m = x.m
    ind += 1
    while (j < size(m, 2)) && (ind > getcolptr(m)[j + 1] - 1) # skip empty cols or one that have been exhusted
        j += 1
    end
    (j > size(m, 2) || ind > getcolptr(m)[end] - 1) && return nothing

    (getnzval(m)[ind], getrowval(m)[ind], j), (j, ind)
end
@inline skip_col(x::IterateNZ{2, <:AbstractSparseMatrixCSC}, state) = 
@inbounds let (j, ind) = state,
    m = x.m
    # skip empty cols
    while getcolptr(m)[j] > getcolptr(m)[j + 1] - 1
        j += 1
    end
    ind = max(ind, getcolptr(m)[j])
    j, ind
end

Base.length(x::IterateNZ{1, <:AbstractSparseVector}) = nnz(x.m)
Base.iterate(x::IterateNZ{1, <:AbstractSparseVector}, state=0) = @inbounds let m = x.m
    state += 1
    if state > nnz(m)
        nothing
    else
        (nonzeros(m)[state], nonzeroinds(m)[state]), state
    end
end

Base.iterate(x::IterateNZ{N, <:AbstractArray}) where N = begin
    c = CartesianIndices(x.m)
    i, s = iterate(c)
    mi, ms = iterate(x.m)
    (mi, Tuple(i)...), (ms, c, s)
end
Base.iterate(x::IterateNZ{N, <:AbstractArray}, state) where N = @inbounds begin
    ms, ii, is = state
    res = iterate(ii, is)
    res === nothing && return nothing
    res1 = iterate(x.m, ms)
    res1 === nothing && return nothing # shoud not happen
    mi, nms = res1
    i, s = res
    (mi, Tuple(i)...), (nms, ii, s)
end

@inline function Base.iterate(x::IterateNZ{2, <:StridedMatrix})
    ind, i, j = firstindex(x.m), 1, 1
    return ((x.m[ind], i, j), (ind, i, j))
end
@inline Base.iterate(x::IterateNZ{2, <:StridedMatrix}, state) = @inbounds begin
    ind, i, j = state
    i += 1
    ind += stride(x.m, 1)
    if i > size(x.m, 1)
        i = 1
        j += 1
        ind = ind - size(x.m, 1) * stride(x.m, 1) + stride(x.m, 2)
    end
    if j > size(x.m, 2)
        return nothing
    end
    (x.m[ind], i, j), (ind, i, j)
end

@inline skip_col(x::IterateNZ{2, <:StridedMatrix}, state) where N =
@inbounds begin
    ind, i, j = state
    ind = ind - (i - 1) * stride(x.m, 1) + stride(x.m, 2)
    (ind, 1, j + 1)
end



Base.length(x::IterateNZ{2, <:Diagonal}) = nnz(x.m.diag)
Base.iterate(x::IterateNZ{2, <:Diagonal}) = let state = iternz(x.m.diag),
    a = iterate(state)

    a === nothing && return nothing
    (v, i), s = a
    (v, i, i), (state, s)
end
Base.iterate(::IterateNZ{2, <:Diagonal}, state) = 
let a = iterate(state...)
    a === nothing && return nothing
    (v, i), s = a
    (v, i, i), (state[1], s)
end


iternz(x::UpperTriangular) = Iterators.filter(iternz(x.data)) do (_, i, j)
    i <= j
end
    
iternz(x::LowerTriangular) = Iterators.filter(iternz(x.data)) do (_, i, j)
    i >= j
end