

# AbstractSparseMatrixCSC
struct IterateNZ{N, T}
    m::T
end
SparseArrays.nnz(x) = length(x)
iternz(x) = IterateNZ{ndims(x), typeof(x)}(x)
Base.length(x::IterateNZ) = length(x.m) 
Base.eltype(x::IterateNZ{N}) where N = Tuple{eltype(x.m), Vararg{Int, N}}

Base.length(x::IterateNZ{2, <:AbstractSparseMatrixCSC}) = nnz(x.m)
Base.iterate(x::IterateNZ{2, <:AbstractSparseMatrixCSC}, state=(1, 0)) = @inbounds let (j, ind) = state,
    m = x.m
    ind += 1
    while (j < size(m, 2)) && (ind > getcolptr(m)[j + 1] - 1) # skip empty cols or one that have been exhusted
        j += 1
    end
    (j > size(m, 2) || ind > getcolptr(m)[end] - 1) && return nothing

    (getnzval(m)[ind], getrowval(m)[ind], j), (j, ind)
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
    (x.m[i], Tuple(i)...), (c, s)
end
Base.iterate(x::IterateNZ{N, <:AbstractArray}, state) where N = @inbounds begin
    res = iterate(state...)
    res === nothing && return nothing
    i, s = res
    (x.m[i], Tuple(i)...), (state[1], s)
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