

# AbstractSparseMatrixCSC
struct IterateNZ{N, T}
    m::T
end


iternz(x) = IterateNZ{ndims(x), typeof(x)}(x)
Base.eltype(x::IterateNZ{N}) where N = Tuple{eltype(x.m), Vararg{Int, N}}
# default definition, do nothing
@inline skip_col(::IterateNZ, s) = s
@inline skip_row(::IterateNZ, s) = s
@inline skip_col_to(::IterateNZ, s, i) = s
@inline skip_row_to(::IterateNZ, s, i) = s

Base.length(x::IterateNZ{2, <:AbstractSparseMatrixCSC}) = nnz(x.m)
@inline Base.iterate(x::IterateNZ{2, <:AbstractSparseMatrixCSC}, state=(1, 1)) =
@inbounds let (j, ind) = state
    while j < size(x.m, 2) && ind >= getcolptr(x.m)[j + 1] # skip empty cols or one that have been exhusted
        j += 1
    end
    (j > size(x.m, 2) || ind > nnz(x.m)) && return nothing
    (getnzval(x.m)[ind], getrowval(x.m)[ind], j), (j, ind + 1)
end
@inline skip_col(x::IterateNZ{2, <:AbstractSparseMatrixCSC}, state) =
@inbounds let (j, ind) = state, m = x.m
    j, max(getcolptr(m)[j + 1] - 1, ind)
end

@inline skip_row_to(x::IterateNZ{2, <:AbstractSparseMatrixCSC{Tv, Ti}}, state, i) where {Tv, Ti}=
@inbounds let (j, ind) = state,
    r1 = convert(Ti, x.m.colptr[j])
    r2 = convert(Ti, x.m.colptr[j + 1] - 1)
    iTi = convert(Ti, i)
    # skip empty cols
    r3 = searchsortedfirst(rowvals(x.m), iTi, r1, r2, Base.Forward)
    if r3 > r2
        j + 1, getcolptr(x.m)[j + 1]
    else
        j, max(ind, convert(Int, r3))
    end
end

@inline Base.length(x::IterateNZ{1, <:AbstractSparseVector}) = nnz(x.m)
@inline Base.iterate(x::IterateNZ{1, <:AbstractSparseVector}, state=0) = @inbounds let m = x.m
    state += 1
    if state > nnz(m)
        nothing
    else
        (nonzeros(m)[state], nonzeroinds(m)[state]), state
    end
end
Base.length(x::IterateNZ{N, <:AbstractArray}) where N = length(x.m)
@inline Base.iterate(x::IterateNZ{N, <:AbstractArray}) where N = begin
    c = CartesianIndices(x.m)
    i, s = iterate(c)
    mi, ms = iterate(x.m)
    (mi, Tuple(i)...), (ms, c, s)
end
@inline Base.iterate(x::IterateNZ{N, <:AbstractArray}, state) where N = @inbounds begin
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
    
    if i > size(x.m, 1)
        ind = ind + stride(x.m, 2) - (size(x.m, 1) - 1) * stride(x.m, 1)
        i = 1
        j += 1
    else
        ind += stride(x.m, 1)
    end
    if j > size(x.m, 2)
        return nothing
    end
    (x.m[ind], i, j), (ind, i, j)
end

@inline skip_col(x::IterateNZ{2, <:StridedMatrix}, state) =
@inbounds begin
    ind, i, j = state
    i1 = size(x.m, 1)
    ind = ind + (i1 - i) * stride(x.m, 1)
    (ind, i1, j)
end

@inline skip_row_to(x::IterateNZ{2, <:StridedMatrix}, state, i1) =
@inbounds begin
    ind, i, j = state
    (ind + (i1 - i - 1) * stride(x.m, 1), i1 - 1, j)
end

Base.IteratorSize(x::IterateNZ{2, <:Transpose}) = Base.IteratorSize(iternz(x.m.parent))
Base.length(x::IterateNZ{2, <:Transpose}) = length(iternz(x.m.parent))
@inline Base.iterate(x::IterateNZ{2, <:Transpose}) = 
let state = iternz(x.m.parent),
    a = iterate(state)

    a === nothing && return nothing
    (v, i, j), s = a
    (v, j, i), (state, s)
end
@inline Base.iterate(::IterateNZ{2, <:Transpose}, state) =
let a = iterate(state...)
    a === nothing && return nothing
    (v, i, j), s = a
    (v, j, i), (state[1], s)
end

Base.IteratorSize(x::IterateNZ{2, <:Adjoint}) = Base.IteratorSize(iternz(x.m.parent))
Base.length(x::IterateNZ{2, <:Adjoint}) = length(iternz(x.m.parent))
@inline Base.iterate(x::IterateNZ{2, <:Adjoint}) = 
let state = iternz(x.m.parent),
    a = iterate(state)

    a === nothing && return nothing
    (v, i, j), s = a
    (adjoint(v), j, i), (state, s)
end
@inline Base.iterate(::IterateNZ{2, <:Adjoint}, state) =
let a = iterate(state...)
    a === nothing && return nothing
    (v, i, j), s = a
    (adjoint(v), j, i), (state[1], s)
end

Base.IteratorSize(x::IterateNZ{2, <:Diagonal}) = Base.IteratorSize(iternz(x.m.diag))
Base.length(x::IterateNZ{2, <:Diagonal}) = length(iternz(x.m.diag))
@inline Base.iterate(x::IterateNZ{2, <:Diagonal}) = let state = iternz(x.m.diag),
    a = iterate(state)

    a === nothing && return nothing
    (v, i), s = a
    (v, i, i), (state, s)
end
@inline Base.iterate(::IterateNZ{2, <:Diagonal}, state) =
let a = iterate(state...)
    a === nothing && return nothing
    (v, i), s = a
    (v, i, i), (state[1], s)
end

Base.IteratorSize(::IterateNZ{2, <:UpperTriangular}) = Base.SizeUnknown()
@inline Base.iterate(x::IterateNZ{2, <:UpperTriangular}) =
let inner_state = iternz(x.m.data)
    iternzut(inner_state, iterate(inner_state))
end

@inline Base.iterate(::IterateNZ{2, <:UpperTriangular}, state) =
let (inner_state, s) = state
    iternzut(inner_state, iterate(inner_state, s))
end

@inline iternzut(iterator, a) = @inbounds begin
    while a !== nothing
        (v, i, j), state = a
        i <= j && return (v, i, j), (iterator, state)
        state = skip_col(iterator, state)
        a = iterate(iterator, state)
    end
    nothing
end



Base.IteratorSize(::IterateNZ{2, <:LowerTriangular}) = Base.SizeUnknown()
@inline Base.iterate(x::IterateNZ{2, <:LowerTriangular}) =
let inner_state = iternz(x.m.data)
    iternzlt(inner_state, iterate(inner_state))
end

@inline Base.iterate(::IterateNZ{2, <:LowerTriangular}, state) =
let (inner_state, s) = state
    iternzlt(inner_state, iterate(inner_state, s))
end

@inline iternzlt(iterator, a) = @inbounds begin
    while a !== nothing
        (v, i, j), state = a
        i >= j && return (v, i, j), (iterator, state)
        state = skip_row_to(iterator, state, j)
        a = iterate(iterator, state)
    end
    nothing
end


Base.IteratorSize(::IterateNZ{2, <:Symmetric}) = Base.SizeUnknown()

@inline Base.iterate(x::IterateNZ{2, <:Symmetric}) =
let iterator = iternz(x.m.data)
    iternzsym(x.m, iterator, iterate(iterator))
end

@inline Base.iterate(x::IterateNZ{2, <:Symmetric}, state) =
let (iterator, (v, i, j), r, s) = state
    if r
        (v, j, i), (iterator, (v, i, j), false, s)
    else
        iternzsym(x.m, iterator, iterate(iterator, s))
    end
end

@inline iternzsym(m::Symmetric, iterator, a) = @inbounds begin
    while a !== nothing
        r, state = a
        (_, i, j) = r
        if m.uplo == 'U'
            i <= j && return r, (iterator, r, i != j, state)
            state = skip_col(iterator, state)
        elseif m.uplo == 'L'
            i >= j && return r, (iterator, r, i != j, state)
            state = skip_row_to(iterator, state, j)
        end
        a = iterate(iterator, state)
    end
    nothing
end




