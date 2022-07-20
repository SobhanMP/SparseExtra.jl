@inline unsafe_sum(a, i0=firstindex(a), i1=lastindex) = unsafe_sum(a, i0:i1)
function unsafe_sum(a::StridedVector{T}, r) where {T}
    stride(a, 1) == 1 || error()
    s = zero(T)
    @inbounds for i in r
        s += a[i]
    end
    s
end

function colnorm!(x::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    for i in axes(x, 2)
        a, b = getcolptr(x)[i], getcolptr(x)[i + 1] - 1
        s = unsafe_sum(getnzval(x), a, b)
        @inbounds for j in a:b
            getnzval(x)[j] /= s
        end
    end
end

function colnorm!(x::SparseMatrixCSC{Tv, Ti}, abs, offset) where {Tv, Ti}
    abs_ind = 1
    for i in axes(x, 2)
        of = if abs_ind <= length(abs) && i == abs[abs_ind]
            abs_ind += 1
            offset
        else
            zero(offset)
        end
        r = getcolptr(x)[i]: getcolptr(x)[i + 1] - 1
        s = unsafe_sum(nonzeros(x), r) + of
        @inbounds for j in r
            nonzeros(x)[j] /= s
        end
    end
end

@inline function colsum(x::SparseMatrixCSC, i)
    @boundscheck checkbounds(x, :, i)
    return unsafe_sum(getnzval(x), getcolptr(x)[i], getcolptr(x)[i + 1] - 1)
end

sparse_like(t::SparseMatrixCSC, T::Type) =
    SparseMatrixCSC(t.m, t.n, t.colptr, t.rowval, Vector{T}(undef, nnz(t)))
sparse_like(t::SparseMatrixCSC, z) =
    SparseMatrixCSC(t.m, t.n, t.colptr, t.rowval, fill(z, nnz(t)))
sparse_like(t::SparseMatrixCSC, z::Vector) =
    SparseMatrixCSC(t.m, t.n, t.colptr, t.rowval, z)
sparse_like(t::SparseMatrixCSC) =
    SparseMatrixCSC(t.m, t.n, t.colptr, t.rowval, copy(t.nzval))



function getindex_(A::SparseMatrixCSC{Tv, Ti}, i0::T, i1::T)::T where {Tv, Ti, T}
    @boundscheck checkbounds(A, i0, i1)
    r1 = convert(Ti, A.colptr[i1])
    r2 = convert(Ti, A.colptr[i1 + 1] - 1)
    (r1 > r2) && return zero(T)
    r3 = searchsortedfirst(rowvals(A), i0, r1, r2, Base.Forward)
    @boundscheck !((r3 > r2) || (rowvals(A)[r3] != i0))
    return r3
end