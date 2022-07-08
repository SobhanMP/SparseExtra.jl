function unsafe_sum(a::AbstractVector{T}, i0::Integer=1, i1::Integer=length(a)) where {T}
    s = zero(T)
    @inbounds for i in i0:i1
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
        s = unsafe_sum(getnzval(x), getcolptr(x)[i], getcolptr(x)[i + 1] - 1) + of
        @inbounds for j in a:b
            xnz[j] /= s
        end
    end
end

function colsum(x::SparseMatrixCSC, i)
    @boundscheck checkbounds(x, :, i)
    return unsafe_sum(getnzval(x), getcolptr(x)[i], getcolptr(x)[i + 1] - 1)
end

sparse_like(t::SparseMatrixCSC{Tv}, T::Type=Tv) where Tv =
    SparseMatrixCSC(t.m, t.n, t.colptr, t.rowval, Vector{T}(undef, nnz(t)))
sparse_like(t::SparseMatrixCSC{Tv, Ti}, z::T) where {T, Tv, Ti} =
    SparseMatrixCSC{T, Ti}(t.m, t.n, t.colptr, t.rowval, fill(z, nnz(t)))
sparse_like(t::SparseMatrixCSC, z::Vector) =
    SparseMatrixCSC(t.m, t.n, t.colptr, t.rowval, z)
sparse_like(t::SparseMatrixCSC) =
    SparseMatrixCSC(t.m, t.n, t.colptr, t.rowval, copy(t.nzval))