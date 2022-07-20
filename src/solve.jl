"""
solve lu\\b in parallel
"""
const LU{T} = Union{Transpose{UmfpackLU{T}}, UmfpackLU{T}}
const LU_has_lock = hasfield(UmfpackLU, :lock)
if LU_has_lock
    @eval begin
        using SparseArrays.UMFPACK: duplicate
        UMFPACK.duplicate(F::Transpose{T, <:UmfpackLU}) where T = Transpose(duplicate(F.parent))
        UMFPACK.duplicate(F::Adjoint{T, <:UmfpackLU}) where T = Adjoint(duplicate(F.parent))
    end
else
    @eval begin
        SparseArrays._goodbuffers(S::AbstractSparseMatrixCSC) = _goodbuffers(size(S)..., getcolptr(S), getrowval(S), nonzeros(S))
        SparseArrays._checkbuffers(S::AbstractSparseMatrixCSC) = (@assert _goodbuffers(S); S)
        duplicate(x) = x
    end
end
function par_solve!(x, lu::LU, b::AbstractMatrix; cols=axes(b, 2)) where F
    if LU_has_lock
        @floop for i in cols
            @init dlu = duplicate(lu)
            v = view(x, :, i)
            ldiv!(v, dlu, view(b, :, i))
        end
    else
        @floop for i in cols
            v = view(x, :, i)
            ldiv!(v, lu, view(b, :, i))
        end
    end
    return x
end
par_solve(lu::LU, b::AbstractMatrix; cols=axes(b, 2)) = par_solve!(similar(b), lu, b; cols)
function par_solve_f!(x, f!::F, g!::G, lu::LU{T}; cols=1:size(lu, 2)) where {T, F, G}
    size(x, 1) == size(lu, 1) || error("can't")
    if LU_has_lock
        @floop for i in cols
            @init dlu = duplicate(lu)
            @init storage = zeros(T, size(lu, 1))
            f!(storage, i)
            v = view(x, :, i)
            ldiv!(v, dlu, storage)
            g!(storage, i)
        end
    else
        @floop for i in cols
            @init storage = zeros(T, size(lu, 1))
            f!(storage, i)
            v = view(x, :, i)
            ldiv!(v, lu, storage)
            g!(storage, i)
        end
    end
    return x
end

@inline pfno(a...) = nothing

function _par_inv_init(x, i)
    x[i] = one(eltype(x))
    return
end
function _par_inv_fin(x, i)
    x[i] = zero(eltype(x))
    return
end

par_inv!(x, lu::LU; cols=1:size(lu, 2)) = par_solve_f!(x, _par_inv_init, _par_inv_fin, lu; cols)
par_inv(lu::LU{T}; cols=1:size(lu, 2)) where T = par_inv!(Matrix{T}(undef, size(lu)...), lu; cols)


using Base.Threads
using SparseArrays, LinearAlgebra
function par_ldiv!_t_f(lhs, F, rhs; cols)
    for i in cols
        ldiv!(view(lhs, :, i), F, view(rhs, :, i))
    end
    return
end

function par_ldiv!_t(lhs::AbstractMatrix{T}, F, rhs::AbstractMatrix{T}; cols=axes(rhs, 2)) where T
    size(lhs) == size(rhs) || error("rhs and lhs are not equal sized")
    c = ceil(Int, length(cols) / nthreads())
    foreach(wait, [
        @spawn par_ldiv!_t_f(lhs, duplicate(F), rhs;
            cols = view(cols, (i - 1) * c + 1: min(i * c, length(cols))))
            for i in 1:nthreads()])
    return lhs
end
par_ldiv_t(F, rhs; cols=axes(rhs, 2)) = par_ldiv!(similar(rhs), F, rhs; cols)

export par_ldiv!_t