"""
solve lu\\b in parallel
"""
const LU = Union{Transpose{T, <: UmfpackLU{T}}, UmfpackLU{T}} where T

const LU_has_lock = hasfield(UmfpackLU, :lock)
if LU_has_lock
    if isdefined(UMFPACK, :duplicate)
        @eval begin
            using SparseArrays.UMFPACK: duplicate
            UMFPACK.duplicate(F::Transpose{T, <:UmfpackLU}) where T = Transpose(duplicate(F.parent))
            UMFPACK.duplicate(F::Adjoint{T, <:UmfpackLU}) where T = Adjoint(duplicate(F.parent))
            Base.copy(F::LU; copynumeric=false, copysymbolic=false) = duplicate(F)
        end
    end
else
    @eval begin
        SparseArrays._goodbuffers(S::AbstractSparseMatrixCSC) = _goodbuffers(size(S)..., getcolptr(S), getrowval(S), nonzeros(S))
        SparseArrays._checkbuffers(S::AbstractSparseMatrixCSC) = (@assert _goodbuffers(S); S)
    end
end

# Base.copy(F::LU; a...) = copy(F)
@inline pfno(a...) = nothing
"""
ldiv! but in parallel. use `cols`` to skip columns and `f(col, index)` to apply a thread safe function to that column.
"""
function par_solve!(x, lu::LU, b::AbstractMatrix; cols=axes(b, 2), f::F=nothing) where F
    if LU_has_lock
        @floop for i in cols
            @init dlu = copy(lu; copynumeric=false, copysymbolic=false)
            v = view(x, :, i)
            ldiv!(v, dlu, view(b, :, i))
            if f !== nothing f(v, i) end
        end
    else
        @floop for i in cols
            v = view(x, :, i)
            ldiv!(v, lu, view(b, :, i))
            if f !== nothing f(v, i) end
        end
    end
    return x
end
par_solve(lu::LU, b::AbstractMatrix; cols=axes(b, 2)) = par_solve!(similar(b), lu, b; cols)
"""
like par_solve but use `f` and `g` to create the column (and then destroy it).
"""
function par_solve_f!(x, f!::F, g!::G, lu::LU{T}; cols=1:size(lu, 2)) where {T, F, G}
    size(x, 1) == size(lu, 1) || error("can't")
    if LU_has_lock
        @floop for i in cols
            @init dlu = copy(lu; copynumeric=false, copysymbolic=false)
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



function _par_inv_init(x, i)
    x[i] = one(eltype(x))
    return
end
function _par_inv_fin(x, i)
    x[i] = zero(eltype(x))
    return
end

"""
return `lu \\ I`
"""
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
        @spawn par_ldiv!_t_f(lhs, copy(F; copynumeric=false, copysymbolic=false), rhs;
            cols = view(cols, (i - 1) * c + 1: min(i * c, length(cols))))
            for i in 1:nthreads()])
    return lhs
end
par_ldiv_t(F, rhs; cols=axes(rhs, 2)) = par_ldiv!(similar(rhs), F, rhs; cols)

export par_ldiv!_t