"""
solve lu\\b in parallel
"""
const LU{T} = Union{Transpose{UmfpackLU{T}}, UmfpackLU{T}}
function par_solve_f!(x, f!::F, lu::LU, b::AbstractMatrix; cols=axes(b, 2)) where F
    @floop for i in cols
        v = view(x, :, i)
        ldiv!(v, lu, view(b, :, i))
        f!(v)
    end
    x
end
function par_solve_f!(x, f!::F, g!::G, lu::LU{T}; cols=1:size(lu, 2)) where {T, F, G}
    size(x, 1) == size(lu, 1) || error("can't")
    @floop for i in cols
        @init storage = Vector{T}(undef, size(lu, 1))
        fill!(storage, zero(T))
        g!(storage, i)
        v = view(x, :, i)
        ldiv!(v, lu, storage)
        f!(v)
    end
    x
end

@inline pfno(a...) = nothing
par_solve!(x, lu, b; cols=axes(b, 2)) = (par_solve_f!(x, pfno, lu, b; cols); x)
par_solve(lu, b; cols=axes(b, 2)) = par_solve!(similar(b), lu, b; cols)

function _par_inv_init(x, i)
    x[i] = one(eltype(x))
    return
end
par_inv!(x, lu::LU; cols=1:size(lu, 2)) = par_solve_f!(x, pfno, _par_inv_init, lu; cols)
par_inv(lu::LU{T}; cols=1:size(lu, 2)) where T = par_inv!(Matrix{T}(undef, size(lu)...), lu; cols)
