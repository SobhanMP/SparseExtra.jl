"""
    Work structure for `dijkstra`
"""
struct DijkstraState{T<:Real,U<:Integer}
    parent::Vector{U}
    distance::Vector{T}
    visited::Vector{Bool}
    q::PriorityQueue{U,T,FasterForward}
end
function set_src(state::DijkstraState{T, U}, srcs) where {T, U}
    for src in srcs
        state.distance[src] = zero(T)
        state.q[src] = zero(T)
    end
end
"""
    DijkstraState(nv, T, src::AbstractVector{U}) where U -> DijkstraState

Initialize a `DijkstraState` for a graph of size `nv`, weight of type `T`, and srcs as initiale points
"""
function DijkstraState(nv, T, srcs::AbstractVector{U}) where U
    state = DijkstraState{T,U}(
        zeros(U, nv),
        fill(typemax(T), nv),
        zeros(Bool, nv),
        PriorityQueue{U,T,FasterForward}(FasterForward()))
    set_src(state, srcs)
    return state
end


"""
    DijkstraState(::DijkstraState{T, U}, src) where {T, U} -> DijkstraState{T, U}
clean up the DijkstraState to reusue the memory and avoid reallocations
"""
function DijkstraState(state::DijkstraState{T, U}, srcs) where {T, U}
    fill!(state.parent, zero(U))
    fill!(state.distance, typemax(T))
    fill!(state.visited, false)
    empty!(state.q)
    set_src(state, srcs)
    return state
end


"""
    dijkstra(distmx::AbstractSparseMatrixCSC, state::DijkstraState, [target]) -> Nothing

Given a `distmx` such that `distmx[j, i]` is the distance of the arc i→j and strucutral zeros mean no arc, run the dijkstra algorithm until termination or reaching target (whichever happens first).
"""
function dijkstra(
    distmx::SparseArrays.AbstractSparseMatrixCSC{T},
    state::DijkstraState{T,U},
    target=nothing) where {T<:Real, U<:Integer}
    nzval = distmx.nzval
    rowval = distmx.rowval
    cptr = distmx.colptr
    visited, distance, parent, q = state.visited, state.distance, state.parent, state.q
    target !== nothing && visited[target] && return
    @inbounds while !isempty(state.q)
        peek(state.q) == target && break
        u = dequeue!(state.q)

        d = distance[u]
        visited[u] && continue
        visited[u] = true
        for r in cptr[u]:cptr[u+1]-1
            v = rowval[r]
            δ = nzval[r]
            alt = d + δ
            if !visited[v] && alt < distance[v]
                parent[v] = u
                distance[v] = alt
                q[v] = alt
            end
        end
    end
end
"""
    extract_path(::DijkstraState{T, U}, d) -> U[]

return the path found to `d`.
"""
function extract_path(state::DijkstraState{T, U}, d) where {T, U}
    r = U[]
    c = d
    while c != 0
        push!(r, c)
        c = state.parent[c]
    end
    return r
end

"""
    Path2Edge(x, [e=length(x)])

[Unbenchmarked] faster version of `zip(view(x, 1:e), view(x, 2:e))`. Only works on 1-index based arrays with unit strides
"""
struct Path2Edge{T,V<:AbstractVector{T}}
    x::V
    e::Int
    function Path2Edge(x::AbstractVector, e=length(x)) where T
        @assert length(x) >= e
        new{eltype(x),typeof(x)}(x, e)
    end
end

Base.length(p::Path2Edge) = p.e - 1
Base.eltype(::Path2Edge{T}) where {T} = NTuple{2,T}
Base.iterate(p::Path2Edge, state=1) =
    if p.e <= state
        nothing
    else
        (p.x[state], p.x[state+1]), state + 1
    end

"""
    path_cost(::AbstractSparseMatrixCSC, r, n)
return the total weight  of the path r[1:n]
"""
function path_cost(w::SparseMatrixCSC{T}, r, n=length(r)) where T
    cost = zero(T)
    for (i, j) in Path2Edge(r, n)
        cost += w[i, j]
    end
    cost
end
"""
    path_cost(f::Function, ::AbstractSparseMatrixCSC, r, n)
return the total weight of the path r[1:n], with `f` applied to each weight before summation
"""
function path_cost(f::F, w::SparseMatrixCSC{T}, r, n=length(r)) where {F, T}
    cost = zero(f(zero(T)))
    for (i, j) in Path2Edge(r, n)
        cost += f(w[i, j])
    end
    cost
end
"""
    path_cost_nt(::AbstractSparseMatrixCSC, r, n)
like `path_cost`, but for matrices that are transposed
"""
function path_cost_nt(w::SparseMatrixCSC{T}, r, n=length(r)) where T
    cost = zero(T)
    for (i, j) in Path2Edge(r, n)
        cost += w[j, i]
    end
    cost
end

"""
    path_cost_nt(f::Function, ::AbstractSparseMatrixCSC, r, n)
like `path_cost`, but for matrices that are transposed
"""
function path_cost_nt(f, w::SparseMatrixCSC{T}, r, n=length(r)) where T
    cost = zero(T)
    for (i, j) in Path2Edge(r, n)
        cost += f(w[j, i])
    end
    cost
end

