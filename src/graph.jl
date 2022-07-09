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
reuse dijkstra state
"""
function DijkstraState(state::DijkstraState{T, U}, srcs) where {T, U}
    fill!(state.parent, zero(U))
    fill!(state.distance, typemax(T))
    fill!(state.visited, false)
    empty!(state.q)
    set_src(state, srcs)
    return state
end

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

function extract_path(state::DijkstraState{T, U}, d) where {T, U}
    r = U[]
    c = d
    while c != 0
        push!(r, c)
        c = state.parent[c]
    end
    return r
end
