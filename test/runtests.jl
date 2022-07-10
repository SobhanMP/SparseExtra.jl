using Test, SparseArrays, SparseExtra, LinearAlgebra
using SparseArrays: getnzval, getcolptr, getrowval
using Graphs
using Graphs.Experimental.ShortestPaths: shortest_paths, dists
using StaticArrays
using VectorizationBase


@testset "simd argmax" begin
    function f(t, i)
        tx = ntuple(_ -> rand(t), i)
        ax = collect(tx)
        vx = Vec{i, t}(tx...)
        @test argmax(vx) == argmax(ax)
        
        return
    end

    for t in [Int8, Int32, Int64, Float64, Float32],
        i in 1:100,
        j in 1:100
        f(Int8, i)
    end
end

@testset "dijkstra" begin
    graph = Graphs.SimpleGraphs.erdos_renyi(100, 0.4; is_directed=true)
    state = DijkstraState(nv(graph), Float64, SVector{0, Int}())
    weight = spzeros(nv(graph), nv(graph))
    for i in vertices(graph),
        j in outneighbors(graph, i)
        weight[i, j] = rand() + 1e-3
    end
    tw = sparse(transpose(weight))
    for src in vertices(graph)
        state = DijkstraState(state, SVector(src))
        dijkstra(tw, state)
        odj = shortest_paths(graph, src, transpose(tw))
        @test dists(odj) ≈ state.distance
        for j in vertices(graph)
            i = state.parent[j]
            i == 0 && continue
            
            @test has_edge(graph, i, j)
            @test state.distance[i] + tw[j, i] ≈ state.distance[j]
        end
    end
 end

@testset "sparse_like" begin
    function good_same(t, z, eq=true)
        @test size(z) == size(t)
        @test getcolptr(z) === getcolptr(t)
        @test getrowval(z) === getrowval(t)
        eq && @test getnzval(z) == getnzval(t)
        @test getnzval(z) !== getnzval(t)
    end
    t = sprandn(100, 100, 0.1)
    good_same(t, sparse_like(t))
    good_same(t, sparse_like(t, rand(1:2, nnz(t))), false)
    good_same(t, sparse_like(t, 0), false)
    all(getnzval(sparse_like(t, 2)) .== 2)
    good_same(t, sparse_like(t, Bool), false)
    @test isa(getnzval(sparse_like(t, Bool)), Vector{Bool})
end

iternz_naive(x) = let f = findnz(x)
    collect(zip(f[end], f[1:end-1]...))
end

@testset "iternz (SparseMatrixCSC)" begin
    for i in 1:20
        A = sprandn(100, 100, 1 / i)
        @test collect(iternz(A)) == iternz_naive(A)
    end
end

@testset "iternz (SparseVector)" begin
    for i in 1:10
        A = sprandn(100, i / 100)
        @test collect(iternz(A)) == iternz_naive(A)
    end
end

@testset "iternz (Vectors)" begin
    for i in 1:10
        a = randn(100 + i)
        @test collect(iternz(a)) == collect(zip(a, eachindex(a)))
    end
end

test_iternz_arr(_a::AbstractArray{T, N}, it=iternz(_a)) where {T, N} = begin
    a = copy(_a)
    x = collect(it)
    @test length(unique(x)) <= length(a)
    @test length(unique(x)) == length(x)
    @test all((a[i...] == v for (v, i...) in x))
    seen = Set{NTuple{N, Int}}()
    for (_, i...) in x
        @test i ∉ seen
        push!(seen, i)
    end
    for I in CartesianIndices(a)
        i = Tuple(I)
        @test iszero(a[I]) || i ∈ seen
    end
end

@testset "iternz (Array)" begin
    for i in 1:5
        a = randn((rand(3:4) for i in 1:i)...)
        test_iternz_arr(a)
    end
    a = randn(10)
    test_iternz_arr(a)
    test_iternz_arr(view(a, 3:6))
end


@testset "iternz (Diagonal)" begin
    for i in 1:10
        test_iternz_arr(Diagonal(randn(i)))
        test_iternz_arr(Diagonal(sprandn(i, 0.5)))
    end
end

@testset "iternz (Upper/Lower Triangular)" begin
    for i in 1:10
        test_iternz_arr(UpperTriangular(randn(i, i)))
        test_iternz_arr(UpperTriangular(sprandn(i, i, 0.5)))
        test_iternz_arr(LowerTriangular(randn(i, i)))
        test_iternz_arr(LowerTriangular(sprandn(i, i, 0.5)))
    end
end

@testset "unsafe sum" begin
    for s in 1:100
        a = randn(s)
        for i in 1:s,
            j in i:s
            @test unsafe_sum(a, i, j) == sum(view(a, i:j))
        end
    end
end

@testset "unsafe sum" begin
    for s in 1:100
        a = sprandn(s, s, 0.1)
        getnzval(a) .= abs.(getnzval(a))
        a = a + I
        b = copy(a)
        colnorm!(a)
        for i in axes(a, 2)
           @test sum(a[:, i]) ≈ 1
           @test colsum(a, i) ≈ sum(a[:, i])
           @test colsum(b, i) ≈ sum(b[:, i])
           @test a[:, i] .* sum(b[:, i]) ≈ b[:, i]
        end
    end
end

@testset "par solve" begin
    for s in 1:100
        a = sprandn(s, s, 0.1) + I
        b = randn(s, s)
        @test maximum(par_inv(lu(a)) * a - I) <= 1e-5
        @test a \ b ≈ par_solve(lu(a), b)
    end
end

@testset "Path2Edge" begin
    for _ in 1:10
        a = rand(1:100, 1000)
        for i in [10, 200, 1000]
            @test collect(Path2Edge(a, i)) == collect(zip(a, view(a, 2:i)))
        end
    end
end

