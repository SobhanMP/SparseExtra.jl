using Test, SparseArrays, SparseExtra, LinearAlgebra
using SparseArrays: getnzval, getcolptr, getrowval
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