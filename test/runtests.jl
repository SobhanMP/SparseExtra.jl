using Test, SparseArrays, SparseExtra, LinearAlgebra
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

test_iternz_arr(a, it=iternz(a)) = begin
    x = collect(it)
    @test length(unique(x)) == length(a)
    @test all((a[i...] == v for (v, i...) in x))
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

test_iternz_diag(a, it) = begin
    x = collect(it)
    @test length(x) <= length(a.diag)
    @test length(unique(x)) == length(x)
    all = Set(1:length(a.diag))
    for (k, i, j) in x
        @test i == j
        @test a[i, j] == k
        delete!(all, i)
    end
    for i in all
        @test iszero(a[i, i])
    end
end

@testset "iternz (Diagonal)" begin
    for i in 1:10
        d = Diagonal(randn(i))
        test_iternz_diag(d, iternz(d))
    end
end