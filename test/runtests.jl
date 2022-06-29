using Test, SparseArrays, SparseExtra
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

test_iternz_arr(a, it) = begin
    x = collect(it)
    @test length(unique(x)) == length(a)
    @test all((a[i...] == v for (v, i...) in x))
end

@testset "iternz (Array)" begin
    for i in 1:5
        a = randn((rand(3:4) for i in 1:10)...)
        test_iternz_arr(a, iternz(a))
    end
end