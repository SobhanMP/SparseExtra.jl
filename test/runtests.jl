using Test, SparseArrays, SparseExtra
iternz_naive(x) = collect(zip(findnz(x)...))

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