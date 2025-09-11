module TestSimilar

using NDimensionalSparseArrays
using Test

@testset "Similar" begin
    A = NDSparseArray{Float64, 2}((3, 4))
    A[1, 1] = 5.0

    B = similar(A)
    @test size(B) == size(A)
    @test eltype(B) == eltype(A)
    @test nnz(B) == 0  # Should be empty
    # Similar arrays start empty

    C = similar(A, Int)
    @test size(C) == size(A)
    @test eltype(C) == Int
    @test nnz(C) == 0
    # Similar arrays start empty
end
end
