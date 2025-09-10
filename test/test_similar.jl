module TestSimilar

using MultidimensionalSparseArrays
using Test

    
    @testset "Similar" begin
        A = SparseArray{Float64, 2}((3, 4))
        A[1, 1] = 5.0
        
        B = similar(A)
        @test size(B) == size(A)
        @test eltype(B) == eltype(A)
        @test nnz(B) == 0  # Should be empty
        @test B.default_value == A.default_value
        
        C = similar(A, Int)
        @test size(C) == size(A)
        @test eltype(C) == Int
        @test nnz(C) == 0
        @test C.default_value == 0
    end
end