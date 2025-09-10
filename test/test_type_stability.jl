module TestTypeStability

using MultidimensionalSparseArrays
using Test

@testset "Type Stability" begin
    A = SparseArray{Float64, 2}((2, 2))

    # Test that operations return correct types
    @test typeof(A[1, 1]) == Float64
    @test typeof(nnz(A)) == Int
    @test typeof(sparsity(A)) == Float64

    # Test with different element types
    B = SparseArray{Complex{Float64}, 2}((2, 2))
    B[1, 1] = 1.0 + 2.0im
    @test B[1, 1] == 1.0 + 2.0im
    @test typeof(B[1, 1]) == Complex{Float64}
end
end
