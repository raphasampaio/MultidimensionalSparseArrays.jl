module TestEdgeCases

using NDimensionalSparseArrays
using Test

@testset "Edge Cases" begin
    # Empty array
    A = SparseArray{Float64, 2}((0, 0))
    @test size(A) == (0, 0)
    @test length(A) == 0
    @test nnz(A) == 0

    # 1D array
    B = SparseArray{Int, 1}((5,))
    B[3] = 42
    @test B[3] == 42
    @test_throws BoundsError B[1]  # Unset index throws error
    @test nnz(B) == 1

    # Large sparse array
    C = SparseArray{Float64, 2}((1000, 1000))
    C[1, 1] = 1.0
    C[500, 500] = 2.0
    C[1000, 1000] = 3.0
    @test nnz(C) == 3
    @test sparsity(C) > 0.999  # Very sparse
end

end
