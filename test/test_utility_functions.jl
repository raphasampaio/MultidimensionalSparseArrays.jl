
module TestUtilityFunctions

using NDimensionalSparseArrays
using Test

@testset "Utility Functions" begin
    A = NDSparseArray{Float64, 2}((4, 4))
    A[1, 1] = 1.0
    A[2, 3] = 2.0
    A[4, 4] = 3.0

    @test nnz(A) == 3
    @test sparsity(A) ≈ 13/16  # 13 zeros out of 16 elements

    # Test stored indices
    indices = collect(stored_indices(A))
    @test length(indices) == 3
    @test CartesianIndex(1, 1) in indices
    @test CartesianIndex(2, 3) in indices
    @test CartesianIndex(4, 4) in indices

    # Test stored values
    values = collect(stored_values(A))
    @test length(values) == 3
    @test 1.0 in values
    @test 2.0 in values
    @test 3.0 in values

    # Test stored pairs
    pairs_dict = Dict(stored_pairs(A))
    @test pairs_dict[CartesianIndex(1, 1)] == 1.0
    @test pairs_dict[CartesianIndex(2, 3)] == 2.0
    @test pairs_dict[CartesianIndex(4, 4)] == 3.0
end
end
