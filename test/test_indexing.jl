
module TestIndexing

using MultidimensionalSparseArrays
using Test

@testset "Indexing" begin
    A = SparseArray{Float64, 2}((3, 3))

    # Setting values
    A[1, 1] = 5.0
    A[2, 3] = -2.5
    A[3, 2] = 1.0

    @test A[1, 1] == 5.0
    @test A[2, 3] == -2.5
    @test A[3, 2] == 1.0
    @test A[1, 2] == 0.0  # Default value
    @test A[2, 2] == 0.0  # Default value

    # CartesianIndex access
    @test A[CartesianIndex(1, 1)] == 5.0
    @test A[CartesianIndex(2, 2)] == 0.0

    A[CartesianIndex(2, 1)] = 3.0
    @test A[2, 1] == 3.0

    # Setting to default value should remove from storage
    A[1, 1] = 0.0
    @test A[1, 1] == 0.0
    @test nnz(A) == 3  # Should be one less stored element

    # Bounds checking
    @test_throws BoundsError A[0, 1]
    @test_throws BoundsError A[4, 1]
    @test_throws BoundsError A[1, 4]
end

end
