module TestArrayInterface

using MultidimensionalSparseArrays
using Test

@testset "Array Interface" begin
    A = SparseArray{Int, 2}((2, 3))
    A[1, 1] = 10
    A[2, 2] = 20

    @test length(A) == 6
    @test ndims(A) == 2
    @test size(A) == (2, 3)
    @test size(A, 1) == 2
    @test size(A, 2) == 3

    # Iteration over stored values only
    values = collect(A)
    expected_stored = [10, 20]  # Only stored values
    @test Set(values) == Set(expected_stored)
    @test length(values) == 2

    # Test iteration over stored indices only
    stored_values = [A[idx] for idx in stored_indices(A)]
    @test Set(stored_values) == Set([10, 20])
    @test length(stored_values) == 2
end
end
