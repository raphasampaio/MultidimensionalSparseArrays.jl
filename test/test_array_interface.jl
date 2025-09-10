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
        
        # Iteration
        values = collect(A)
        expected = [10 0 0; 0 20 0]  # 2D array layout (column-major)
        @test values == expected
        
        # Test flat iteration using linear indexing
        flat_values = [A[CartesianIndices(A)[i]] for i in 1:length(A)]
        expected_flat = [10, 0, 0, 20, 0, 0]  # Column-major order
        @test flat_values == expected_flat
    end
end