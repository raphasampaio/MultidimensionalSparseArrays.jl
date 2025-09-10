using Test
using MultidimensionalSparseArrays

@testset "SparseArray Tests" begin
    
    @testset "Construction" begin
        # Basic construction
        A = SparseArray{Float64, 2}((3, 4))
        @test size(A) == (3, 4)
        @test eltype(A) == Float64
        @test nnz(A) == 0
        @test A.default_value == 0.0
        
        # Construction with custom default value
        B = SparseArray{Int, 2}((2, 2), -1)
        @test B.default_value == -1
        @test B[1, 1] == -1  # Should return default value
        
        # Convenience constructors
        C = SparseArray{Float64}((3, 3))
        @test size(C) == (3, 3)
        @test eltype(C) == Float64
        
        D = SparseArray{Int}(2, 3, 4)
        @test size(D) == (2, 3, 4)
        @test eltype(D) == Int
        
        # Construction from dense array
        dense = [1 0 3; 0 0 0; 2 0 4]
        E = SparseArray(dense)
        @test size(E) == (3, 3)
        @test E[1, 1] == 1
        @test E[1, 2] == 0
        @test E[1, 3] == 3
        @test E[2, 1] == 0
        @test E[3, 1] == 2
        @test E[3, 3] == 4
        @test nnz(E) == 4  # Only non-zero elements stored
    end
    
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
    
    @testset "Multidimensional Arrays" begin
        # 3D array
        A = SparseArray{Float64, 3}((2, 2, 2))
        A[1, 1, 1] = 1.0
        A[2, 2, 2] = 8.0
        
        @test A[1, 1, 1] == 1.0
        @test A[2, 2, 2] == 8.0
        @test A[1, 2, 1] == 0.0
        @test nnz(A) == 2
        
        # 4D array
        B = SparseArray{Int, 4}((2, 2, 2, 2))
        B[1, 1, 1, 1] = 100
        @test B[1, 1, 1, 1] == 100
        @test nnz(B) == 1
    end
    
    @testset "Utility Functions" begin
        A = SparseArray{Float64, 2}((4, 4))
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
    
    @testset "Equality and Copying" begin
        A = SparseArray{Int, 2}((3, 3))
        A[1, 1] = 5
        A[2, 2] = 10
        
        B = SparseArray{Int, 2}((3, 3))
        B[1, 1] = 5
        B[2, 2] = 10
        
        @test A == B
        
        # Different values
        B[1, 1] = 6
        @test A != B
        
        # Different sizes
        C = SparseArray{Int, 2}((2, 2))
        @test A != C
        
        # Copy
        D = copy(A)
        @test A == D
        @test A.data !== D.data  # Different objects
        
        # Modify copy shouldn't affect original
        D[3, 3] = 15
        @test A != D
        @test A[3, 3] == 0
        @test D[3, 3] == 15
    end
    
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
        @test B[1] == 0
        @test nnz(B) == 1
        
        # Large sparse array
        C = SparseArray{Float64, 2}((1000, 1000))
        C[1, 1] = 1.0
        C[500, 500] = 2.0
        C[1000, 1000] = 3.0
        @test nnz(C) == 3
        @test sparsity(C) > 0.999  # Very sparse
    end
    
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