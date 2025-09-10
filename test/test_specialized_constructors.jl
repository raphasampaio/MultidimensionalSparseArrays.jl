module TestSpecializedConstructors

using MultidimensionalSparseArrays
using Test

@testset "Specialized Constructors" begin
    @testset "spzeros" begin
        # Test basic spzeros
        A = spzeros(Int, 3, 4)
        @test size(A) == (3, 4)
        @test eltype(A) == Int
        @test nnz(A) == 0
        @test A.default_value == 0
        @test A[1, 1] == 0
        @test A[3, 4] == 0
        
        # Test with Float64 (default type)
        B = spzeros(2, 3)
        @test eltype(B) == Float64
        @test size(B) == (2, 3)
        @test nnz(B) == 0
        
        # Test with tuple dimensions
        C = spzeros(Float32, (2, 2, 2))
        @test eltype(C) == Float32
        @test size(C) == (2, 2, 2)
        @test ndims(C) == 3
        
        # Test setting values
        A[1, 1] = 5
        @test A[1, 1] == 5
        @test nnz(A) == 1
    end
    
    @testset "spones" begin
        # Test basic spones
        A = spones(Int, 2, 2)
        @test size(A) == (2, 2)
        @test eltype(A) == Int
        @test nnz(A) == 4  # All elements should be 1
        @test A[1, 1] == 1
        @test A[1, 2] == 1
        @test A[2, 1] == 1
        @test A[2, 2] == 1
        
        # Test with Float64 (default)
        B = spones(3, 2)
        @test eltype(B) == Float64
        @test size(B) == (3, 2)
        @test nnz(B) == 6
        @test all(B[i, j] == 1.0 for i in 1:3, j in 1:2)
        
        # Test with complex numbers
        C = spones(Complex{Float64}, 2, 2)
        @test eltype(C) == Complex{Float64}
        @test all(C[i, j] == 1.0 + 0.0im for i in 1:2, j in 1:2)
    end
    
    @testset "spfill" begin
        # Test filling with non-zero value
        A = spfill(5, 2, 3)
        @test size(A) == (2, 3)
        @test eltype(A) == Int
        @test nnz(A) == 6  # All elements should be stored
        @test all(A[i, j] == 5 for i in 1:2, j in 1:3)
        
        # Test filling with zero (should result in empty sparse array)
        B = spfill(0.0, 3, 3)
        @test nnz(B) == 0
        @test all(B[i, j] == 0.0 for i in 1:3, j in 1:3)
        
        # Test with different types
        C = spfill(2.5, 2, 2)
        @test eltype(C) == Float64
        @test all(C[i, j] == 2.5 for i in 1:2, j in 1:2)
        
        # Test with negative values
        D = spfill(-3, 1, 4)
        @test all(D[1, j] == -3 for j in 1:4)
        @test nnz(D) == 4
    end
    
    @testset "fill!" begin
        A = SparseArray{Int, 2}((3, 3))
        A[1, 1] = 5
        A[2, 2] = 10
        @test nnz(A) == 2
        
        # Fill with non-default value
        fill!(A, 7)
        @test nnz(A) == 9  # All positions should be stored
        @test all(A[i, j] == 7 for i in 1:3, j in 1:3)
        
        # Fill with default value (should clear all)
        fill!(A, 0)
        @test nnz(A) == 0
        @test all(A[i, j] == 0 for i in 1:3, j in 1:3)
        
        # Test return value
        B = fill!(A, 3)
        @test B === A
        @test A[1, 1] == 3
    end
    
    @testset "Dense Array Constructor with Tolerance" begin
        # Test without tolerance (exact comparison)
        dense = [1.0 0.0 3.0; 0.0 0.0 0.0; 2.0 0.0 4.0]
        A = SparseArray(dense)
        @test nnz(A) == 4
        @test A[1, 1] == 1.0
        @test A[1, 3] == 3.0
        @test A[3, 1] == 2.0
        @test A[3, 3] == 4.0
        
        # Test with tolerance for floating point
        dense_noisy = [1.0 1e-15 3.0; 1e-16 0.0 0.0; 2.0 0.0 4.0]
        B = SparseArray(dense_noisy, atol=1e-14)
        @test nnz(B) == 4  # Small values should be treated as zero
        @test B[1, 1] == 1.0
        @test B[1, 2] == 0.0  # 1e-15 should be treated as zero
        @test B[2, 1] == 0.0  # 1e-16 should be treated as zero
        
        # Test with larger tolerance
        dense_approx = [1.0 0.001 3.0; 0.0005 0.0 0.0; 2.0 0.0 4.0]
        C = SparseArray(dense_approx, atol=0.01)
        @test nnz(C) == 4  # Values < 0.01 should be treated as zero
        
        # Test with integer arrays (should use exact comparison)
        dense_int = [1 0 3; 0 0 0; 2 0 4]
        D = SparseArray(dense_int, atol=0.5)  # atol should be ignored for integers
        @test nnz(D) == 4
        @test eltype(D) == Int
    end
    
    @testset "Constructor Edge Cases" begin
        # Test with empty dimensions
        A = spzeros(0, 0)
        @test size(A) == (0, 0)
        @test length(A) == 0
        @test nnz(A) == 0
        
        # Test with single dimension
        B = spzeros(5)
        @test size(B) == (5,)
        @test ndims(B) == 1
        
        # Test with very large dimensions (should not allocate much memory)
        C = spzeros(1000, 1000)
        @test size(C) == (1000, 1000)
        @test nnz(C) == 0
        # Memory usage should be minimal since it's sparse
    end
end

end