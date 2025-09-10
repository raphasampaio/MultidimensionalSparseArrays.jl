module TestMemoryPerformance

using MultidimensionalSparseArrays
using Test

@testset "Memory Management and Performance Edge Cases" begin
    @testset "Memory Efficiency with Large Sparse Arrays" begin
        # Test that large sparse arrays don't consume excessive memory
        A = SparseArray{Float64, 2}((10^4, 10^4))  # 100M potential elements
        @test size(A) == (10^4, 10^4)
        @test length(A) == 10^8
        @test nnz(A) == 0
        
        # Add a few elements - memory should remain minimal
        A[1, 1] = 1.0
        A[5000, 5000] = 2.0
        A[9999, 9999] = 3.0
        
        @test nnz(A) == 3
        @test A[1, 1] == 1.0
        @test A[5000, 5000] == 2.0
        @test A[9999, 9999] == 3.0
        
        # Sparsity should be very high
        @test sparsity(A) > 0.99999
    end
    
    @testset "Memory Cleanup with dropstored!" begin
        A = SparseArray{Int, 2}((100, 100))
        
        # Fill with pattern, then clean up
        for i in 1:50
            A[i, i] = i
            A[i, i+50] = i + 100
        end
        @test nnz(A) == 100
        
        # Remove all values equal to specific numbers
        dropstored!(A, 25)
        @test nnz(A) == 99
        @test !hasindex(A, 25, 25)
        
        # Remove all values in a range
        for val in 1:10
            dropstored!(A, val)
        end
        @test nnz(A) <= 90  # Should have removed at least 10 elements
        
        # Test removing non-existent values doesn't break anything
        dropstored!(A, 999)
        @test nnz(A) <= 90  # Should be unchanged
    end
    
    @testset "Memory Cleanup with compress!" begin
        A = SparseArray{Float64, 2}((50, 50))
        
        # Add various values including zeros
        A[1, 1] = 5.0
        A[2, 2] = 0.0
        A[3, 3] = -2.5
        A[4, 4] = 0.0
        A[5, 5] = 1.0
        
        @test nnz(A) == 5
        
        # Compress should remove stored zeros
        compress!(A)
        @test nnz(A) == 3  # Should remove the two zeros
        @test A[1, 1] == 5.0
        @test A[3, 3] == -2.5
        @test A[5, 5] == 1.0
        @test !hasindex(A, 2, 2)
        @test !hasindex(A, 4, 4)
    end
    
    @testset "Copy Performance and Memory Independence" begin
        A = SparseArray{Int, 2}((100, 100))
        
        # Add diagonal pattern
        for i in 1:100
            A[i, i] = i^2
        end
        
        # Test copy creates independent memory
        B = copy(A)
        @test A == B
        @test A.data !== B.data  # Different memory
        @test nnz(A) == nnz(B)
        
        # Modifying one shouldn't affect the other
        A[1, 2] = 999
        @test hasindex(A, 1, 2)
        @test !hasindex(B, 1, 2)
        @test A != B
        
        # Test copy preserves all values
        for i in 1:100
            @test B[i, i] == i^2
        end
    end
    
    @testset "Performance with Different Access Patterns" begin
        A = SparseArray{Float64, 2}((1000, 1000))
        
        # Test diagonal access pattern (should be efficient)
        for i in 1:100:1000
            A[i, i] = Float64(i)
        end
        @test nnz(A) == 10
        
        # Test row-wise access pattern
        for j in 1:10
            A[100, j*10] = Float64(j)
        end
        @test nnz(A) == 20  # 10 original + 10 new
        
        # Test column-wise access pattern
        for i in 1:10
            A[i*10, 200] = Float64(i + 100)
        end
        @test nnz(A) == 30
        
        # Test pseudo-random access pattern  
        for k in 1:50
            i = (k * 37) % 1000 + 1  # Pseudo-random pattern
            j = (k * 71) % 1000 + 1
            A[i, j] = Float64(k)
        end
        @test nnz(A) >= 30  # At least the previous entries
    end
    
    @testset "Memory Usage with fill! Operations" begin
        A = SparseArray{Int, 2}((100, 100))
        
        # Initially empty
        @test nnz(A) == 0
        
        # fill! with non-zero value should store everywhere
        fill!(A, 42)
        @test nnz(A) == 10000  # 100 * 100
        @test all(A[i, j] == 42 for i in 1:100, j in 1:100)
        
        # fill! with zero should still store everywhere (by design)
        fill!(A, 0)
        @test nnz(A) == 10000
        @test all(A[i, j] == 0 for i in 1:100, j in 1:100)
        
        # Use compress! to clean up zeros
        compress!(A)
        @test nnz(A) == 0
    end
    
    @testset "Memory Behavior with Arithmetic Operations" begin
        A = SparseArray{Float64, 2}((500, 500))
        B = SparseArray{Float64, 2}((500, 500))
        
        # Create sparse diagonal matrices
        for i in 1:50:500
            A[i, i] = Float64(i)
            B[i, i] = Float64(i * 2)
        end
        
        # Addition should preserve sparsity
        C = A + B
        @test nnz(C) == nnz(A)  # Same number of stored elements
        @test nnz(C) == nnz(B)
        
        # Subtraction might create zeros
        D = A - B
        for i in 1:50:500
            if A[i, i] == B[i, i]
                @test !hasindex(D, i, i)  # Zero not stored
            else
                @test hasindex(D, i, i)
            end
        end
        
        # Scalar multiplication should preserve sparsity structure
        E = A * 2.0
        @test nnz(E) == nnz(A)
        
        # Multiplication by zero should create empty array
        F = A * 0.0
        @test nnz(F) == 0
    end
    
    @testset "Performance with High-Dimensional Arrays" begin
        # Test 4D array
        A = SparseArray{Int, 4}((10, 10, 10, 10))  # 10,000 potential elements
        
        # Add sparse pattern
        for i in 1:2:10
            A[i, i, i, i] = i^4
        end
        @test nnz(A) == 5  # Only 5 elements stored
        
        # Test access performance
        for i in 1:2:10
            @test A[i, i, i, i] == i^4
        end
        
        # Test that unset indices behave correctly
        @test_throws BoundsError A[2, 2, 2, 2]
    end
    
    @testset "Memory Patterns with Similar Arrays" begin
        A = SparseArray{Float64, 2}((100, 100))
        A[50, 50] = 3.14
        
        # similar should create empty array with same structure
        B = similar(A)
        @test size(B) == size(A)
        @test eltype(B) == eltype(A)
        @test nnz(B) == 0  # Should be empty
        @test !hasindex(B, 50, 50)
        
        # similar with different type
        C = similar(A, Int)
        @test size(C) == size(A)
        @test eltype(C) == Int
        @test nnz(C) == 0
        
        # similar with different dimensions
        D = similar(A, Float32, (50, 75))
        @test size(D) == (50, 75)
        @test eltype(D) == Float32
        @test nnz(D) == 0
    end
    
    @testset "Stress Test: Repeated Operations" begin
        A = SparseArray{Int, 2}((100, 100))
        
        # Repeatedly add and remove elements
        for iteration in 1:100
            # Add elements
            for i in 1:10
                A[i, iteration % 100 + 1] = iteration
            end
            
            # Remove some elements
            if iteration > 50
                dropstored!(A, iteration - 50)
            end
        end
        
        # Array should still be functional
        @test nnz(A) > 0
        @test size(A) == (100, 100)
        
        # Test that we can still access and modify
        A[99, 99] = 12345
        @test A[99, 99] == 12345
    end
    
    @testset "Memory Efficiency: Constructor Patterns" begin
        # Test construction from dense arrays with different sparsity levels
        
        # Very sparse pattern
        dense1 = zeros(Int, 100, 100)
        dense1[1, 1] = 1
        dense1[100, 100] = 2
        
        sparse1 = SparseArray(dense1)
        @test nnz(sparse1) == 2
        @test sparsity(sparse1) > 0.99
        
        # Medium sparse pattern
        dense2 = zeros(Int, 50, 50)
        for i in 1:5:50
            dense2[i, i] = i
        end
        
        sparse2 = SparseArray(dense2)
        @test nnz(sparse2) == 10
        @test sparsity(sparse2) == 0.996  # (2500 - 10) / 2500
        
        # Test with tolerance
        dense3 = zeros(Float64, 20, 20)
        dense3[5, 5] = 1e-16  # Very small value
        dense3[10, 10] = 1.0
        
        sparse3 = SparseArray(dense3, atol=1e-15)
        @test nnz(sparse3) == 1  # Small value should be ignored
        @test hasindex(sparse3, 10, 10)
        @test !hasindex(sparse3, 5, 5)
    end
    
    @testset "Edge Case: Empty Operations" begin
        A = SparseArray{Float64, 2}((10, 10))
        B = SparseArray{Float64, 2}((10, 10))
        
        # Operations on empty arrays should remain empty
        C = A + B
        @test nnz(C) == 0
        
        D = A - B
        @test nnz(D) == 0
        
        E = A * 5.0
        @test nnz(E) == 0
        
        # Copy of empty array
        F = copy(A)
        @test nnz(F) == 0
        @test A == F
    end
end

end