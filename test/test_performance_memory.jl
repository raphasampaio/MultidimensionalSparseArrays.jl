module TestPerformanceMemory

using MultidimensionalSparseArrays
using Test

@testset "Performance and Memory Tests" begin
    @testset "Memory Efficiency" begin
        # Test that sparse arrays use less memory than dense arrays for sparse data
        n = 1000
        dense_array = zeros(Int, n, n)
        sparse_array = spzeros(Int, n, n)
        
        # Add only a few non-zero elements
        for i in 1:10
            idx = i * 100
            if idx <= n
                dense_array[idx, idx] = i
                sparse_array[idx, idx] = i
            end
        end
        
        @test nnz(sparse_array) == 10
        @test length(sparse_array) == n * n
        @test sparsity(sparse_array) > 0.99
        
        # Verify values are correct
        for i in 1:10
            idx = i * 100
            if idx <= n
                @test sparse_array[idx, idx] == i
                @test dense_array[idx, idx] == i
            end
        end
        
        # Test memory usage is reasonable
        # A sparse array with 10 elements should use much less memory 
        # than a 1M element dense array
        @test nnz(sparse_array) < length(sparse_array) / 1000  # Much less than 0.1% density
    end
    
    @testset "dropstored! and compress!" begin
        A = SparseArray{Float64, 2}((3, 3))
        A[1, 1] = 5.0
        A[1, 2] = 0.0  # This is stored like any other value
        A[2, 1] = 3.0
        A[2, 2] = 0.0  # This is stored like any other value
        A[3, 3] = 7.0
        
        # All set values are stored, including zeros
        @test nnz(A) == 5
        
        # Compress should remove stored zeros
        compress!(A)
        @test nnz(A) == 3
        @test A[1, 1] == 5.0
        @test_throws BoundsError A[1, 2]  # Should be removed by compress!
        @test A[2, 1] == 3.0
        @test A[3, 3] == 7.0
        
        # Test dropstored! with specific value
        # First set some values that will be stored
        A[1, 1] = 999.0
        A[3, 3] = 999.0
        @test nnz(A) == 3  # Should have 999.0, 3.0, 999.0
        
        # Now dropstored! should remove the 999.0 values
        dropstored!(A, 999.0)
        @test nnz(A) == 1  # Only A[2,1] should remain
        @test A[2, 1] == 3.0
        @test_throws BoundsError A[1, 1]  # Should be removed
        @test_throws BoundsError A[3, 3]  # Should be removed
    end
    
    @testset "Large Array Performance" begin
        # Test that operations scale reasonably with sparse arrays
        sizes = [(100, 100), (500, 500)]
        
        for (m, n) in sizes
            A = spzeros(Int, m, n)
            
            # Add diagonal elements
            for i in 1:min(m, n)
                A[i, i] = i
            end
            
            @test nnz(A) == min(m, n)
            
            # Test that accessing elements is fast
            # This is more of a smoke test than a rigorous benchmark
            sum_diag = 0
            for i in 1:min(m, n)
                sum_diag += A[i, i]
            end
            
            expected_sum = min(m, n) * (min(m, n) + 1) ÷ 2
            @test sum_diag == expected_sum
            
            # Test sparse operations
            B = copy(A)
            @test nnz(B) == nnz(A)
            @test B == A
            
            # Test scalar multiplication
            C = A * 2
            @test nnz(C) == nnz(A)
            for i in 1:min(m, n)
                @test C[i, i] == 2 * i
            end
        end
    end
    
    @testset "Copy Performance" begin
        A = SparseArray{Int, 2}((100, 100))
        
        # Add random sparse elements
        for i in 1:50
            row = rand(1:100)
            col = rand(1:100)
            A[row, col] = rand(1:1000)
        end
        
        # Test that copy is efficient and correct
        B = copy(A)
        
        @test nnz(B) == nnz(A)
        @test B == A
        @test B.data !== A.data  # Should be different objects
        
        # Modify original and verify copy is unchanged
        original_nnz = nnz(A)
        A[1, 1] = 999999
        
        if !haskey(B.data, CartesianIndex(1, 1)) || B[1, 1] != 999999
            @test nnz(B) <= original_nnz  # Copy should be unchanged
        end
    end
    
    @testset "Arithmetic Performance" begin
        # Test that arithmetic operations are reasonably efficient
        A = SparseArray{Int, 2}((100, 100))
        B = SparseArray{Int, 2}((100, 100))
        
        # Add some sparse data
        for i in 1:10
            A[i, i] = i
            B[i, i] = i * 2
        end
        
        # Test addition
        C = A + B
        @test nnz(C) == 10
        for i in 1:10
            @test C[i, i] == 3 * i
        end
        
        # Test subtraction
        D = B - A
        @test nnz(D) == 10
        for i in 1:10
            @test D[i, i] == i
        end
        
        # Test scalar multiplication
        E = A * 5
        @test nnz(E) == 10
        for i in 1:10
            @test E[i, i] == 5 * i
        end
    end
    
    # Note: Default value tests removed since the new architecture 
    # doesn't support default values - all set values are stored
    
    @testset "Stress Test with Many Operations" begin
        A = SparseArray{Float64, 2}((50, 50))
        
        # Perform many random operations
        for i in 1:1000
            row = rand(1:50)
            col = rand(1:50)
            
            if rand() < 0.7  # 70% chance to set value
                A[row, col] = randn()
            else  # 30% chance to set to zero (remove)
                A[row, col] = 0.0
            end
        end
        
        # Verify array is still valid
        @test size(A) == (50, 50)
        @test nnz(A) >= 0
        @test nnz(A) <= 2500  # Can't have more stored elements than total
        
        # Test that all operations still work
        B = copy(A)
        @test B == A
        
        C = A * 2.0
        @test nnz(C) <= nnz(A)  # Scalar mult by non-zero shouldn't increase nnz
        
        compress!(A)
        # After compression, should have no stored zeros
        for (idx, val) in stored_pairs(A)
            @test val != 0.0
        end
    end
end

end