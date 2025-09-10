module TestLinearIndexing

using MultidimensionalSparseArrays
using Test

@testset "Linear Indexing" begin
    @testset "2D Array Linear Indexing" begin
        A = SparseArray{Int, 2}((2, 3))
        
        # Test setting values with linear indexing
        A[1] = 10  # (1, 1)
        A[3] = 20  # (1, 2) - column major
        A[6] = 30  # (2, 3)
        
        @test A[1, 1] == 10
        @test A[1, 2] == 20
        @test A[2, 3] == 30
        @test nnz(A) == 3
        
        # Test getting values with linear indexing
        @test A[1] == 10
        @test A[2] == 0   # (2, 1) should be zero
        @test A[3] == 20
        @test A[6] == 30
        
        # Test bounds checking
        @test_throws BoundsError A[0]
        @test_throws BoundsError A[7]
        
        # Test setting to default value removes entry
        A[1] = 0
        @test A[1] == 0
        @test nnz(A) == 2
    end
    
    @testset "3D Array Linear Indexing" begin
        B = SparseArray{Float64, 3}((2, 2, 2))
        
        # Test 3D linear indexing (column-major order)
        B[1] = 1.0  # (1, 1, 1)
        B[5] = 5.0  # (1, 1, 2)
        B[8] = 8.0  # (2, 2, 2)
        
        @test B[1, 1, 1] == 1.0
        @test B[1, 1, 2] == 5.0
        @test B[2, 2, 2] == 8.0
        @test nnz(B) == 3
        
        # Verify linear indexing matches CartesianIndex
        for i in 1:length(B)
            cart_idx = CartesianIndices(B)[i]
            @test B[i] == B[cart_idx]
        end
    end
    
    @testset "1D Array Linear Indexing" begin
        C = SparseArray{Int, 1}((5,))
        
        C[2] = 20
        C[4] = 40
        
        @test C[2] == 20
        @test C[4] == 40
        @test C[1] == 0
        @test C[3] == 0
        @test C[5] == 0
        @test nnz(C) == 2
    end
    
    @testset "Linear Indexing with CartesianIndex Equivalence" begin
        A = SparseArray{Int, 2}((3, 4))
        
        # Fill with some test data
        for i in 1:3, j in 1:4
            if (i + j) % 3 == 0
                A[i, j] = i * 10 + j
            end
        end
        
        # Verify linear and cartesian indexing give same results
        linear_idx = 1
        for cart_idx in CartesianIndices(A)
            @test A[linear_idx] == A[cart_idx]
            linear_idx += 1
        end
    end
    
    @testset "Linear Indexing Performance" begin
        # Test that linear indexing doesn't degrade performance significantly
        A = SparseArray{Int, 2}((100, 100))
        
        # Add some sparse data
        for i in 1:10:100
            A[i, i] = i
        end
        
        # Time linear access vs cartesian access
        linear_sum = 0
        cartesian_sum = 0
        
        # This is more of a smoke test than a rigorous benchmark
        for i in 1:length(A)
            linear_sum += A[i]
        end
        
        for I in CartesianIndices(A)
            cartesian_sum += A[I]
        end
        
        @test linear_sum == cartesian_sum
    end
end

end