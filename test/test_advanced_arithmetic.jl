module TestAdvancedArithmetic

using MultidimensionalSparseArrays
using Test

@testset "Advanced Arithmetic Operations" begin
    @testset "Addition Edge Cases" begin
        # Test addition with completely disjoint sparse patterns
        A = SparseArray{Int, 2}((3, 3))
        A[1, 1] = 5
        A[2, 2] = 10
        
        B = SparseArray{Int, 2}((3, 3))
        B[1, 3] = 3
        B[3, 1] = 7
        
        C = A + B
        @test C[1, 1] == 5
        @test C[2, 2] == 10
        @test C[1, 3] == 3
        @test C[3, 1] == 7
        @test nnz(C) == 4
        @test !hasindex(C, 2, 1)
        @test !hasindex(C, 3, 3)
        
        # Test addition with overlapping indices
        D = SparseArray{Int, 2}((3, 3))
        D[1, 1] = 2
        D[2, 3] = 8
        
        E = A + D
        @test E[1, 1] == 7  # 5 + 2
        @test E[2, 2] == 10  # 10 + 0 (D doesn't have [2,2])
        @test E[2, 3] == 8   # 0 + 8 (A doesn't have [2,3])
        @test nnz(E) == 3
    end
    
    @testset "Subtraction with Zero Results" begin
        A = SparseArray{Int, 2}((3, 3))
        A[1, 1] = 5
        A[2, 2] = 10
        A[1, 3] = 7
        
        B = SparseArray{Int, 2}((3, 3))
        B[1, 1] = 5  # Same as A[1,1] - should result in zero
        B[2, 2] = 3  # Different from A[2,2]
        B[3, 1] = 4  # Not in A
        
        C = A - B
        @test !hasindex(C, 1, 1)  # 5 - 5 = 0, not stored
        @test C[2, 2] == 7        # 10 - 3 = 7
        @test C[1, 3] == 7        # 7 - 0 = 7
        @test C[3, 1] == -4       # 0 - 4 = -4
        @test nnz(C) == 3
    end
    
    @testset "Type Promotion in Arithmetic" begin
        # Int + Float64
        A = SparseArray{Int, 2}((2, 2))
        A[1, 1] = 5
        
        B = SparseArray{Float64, 2}((2, 2))
        B[1, 1] = 2.5
        B[2, 2] = 3.7
        
        C = A + B
        @test eltype(C) == Float64
        @test C[1, 1] == 7.5
        @test C[2, 2] == 3.7
        
        # Int + Complex
        D = SparseArray{Complex{Float64}, 2}((2, 2))
        D[1, 1] = 1.0 + 2.0im
        
        E = A + D
        @test eltype(E) == Complex{Float64}
        @test E[1, 1] == 6.0 + 2.0im
        
        # Rational arithmetic
        F = SparseArray{Rational{Int}, 2}((2, 2))
        F[1, 1] = 1//3
        
        G = SparseArray{Rational{Int}, 2}((2, 2))
        G[1, 1] = 1//6
        G[2, 1] = 2//3
        
        H = F + G
        @test H[1, 1] == 1//2  # 1/3 + 1/6 = 1/2
        @test H[2, 1] == 2//3
    end
    
    @testset "Scalar Multiplication Edge Cases" begin
        A = SparseArray{Float64, 2}((3, 3))
        A[1, 1] = 2.0
        A[2, 3] = -1.5
        A[3, 2] = 0.0  # Explicitly stored zero
        
        # Multiplication by zero
        B = A * 0
        @test nnz(B) == 0  # All values become zero, nothing stored
        
        # Multiplication by negative number
        C = A * (-2)
        @test C[1, 1] == -4.0
        @test C[2, 3] == 3.0
        @test C[3, 2] == 0.0
        @test nnz(C) == 3  # Zero is still stored
        
        # Multiplication by complex number
        D = A * (1 + 1im)
        @test eltype(D) == Complex{Float64}
        @test D[1, 1] == 2.0 + 2.0im
        @test D[2, 3] == -1.5 - 1.5im
        
        # Left multiplication
        E = (2 + 3im) * A
        @test E[1, 1] == 4.0 + 6.0im
        @test E[2, 3] == -3.0 - 4.5im
    end
    
    @testset "Large Sparse Array Arithmetic" begin
        # Test arithmetic on larger arrays to ensure efficiency
        A = SparseArray{Float64, 2}((100, 100))
        B = SparseArray{Float64, 2}((100, 100))
        
        # Add sparse diagonal pattern
        for i in 1:10:100
            A[i, i] = Float64(i)
            B[i, i] = Float64(i) * 0.5
        end
        
        # Add some off-diagonal elements
        A[25, 75] = 12.5
        B[75, 25] = 8.3
        
        C = A + B
        @test nnz(C) == 12  # 10 diagonal + 2 off-diagonal
        @test C[11, 11] == 16.5  # 11 + 5.5
        @test C[25, 75] == 12.5
        @test C[75, 25] == 8.3
        
        # Scalar multiplication should preserve sparsity
        D = A * 2.0
        @test nnz(D) == nnz(A)
    end
    
    @testset "Arithmetic with Different Dimensions" begin
        A = SparseArray{Int, 3}((2, 2, 2))
        A[1, 1, 1] = 5
        A[2, 2, 2] = 10
        
        B = SparseArray{Int, 3}((2, 2, 2))
        B[1, 1, 1] = 3
        B[1, 2, 1] = 7
        
        C = A + B
        @test C[1, 1, 1] == 8
        @test C[2, 2, 2] == 10
        @test C[1, 2, 1] == 7
        @test nnz(C) == 3
    end
    
    @testset "Arithmetic Result Optimization" begin
        # Test that arithmetic operations don't store unnecessary zeros
        A = SparseArray{Float64, 2}((3, 3))
        A[1, 1] = 5.0
        A[2, 2] = 3.0
        
        B = SparseArray{Float64, 2}((3, 3))
        B[1, 1] = 5.0  # Will cancel out in subtraction
        B[3, 3] = 2.0
        
        C = A - B
        @test !hasindex(C, 1, 1)  # Should not store 5.0 - 5.0 = 0.0
        @test C[2, 2] == 3.0
        @test C[3, 3] == -2.0
        @test nnz(C) == 2
    end
    
    @testset "Precision and Floating Point Arithmetic" begin
        A = SparseArray{Float64, 2}((2, 2))
        A[1, 1] = 0.1 + 0.2  # This is 0.30000000000000004 in floating point
        
        B = SparseArray{Float64, 2}((2, 2))
        B[1, 1] = 0.3
        
        C = A - B
        # The result should be close to zero but not exactly zero due to floating point precision
        @test hasindex(C, 1, 1)
        @test abs(C[1, 1]) < 1e-15
        @test C[1, 1] != 0.0
    end
    
    @testset "Chain Operations" begin
        A = SparseArray{Int, 2}((3, 3))
        A[1, 1] = 1
        A[2, 2] = 2
        
        B = SparseArray{Int, 2}((3, 3))
        B[1, 1] = 1
        B[1, 3] = 3
        
        C = SparseArray{Int, 2}((3, 3))
        C[2, 2] = 1
        C[3, 1] = 4
        
        # Test (A + B) - C
        D = (A + B) - C
        @test D[1, 1] == 2   # (1 + 1) - 0 = 2
        @test D[2, 2] == 1   # (2 + 0) - 1 = 1
        @test D[1, 3] == 3   # (0 + 3) - 0 = 3
        @test D[3, 1] == -4  # (0 + 0) - 4 = -4
        @test nnz(D) == 4
    end
end

end