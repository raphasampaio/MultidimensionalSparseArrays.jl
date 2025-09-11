module TestAdvancedIntegration

using NDimensionalSparseArrays
using Test
using LinearAlgebra

@testset "Advanced Integration Tests" begin
    @testset "Linear Algebra Operations" begin
        # Test transpose-like operations manually
        A = NDSparseArray{Float64, 2}((3, 4))
        A[1, 2] = 1.5
        A[2, 1] = 2.5
        A[3, 4] = 3.5

        # Manual transpose
        AT = NDSparseArray{Float64, 2}((4, 3))
        for (idx, val) in stored_pairs(A)
            i, j = Tuple(idx)
            AT[j, i] = val
        end

        @test AT[2, 1] == 1.5
        @test AT[1, 2] == 2.5
        @test AT[4, 3] == 3.5
        @test nnz(AT) == nnz(A)

        # Test diagonal extraction
        B = NDSparseArray{Int, 2}((5, 5))
        for i in 1:5
            B[i, i] = i^2
        end

        diag_vals = [B[i, i] for i in 1:5]
        @test diag_vals == [1, 4, 9, 16, 25]

        # Test trace-like operation
        trace_val = sum(B[i, i] for i in 1:5)
        @test trace_val == 55  # 1 + 4 + 9 + 16 + 25
    end

    @testset "Matrix-like Properties" begin
        # Test symmetric matrix creation
        A = NDSparseArray{Float64, 2}((4, 4))
        A[1, 2] = 1.5
        A[2, 1] = 1.5
        A[1, 4] = 2.5
        A[4, 1] = 2.5
        A[2, 3] = 3.5
        A[3, 2] = 3.5

        # Check symmetry manually
        @test A[1, 2] == A[2, 1]
        @test A[1, 4] == A[4, 1]
        @test A[2, 3] == A[3, 2]

        # Test upper/lower triangular patterns
        U = NDSparseArray{Int, 2}((4, 4))
        for i in 1:4, j in i:4
            if i <= j
                U[i, j] = i + j
            end
        end

        # Verify upper triangular
        for i in 1:4, j in 1:4
            if i > j
                @test !hasindex(U, i, j)
            else
                @test hasindex(U, i, j)
                @test U[i, j] == i + j
            end
        end
    end

    @testset "Norm-like Operations" begin
        A = NDSparseArray{Float64, 2}((3, 3))
        A[1, 1] = 3.0
        A[2, 2] = 4.0
        A[3, 3] = 0.0  # Explicitly stored zero

        # Frobenius norm (manual implementation)
        frobenius_norm = sqrt(sum(val^2 for val in stored_values(A)))
        @test frobenius_norm == 5.0  # sqrt(3^2 + 4^2 + 0^2) = sqrt(25) = 5

        # 1-norm (sum of absolute values)
        one_norm = sum(abs(val) for val in stored_values(A))
        @test one_norm == 7.0  # |3| + |4| + |0| = 7

        # Infinity norm (maximum absolute value)
        inf_norm = maximum(abs(val) for val in stored_values(A))
        @test inf_norm == 4.0
    end

    @testset "Structured Matrix Patterns" begin
        # Tridiagonal matrix
        n = 10
        T = NDSparseArray{Float64, 2}((n, n))

        for i in 1:n
            T[i, i] = 2.0  # Main diagonal
            if i > 1
                T[i, i-1] = -1.0  # Lower diagonal
            end
            if i < n
                T[i, i+1] = -1.0  # Upper diagonal
            end
        end

        @test nnz(T) == 3*n - 2  # n main + (n-1) lower + (n-1) upper
        @test T[1, 1] == 2.0
        @test T[1, 2] == -1.0
        @test T[2, 1] == -1.0
        @test !hasindex(T, 1, 3)

        # Circulant-like pattern
        C = NDSparseArray{Int, 2}((5, 5))
        for i in 1:5
            C[i, i] = i
            C[i, (i%5)+1] = i + 10
        end

        @test C[1, 1] == 1
        @test C[1, 2] == 11
        @test C[5, 1] == 15  # Wraparound
    end

    @testset "Advanced Array Transformations" begin
        # Test reshape-like operations (manual implementation)
        A = NDSparseArray{Int, 2}((2, 6))
        A[1, 2] = 10
        A[1, 5] = 20
        A[2, 3] = 30

        # Manual "reshape" to 3x4 by mapping indices
        B = NDSparseArray{Int, 2}((3, 4))
        for (old_idx, val) in stored_pairs(A)
            old_linear = LinearIndices((2, 6))[old_idx]
            new_idx = CartesianIndices((3, 4))[old_linear]
            B[new_idx] = val
        end

        @test nnz(B) == nnz(A)
        # Verify specific mappings
        @test B[CartesianIndices((3, 4))[LinearIndices((2, 6))[1, 2]]] == 10

        # Test permutation-like operations
        P = NDSparseArray{Int, 2}((4, 4))
        P[1, 4] = 1
        P[2, 3] = 1
        P[3, 2] = 1
        P[4, 1] = 1

        @test nnz(P) == 4
        # This represents a permutation matrix
        for i in 1:4
            row_sum = sum(P[i, j] for j in 1:4 if hasindex(P, i, j))
            @test row_sum == 1
        end
    end

    @testset "Kronecker Product-like Operations" begin
        # Manual implementation of Kronecker product for small matrices
        A = NDSparseArray{Int, 2}((2, 2))
        A[1, 1] = 1
        A[1, 2] = 2
        A[2, 1] = 3
        A[2, 2] = 4

        B = NDSparseArray{Int, 2}((2, 2))
        B[1, 1] = 5
        B[2, 2] = 6

        # Manual Kronecker product A ⊗ B
        K = NDSparseArray{Int, 2}((4, 4))
        for (idx_a, val_a) in stored_pairs(A)
            for (idx_b, val_b) in stored_pairs(B)
                i_a, j_a = Tuple(idx_a)
                i_b, j_b = Tuple(idx_b)
                i_k = (i_a - 1) * 2 + i_b
                j_k = (j_a - 1) * 2 + j_b
                K[i_k, j_k] = val_a * val_b
            end
        end

        @test nnz(K) == 8  # Each element of A creates a 2x2 block
        @test K[1, 1] == 5   # A[1,1] * B[1,1] = 1 * 5
        @test K[2, 2] == 6   # A[1,1] * B[2,2] = 1 * 6
        @test K[3, 1] == 15  # A[2,1] * B[1,1] = 3 * 5
        @test K[1, 3] == 10  # A[1,2] * B[1,1] = 2 * 5
    end

    @testset "Special Matrix Properties" begin
        # Test orthogonal-like matrices
        Q = NDSparseArray{Float64, 2}((3, 3))
        Q[1, 1] = 1.0
        Q[2, 2] = cos(π/4)
        Q[2, 3] = -sin(π/4)
        Q[3, 2] = sin(π/4)
        Q[3, 3] = cos(π/4)

        # Check that columns have unit norm (approximately)
        col1_norm = Q[1, 1]^2
        @test abs(col1_norm - 1.0) < 1e-10

        col2_norm = Q[2, 2]^2 + Q[3, 2]^2
        @test abs(col2_norm - 1.0) < 1e-10

        # Test band matrices
        B = NDSparseArray{Int, 2}((5, 5))
        for i in 1:5, j in 1:5
            if abs(i - j) <= 1  # Bandwidth of 1
                B[i, j] = i + j
            end
        end

        @test nnz(B) == 13  # 5 diagonal + 4 super + 4 sub
        @test B[1, 1] == 2
        @test B[1, 2] == 3
        @test !hasindex(B, 1, 3)
    end

    @testset "Advanced Sparsity Patterns" begin
        # Checkerboard pattern
        C = NDSparseArray{Int, 2}((8, 8))
        for i in 1:8, j in 1:8
            if (i + j) % 2 == 0
                C[i, j] = i * j
            end
        end

        @test nnz(C) == 32  # Half of 64
        @test hasindex(C, 1, 1)  # (1+1) % 2 == 0
        @test !hasindex(C, 1, 2)  # (1+2) % 2 == 1

        # Pseudo-random sparse pattern with specific density (deterministic)
        R = NDSparseArray{Float64, 2}((50, 50))
        # Add elements in a deterministic pseudo-random pattern
        for i in 1:5:50
            for j in 1:7:50
                R[i, j] = Float64(i + j)
            end
        end

        @test nnz(R) > 0
        @test nnz(R) < 50 * 50  # Should be sparse
    end

    @testset "Numerical Stability Tests" begin
        # Test with very small numbers
        A = NDSparseArray{Float64, 2}((3, 3))
        A[1, 1] = 1e-100
        A[2, 2] = 1e100
        A[3, 3] = 1.0

        # Operations should preserve precision
        B = A * 2.0
        @test B[1, 1] == 2e-100
        @test B[2, 2] == 2e100
        @test B[3, 3] == 2.0

        # Test with complex numbers
        C = NDSparseArray{Complex{Float64}, 2}((2, 2))
        C[1, 1] = 1e-50 + 1e-50im
        C[2, 2] = 1e50 - 1e50im

        D = C + C
        @test real(D[1, 1]) ≈ 2e-50
        @test imag(D[1, 1]) ≈ 2e-50
        @test real(D[2, 2]) ≈ 2e50
        @test imag(D[2, 2]) ≈ -2e50
    end

    @testset "Integration with Base Functions" begin
        A = NDSparseArray{Float64, 2}((4, 4))
        A[1, 1] = 1.0
        A[2, 2] = 4.0
        A[3, 3] = 9.0
        A[4, 4] = 16.0

        # Test that our array works with Base functions that expect AbstractArray
        @test size(A) == (4, 4)
        @test length(A) == 16
        @test ndims(A) == 2
        @test eltype(A) == Float64
        @test axes(A) == (1:4, 1:4)
        @test axes(A, 1) == 1:4
        @test axes(A, 2) == 1:4

        # Test with functions that might use iteration
        stored_vals = collect(A)
        @test length(stored_vals) == 4
        @test 1.0 in stored_vals
        @test 16.0 in stored_vals

        # Test count function over stored values
        count_positive = count(x -> x > 0, stored_values(A))
        @test count_positive == 4

        count_large = count(x -> x > 10, stored_values(A))
        @test count_large == 1  # Only 16 > 10
    end

    @testset "Custom Reduction Operations" begin
        A = NDSparseArray{Int, 2}((3, 4))
        A[1, 1] = 2
        A[1, 4] = 8
        A[2, 2] = 6
        A[3, 1] = 4

        # Test various reductions over stored values
        total = sum(stored_values(A))
        @test total == 20  # 2 + 8 + 6 + 4

        product = prod(stored_values(A))
        @test product == 384  # 2 * 8 * 6 * 4

        maximum_val = maximum(stored_values(A))
        @test maximum_val == 8

        minimum_val = minimum(stored_values(A))
        @test minimum_val == 2

        # Test any/all predicates
        all_positive = all(x -> x > 0, stored_values(A))
        @test all_positive == true

        any_large = any(x -> x > 5, stored_values(A))
        @test any_large == true

        any_negative = any(x -> x < 0, stored_values(A))
        @test any_negative == false
    end
end

end
