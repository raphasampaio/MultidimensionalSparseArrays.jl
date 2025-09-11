module TestIntegration

using NDimensionalSparseArrays
using Test
using LinearAlgebra

@testset "AbstractArray Interface Compliance" begin
    A = SparseArray{Int, 2}((3, 4))
    A[1, 1] = 5
    A[2, 3] = -2
    A[3, 4] = 10

    # Test that SparseArray is recognized as AbstractArray
    @test A isa AbstractArray
    @test A isa AbstractArray{Int, 2}

    # Test basic AbstractArray methods
    @test size(A) == (3, 4)
    @test length(A) == 12
    @test ndims(A) == 2
    @test eltype(A) == Int
    @test axes(A) == (1:3, 1:4)
    @test axes(A, 1) == 1:3
    @test axes(A, 2) == 1:4

    # Test indexing works with CartesianIndex
    @test A[CartesianIndex(1, 1)] == 5
    @test_throws BoundsError A[CartesianIndex(2, 2)]  # Unset index

    # Test that bounds checking works
    @test_throws BoundsError A[0, 1]
    @test_throws BoundsError A[4, 1]
    @test_throws BoundsError A[1, 5]
end

@testset "Iteration Interface" begin
    A = SparseArray{Int, 2}((2, 3))
    A[1, 1] = 10
    A[2, 2] = 20

    # Test basic iteration over stored values only
    values = collect(A)
    expected_stored = [10, 20]  # Only stored values
    @test Set(values) == Set(expected_stored)
    @test length(values) == 2

    # Test iteration over stored values with enumerate
    stored_vals = collect(A)
    for (i, val) in enumerate(stored_vals)
        @test val in [10, 20]  # Only stored values
    end

    # Test working with stored pairs
    B = similar(A)
    for (idx, val) in stored_pairs(A)
        B[idx] = val * 2
    end
    @test B[1, 1] == 20
    @test B[2, 2] == 40
    @test !hasindex(B, 1, 2)  # Should not be set
end

@testset "Broadcasting Compatibility" begin
    A = SparseArray{Int, 2}((2, 2))
    A[1, 1] = 5
    A[2, 2] = 10

    # Test that basic broadcasting-like operations work
    # Note: Full broadcasting support would require additional implementation
    # Here we test what currently works

    # Test scalar operations that are implemented
    B = A * 2
    @test B[1, 1] == 10
    @test B[2, 2] == 20
    @test !hasindex(B, 1, 2)  # Unset index

    C = 3 * A
    @test C[1, 1] == 15
    @test C[2, 2] == 30
end

@testset "Array Construction Functions" begin
    # Test compatibility with standard array functions

    # Test zeros-like behavior (empty sparse array)
    A = spzeros(Int, 3, 3)
    @test nnz(A) == 0
    @test !hasindex(A, 1, 1)  # No values stored initially

    # Test ones-like behavior  
    B = spones(Int, 2, 2)
    @test all(B[i, j] == 1 for i in 1:2, j in 1:2)
    @test nnz(B) == 4

    # Test fill-like behavior
    C = spfill(42, 2, 3)
    @test all(C[i, j] == 42 for i in 1:2, j in 1:3)
    @test nnz(C) == 6
end

@testset "Type System Integration" begin
    # Test type promotion and conversion
    A = SparseArray{Int, 2}((2, 2))
    A[1, 1] = 5

    B = SparseArray{Float64, 2}((2, 2))
    B[1, 1] = 2.5

    # Test that operations promote types correctly
    C = A + B
    @test eltype(C) == Float64
    @test C[1, 1] == 7.5

    # Test similar with different types
    D = similar(A, Float32)
    @test eltype(D) == Float32
    @test size(D) == size(A)

    E = similar(A, Complex{Int})
    @test eltype(E) == Complex{Int}
end

@testset "Linear Algebra Compatibility" begin
    # Test basic linear algebra operations where applicable
    A = SparseArray{Float64, 2}((3, 3))
    A[1, 1] = 1.0
    A[2, 2] = 2.0
    A[3, 3] = 3.0

    # Test that we can extract diagonal (manually since diag() might not work directly)
    diagonal_elements = [A[i, i] for i in 1:3]
    @test diagonal_elements == [1.0, 2.0, 3.0]

    # Test transpose by manual implementation using stored values
    B = SparseArray{Float64, 2}((3, 3))
    for (idx, val) in stored_pairs(A)
        i, j = Tuple(idx)
        B[j, i] = val
    end

    @test B[1, 1] == A[1, 1]
    @test B[2, 2] == A[2, 2]
    @test B[3, 3] == A[3, 3]
end

@testset "Memory and Performance with Standard Operations" begin
    # Test that common array operations work efficiently
    n = 100
    A = spzeros(Float64, n, n)

    # Add sparse structure (tridiagonal-like)
    for i in 1:n
        A[i, i] = 2.0
        if i > 1
            A[i, i-1] = -1.0
        end
        if i < n
            A[i, i+1] = -1.0
        end
    end

    @test nnz(A) <= 3 * n - 2  # At most 3n-2 non-zeros for tridiagonal

    # Test copy operation
    B = copy(A)
    @test B == A
    @test nnz(B) == nnz(A)

    # Test fill! operation
    C = copy(A)
    fill!(C, 0.0)
    @test nnz(C) == n * n  # fill! stores the value at all positions
    @test all(C[i, j] == 0.0 for i in 1:n, j in 1:n)

    # Test scalar operations
    D = A * 2.0
    @test nnz(D) == nnz(A)
    for i in 1:n
        @test D[i, i] == 4.0
    end
end

@testset "Edge Cases and Error Handling" begin
    # Test proper error handling for common mistakes
    A = SparseArray{Int, 2}((2, 3))
    B = SparseArray{Int, 2}((3, 2))  # Different size

    # Test dimension mismatch errors
    @test_throws DimensionMismatch A + B
    @test_throws DimensionMismatch A - B

    # Test bounds errors
    @test_throws BoundsError A[0, 1]
    @test_throws BoundsError A[3, 1]
    @test_throws BoundsError A[1, 4]

    # Test that setindex! returns the value
    result = (A[1, 1] = 42)
    @test result == 42
    @test A[1, 1] == 42
end

@testset "Conversion and Construction" begin
    # Test construction from other array types
    dense = [1 0 3; 0 2 0; 0 0 4]
    sparse = SparseArray(dense)

    @test sparse[1, 1] == 1
    @test sparse[2, 2] == 2
    @test sparse[3, 3] == 4
    @test !hasindex(sparse, 1, 2)  # Zero not stored
    @test nnz(sparse) == 4

    # Test that we can convert back to dense
    dense_reconstructed = to_dense(sparse)
    @test dense_reconstructed == dense

    # Test with different numeric types
    complex_dense = [1+0im 0+0im; 0+0im 2+3im]
    complex_sparse = SparseArray(complex_dense)
    @test eltype(complex_sparse) == Complex{Int}
    @test complex_sparse[1, 1] == 1 + 0im
    @test complex_sparse[2, 2] == 2 + 3im
    @test nnz(complex_sparse) == 2
end

end
