module TestErrorHandling

using NDimensionalSparseArrays
using Test

@testset "Error Handling and Bounds Checking" begin
    @testset "Bounds Checking - 2D Arrays" begin
        A = SparseArray{Int, 2}((3, 4))
        A[2, 3] = 42

        # Test all boundary conditions
        @test_throws BoundsError A[0, 1]
        @test_throws BoundsError A[1, 0]
        @test_throws BoundsError A[4, 1]
        @test_throws BoundsError A[1, 5]
        @test_throws BoundsError A[-1, 2]
        @test_throws BoundsError A[2, -1]

        # Test valid bounds don't throw
        @test A[2, 3] == 42

        # Test setindex bounds checking
        @test_throws BoundsError A[0, 1] = 5
        @test_throws BoundsError A[4, 1] = 5
        @test_throws BoundsError A[1, 5] = 5

        # Valid setindex should work
        A[1, 1] = 10
        @test A[1, 1] == 10
    end

    @testset "Bounds Checking - Higher Dimensions" begin
        # 3D array
        A = SparseArray{Float64, 3}((2, 3, 4))

        @test_throws BoundsError A[0, 1, 1]
        @test_throws BoundsError A[1, 0, 1]
        @test_throws BoundsError A[1, 1, 0]
        @test_throws BoundsError A[3, 1, 1]
        @test_throws BoundsError A[1, 4, 1]
        @test_throws BoundsError A[1, 1, 5]

        # 4D array
        B = SparseArray{Int, 4}((2, 2, 2, 2))
        @test_throws BoundsError B[0, 1, 1, 1]
        @test_throws BoundsError B[1, 1, 1, 3]

        # 1D array
        C = SparseArray{Int, 1}((5,))
        @test_throws BoundsError C[0]
        @test_throws BoundsError C[6]
        @test_throws BoundsError C[-1]
    end

    @testset "CartesianIndex Bounds Checking" begin
        A = SparseArray{Int, 2}((3, 3))
        A[CartesianIndex(2, 2)] = 15

        @test_throws BoundsError A[CartesianIndex(0, 1)]
        @test_throws BoundsError A[CartesianIndex(1, 4)]
        @test_throws BoundsError A[CartesianIndex(4, 1)]

        # Valid access
        @test A[CartesianIndex(2, 2)] == 15

        # Test setindex with CartesianIndex
        @test_throws BoundsError A[CartesianIndex(0, 2)] = 5
        A[CartesianIndex(1, 3)] = 25
        @test A[CartesianIndex(1, 3)] == 25
    end

    @testset "Linear Indexing Bounds Checking" begin
        A = SparseArray{Int, 2}((2, 3))  # 6 elements total
        A[3] = 42  # Linear index 3 -> CartesianIndex(1, 2)

        @test_throws BoundsError A[0]
        @test_throws BoundsError A[7]
        @test_throws BoundsError A[-1]

        # Valid linear access
        @test A[3] == 42

        # Test linear setindex bounds
        @test_throws BoundsError A[0] = 5
        @test_throws BoundsError A[7] = 5
        A[5] = 25  # Should work
        @test A[5] == 25
    end

    @testset "Dimension Mismatch Errors" begin
        A = SparseArray{Int, 2}((3, 3))
        A[1, 1] = 5

        B = SparseArray{Int, 2}((2, 2))
        B[1, 1] = 3

        C = SparseArray{Int, 2}((3, 2))
        C[1, 1] = 7

        # Test addition dimension mismatch
        @test_throws DimensionMismatch A + B
        @test_throws DimensionMismatch A + C
        @test_throws DimensionMismatch B + C

        # Test subtraction dimension mismatch
        @test_throws DimensionMismatch A - B
        @test_throws DimensionMismatch A - C
        @test_throws DimensionMismatch B - C

        # Same size should work
        D = SparseArray{Int, 2}((3, 3))
        D[2, 2] = 10
        result = A + D
        @test result[1, 1] == 5
        @test result[2, 2] == 10
    end

    @testset "Type Constructor Errors" begin
        # Note: Current implementation allows zero dimensions
        # These tests verify that the constructor accepts various inputs
        A = SparseArray{Int, 2}((1, 3))
        @test size(A) == (1, 3)

        B = SparseArray{Int, 1}((5,))
        @test size(B) == (5,)

        # Test dimension count mismatch (this should fail with MethodError)
        @test_throws MethodError SparseArray{Int, 2}((3, 3, 3))  # 3 dims for 2D array
        @test_throws MethodError SparseArray{Int, 3}((3, 3))     # 2 dims for 3D array
    end

    @testset "Unset Index Access Errors" begin
        A = SparseArray{Float64, 2}((3, 3))
        A[1, 1] = 5.0
        A[3, 3] = 10.0

        # Test that accessing unset indices throws BoundsError
        @test_throws BoundsError A[1, 2]
        @test_throws BoundsError A[2, 1]
        @test_throws BoundsError A[2, 2]
        @test_throws BoundsError A[2, 3]
        @test_throws BoundsError A[3, 1]
        @test_throws BoundsError A[3, 2]

        # But set indices should work
        @test A[1, 1] == 5.0
        @test A[3, 3] == 10.0

        # Test with CartesianIndex
        @test_throws BoundsError A[CartesianIndex(2, 2)]
        @test A[CartesianIndex(1, 1)] == 5.0

        # Test with linear indexing
        @test_throws BoundsError A[5]  # Linear index 5 -> (2, 2)
        @test A[1] == 5.0  # Linear index 1 -> (1, 1)
    end

    @testset "Delete Operation Error Handling" begin
        A = SparseArray{Int, 2}((3, 3))
        A[1, 1] = 5
        A[2, 2] = 10

        # Test deleting existing values
        delete!(A, 1, 1)
        @test !hasindex(A, 1, 1)
        @test_throws BoundsError A[1, 1]

        # Test deleting bounds-checked indices
        @test_throws BoundsError delete!(A, 0, 1)
        @test_throws BoundsError delete!(A, 4, 1)
        @test_throws BoundsError delete!(A, 1, 4)

        # Test deleting already unset index (should not error)
        delete!(A, 1, 3)  # This was never set
        @test !hasindex(A, 1, 3)

        # Test with CartesianIndex
        delete!(A, CartesianIndex(2, 2))
        @test !hasindex(A, 2, 2)

        @test_throws BoundsError delete!(A, CartesianIndex(0, 1))
    end

    @testset "Function Argument Validation" begin
        A = SparseArray{Int, 2}((3, 3))

        # Test hasindex with wrong number of arguments
        @test_throws MethodError hasindex(A, 1)  # Need 2 indices for 2D array
        @test_throws MethodError hasindex(A, 1, 2, 3)  # Too many indices

        # Test dropstored! with different types
        A[1, 1] = 5
        A[2, 2] = 0

        dropstored!(A, 0)
        @test !hasindex(A, 2, 2)
        @test hasindex(A, 1, 1)

        # Type should match for effective dropstored!
        dropstored!(A, 0.0)  # Float vs Int - should not remove anything
        @test hasindex(A, 1, 1)
    end

    @testset "Memory and Resource Errors" begin
        # Test that extremely large arrays handle memory gracefully
        # (This is more of a smoke test)
        A = SparseArray{Int, 2}((10^6, 10^6))  # Very large but sparse
        @test size(A) == (10^6, 10^6)
        @test nnz(A) == 0

        # Adding a few elements should still be efficient
        A[1, 1] = 42
        A[10^6, 10^6] = 24
        @test nnz(A) == 2
        @test A[1, 1] == 42
        @test A[10^6, 10^6] == 24
    end

    @testset "Edge Case: Zero-Dimensional Arrays" begin
        # Test behavior with edge cases in dimensions
        # Zero-dimensional arrays are actually allowed by the implementation
        A = SparseArray{Int, 0}(())
        @test size(A) == ()
        @test ndims(A) == 0

        @test_throws MethodError SparseArray{Int, 1}()  # No dimensions provided
    end

    @testset "Complex Error Scenarios" begin
        A = SparseArray{Complex{Float64}, 2}((2, 2))
        A[1, 1] = 1.0 + 2.0im

        # Test bounds errors with complex numbers
        @test_throws BoundsError A[0, 1]
        @test_throws BoundsError A[3, 1]

        # Test arithmetic dimension mismatches with complex arrays
        B = SparseArray{Complex{Float64}, 2}((3, 3))
        @test_throws DimensionMismatch A + B

        # Test that complex arithmetic works when dimensions match
        C = SparseArray{Complex{Float64}, 2}((2, 2))
        C[1, 1] = 1.0 - 2.0im
        D = A + C
        @test D[1, 1] == 2.0 + 0.0im
    end
end

end
