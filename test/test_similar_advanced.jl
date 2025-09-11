module TestSimilarAdvanced

using NDimensionalSparseArrays
using Test

@testset "similar Advanced Cases" begin
    @testset "Zero dimensions" begin
        A = NDSparseArray{Int, 2}((3, 4))
        A[1, 1] = 5

        # Test similar with zero dimensions
        B = similar(A, Float64, (0, 0))
        @test size(B) == (0, 0)
        @test eltype(B) == Float64
        @test nnz(B) == 0

        # Test with one zero dimension
        C = similar(A, String, (0, 5))
        @test size(C) == (0, 5)
        @test eltype(C) == String
        @test nnz(C) == 0

        D = similar(A, Bool, (3, 0))
        @test size(D) == (3, 0)
        @test eltype(D) == Bool
        @test nnz(D) == 0
    end

    @testset "Very large dimensions" begin
        A = NDSparseArray{Int, 2}((3, 4))
        A[1, 1] = 5

        # Test similar with very large dimensions
        large_dim = 10^6
        C = similar(A, Int8, (large_dim, large_dim))
        @test size(C) == (large_dim, large_dim)
        @test eltype(C) == Int8
        @test nnz(C) == 0
        @test length(C) == large_dim^2

        # Test that we can still add elements to large array
        C[1, 1] = Int8(42)
        @test C[1, 1] == Int8(42)
        @test nnz(C) == 1
    end

    @testset "Different dimensionality" begin
        A = NDSparseArray{Int, 2}((3, 4))
        A[1, 1] = 5

        # Test similar with different number of dimensions
        D = similar(A, Float32, (2, 3, 4))
        @test size(D) == (2, 3, 4)
        @test ndims(D) == 3
        @test eltype(D) == Float32
        @test nnz(D) == 0

        # Test 1D
        E = similar(A, Complex{Int}, (10,))
        @test size(E) == (10,)
        @test ndims(E) == 1
        @test eltype(E) == Complex{Int}

        # Test 4D
        F = similar(A, Rational{Int}, (2, 2, 2, 2))
        @test size(F) == (2, 2, 2, 2)
        @test ndims(F) == 4
        @test eltype(F) == Rational{Int}
    end

    @testset "Similar with same type" begin
        A = NDSparseArray{Float64, 3}((2, 3, 4))
        A[1, 1, 1] = 3.14
        A[2, 3, 4] = 2.71

        # Test similar without type change
        B = similar(A)
        @test size(B) == size(A)
        @test eltype(B) == eltype(A)
        @test ndims(B) == ndims(A)
        @test nnz(B) == 0  # Should be empty

        # Original should be unchanged
        @test nnz(A) == 2
        @test A[1, 1, 1] == 3.14
    end

    @testset "Similar with type change only" begin
        A = NDSparseArray{Int, 2}((3, 3))
        A[1, 1] = 5
        A[2, 2] = 10

        # Test similar with only type change
        B = similar(A, Float64)
        @test size(B) == size(A)
        @test eltype(B) == Float64
        @test ndims(B) == ndims(A)
        @test nnz(B) == 0

        # Test with complex type
        C = similar(A, Complex{Float32})
        @test size(C) == size(A)
        @test eltype(C) == Complex{Float32}
        @test ndims(C) == ndims(A)
        @test nnz(C) == 0
    end

    @testset "Edge case dimensions" begin
        A = NDSparseArray{Int, 2}((5, 5))

        # Test with single element dimensions
        B = similar(A, Float64, (1, 1))
        @test size(B) == (1, 1)
        @test eltype(B) == Float64

        # Test that we can use the single element array
        B[1, 1] = 99.5
        @test B[1, 1] == 99.5
        @test nnz(B) == 1

        # Test with mixed small/large dimensions
        C = similar(A, Int8, (1, 1000))
        @test size(C) == (1, 1000)
        @test eltype(C) == Int8
        @test nnz(C) == 0

        D = similar(A, UInt64, (1000, 1))
        @test size(D) == (1000, 1)
        @test eltype(D) == UInt64
        @test nnz(D) == 0
    end

    @testset "Similar preserves independence" begin
        A = NDSparseArray{Int, 2}((3, 3))
        A[1, 1] = 100
        A[2, 2] = 200

        B = similar(A)
        @test nnz(B) == 0

        # Modifying B should not affect A
        B[1, 1] = 999
        B[3, 3] = 888

        @test A[1, 1] == 100
        @test A[2, 2] == 200
        @test nnz(A) == 2
        @test !hasindex(A, 3, 3)

        @test B[1, 1] == 999
        @test B[3, 3] == 888
        @test nnz(B) == 2
        @test !hasindex(B, 2, 2)
    end

    @testset "Similar with unusual types" begin
        A = NDSparseArray{Int, 2}((2, 2))

        # Test with String type
        B = similar(A, String, (2, 2))
        @test eltype(B) == String
        B[1, 1] = "hello"
        @test B[1, 1] == "hello"

        # Test with custom struct (if applicable)
        struct CustomNumber
            value::Int
        end

        C = similar(A, CustomNumber, (2, 2))
        @test eltype(C) == CustomNumber
        C[1, 1] = CustomNumber(42)
        @test C[1, 1].value == 42

        # Test with Any type
        D = similar(A, Any, (2, 2))
        @test eltype(D) == Any
        D[1, 1] = "mixed"
        D[2, 2] = 123
        @test D[1, 1] == "mixed"
        @test D[2, 2] == 123
    end
end

end
