module TestInPlaceOperations

using NDimensionalSparseArrays
using Test

@testset "In-Place Arithmetic Operations" begin
    @testset "add! with NDSparseArray" begin
        # Basic array addition
        A = NDSparseArray{Int, 2}((3, 3))
        A[1, 1] = 5
        A[2, 2] = 10
        A[3, 1] = -3

        B = NDSparseArray{Int, 2}((3, 3))
        B[1, 1] = 2
        B[2, 3] = 7
        B[3, 1] = 8

        original_A_ref = A
        result = add!(A, B)

        # Test return value and in-place modification
        @test result === A
        @test result === original_A_ref

        # Test values
        @test A[1, 1] == 7  # 5 + 2
        @test A[2, 2] == 10  # 10 + 0
        @test A[2, 3] == 7   # 0 + 7
        @test A[3, 1] == 5   # -3 + 8
        @test nnz(A) == 4

        # Test that B is unchanged
        @test B[1, 1] == 2
        @test B[2, 3] == 7
        @test B[3, 1] == 8
        @test nnz(B) == 3
    end

    @testset "add! with type conversion" begin
        A = NDSparseArray{Int, 2}((2, 2))
        A[1, 1] = 5
        A[2, 2] = 10

        B = NDSparseArray{Float64, 2}((2, 2))
        B[1, 1] = 2.0  # Use whole number that can be converted to Int
        B[1, 2] = 3.0

        add!(A, B)

        # Test values after type conversion - A remains Int type
        @test A[1, 1] == 7    # 5 + 2 (converted to Int)
        @test A[2, 2] == 10   # 10 + 0
        @test A[1, 2] == 3    # 0 + 3 (converted to Int)
        @test eltype(A) == Int  # Array type should remain unchanged
    end

    @testset "add! with complex numbers" begin
        A = NDSparseArray{Complex{Int}, 2}((2, 2))
        A[1, 1] = 3 + 4im
        A[2, 2] = 1 + 0im

        B = NDSparseArray{Complex{Int}, 2}((2, 2))
        B[1, 1] = 2 + 1im
        B[1, 2] = 0 + 3im

        add!(A, B)

        @test A[1, 1] == 5 + 5im
        @test A[2, 2] == 1 + 0im
        @test A[1, 2] == 0 + 3im
        @test nnz(A) == 3
    end

    @testset "add! dimension mismatch" begin
        A = NDSparseArray{Int, 2}((2, 2))
        A[1, 1] = 5

        B = NDSparseArray{Int, 2}((3, 3))
        B[1, 1] = 2

        @test_throws DimensionMismatch add!(A, B)

        # A should be unchanged after error
        @test A[1, 1] == 5
        @test nnz(A) == 1
    end

    @testset "add! with scalar" begin
        A = NDSparseArray{Int, 2}((3, 3))
        A[1, 1] = 5
        A[2, 2] = 10
        A[3, 1] = -2

        original_ref = A
        result = add!(A, 3)

        @test result === A
        @test result === original_ref
        @test A[1, 1] == 8   # 5 + 3
        @test A[2, 2] == 13  # 10 + 3
        @test A[3, 1] == 1   # -2 + 3
        @test nnz(A) == 3
    end

    @testset "add! with zero scalar" begin
        A = NDSparseArray{Int, 2}((2, 2))
        A[1, 1] = 5
        A[2, 2] = 10

        add!(A, 0)

        @test A[1, 1] == 5
        @test A[2, 2] == 10
        @test nnz(A) == 2
    end

    @testset "add! scalar conversion" begin
        A = NDSparseArray{Int, 2}((2, 2))
        A[1, 1] = 5
        A[2, 2] = 10

        add!(A, 2)  # Use integer to avoid conversion issues

        @test A[1, 1] == 7
        @test A[2, 2] == 12
        @test nnz(A) == 2

        # Test with Float64 array and float scalar
        B = NDSparseArray{Float64, 2}((2, 2))
        B[1, 1] = 5.0
        B[2, 2] = 10.0

        add!(B, 2.5)

        @test B[1, 1] == 7.5
        @test B[2, 2] == 12.5
        @test nnz(B) == 2
    end

    @testset "sub! with NDSparseArray" begin
        A = NDSparseArray{Int, 2}((3, 3))
        A[1, 1] = 10
        A[2, 2] = 15
        A[3, 1] = 8

        B = NDSparseArray{Int, 2}((3, 3))
        B[1, 1] = 3
        B[2, 3] = 5
        B[3, 1] = 8  # This should result in zero

        original_ref = A
        result = sub!(A, B)

        @test result === A
        @test result === original_ref
        @test A[1, 1] == 7    # 10 - 3
        @test A[2, 2] == 15   # 15 - 0
        @test A[2, 3] == -5   # 0 - 5
        @test !hasindex(A, 3, 1)  # 8 - 8 = 0, should be removed
        @test nnz(A) == 3
    end

    @testset "sub! with scalar" begin
        A = NDSparseArray{Int, 2}((2, 2))
        A[1, 1] = 10
        A[2, 2] = 5

        sub!(A, 3)

        @test A[1, 1] == 7
        @test A[2, 2] == 2
        @test nnz(A) == 2
    end

    @testset "sub! with zero scalar" begin
        A = NDSparseArray{Int, 2}((2, 2))
        A[1, 1] = 5
        A[2, 2] = 10

        sub!(A, 0)

        @test A[1, 1] == 5
        @test A[2, 2] == 10
        @test nnz(A) == 2
    end

    @testset "mul! with scalar" begin
        A = NDSparseArray{Int, 2}((3, 3))
        A[1, 1] = 5
        A[2, 2] = 10
        A[3, 1] = -2

        original_ref = A
        result = mul!(A, 3)

        @test result === A
        @test result === original_ref
        @test A[1, 1] == 15   # 5 * 3
        @test A[2, 2] == 30   # 10 * 3
        @test A[3, 1] == -6   # -2 * 3
        @test nnz(A) == 3
    end

    @testset "mul! with zero scalar" begin
        A = NDSparseArray{Int, 2}((2, 2))
        A[1, 1] = 5
        A[2, 2] = 10

        mul!(A, 0)

        @test nnz(A) == 0
        @test length(A.data) == 0
    end

    @testset "mul! with one scalar" begin
        A = NDSparseArray{Int, 2}((2, 2))
        A[1, 1] = 5
        A[2, 2] = 10

        mul!(A, 1)

        @test A[1, 1] == 5
        @test A[2, 2] == 10
        @test nnz(A) == 2
    end

    @testset "mul! with fractional scalar" begin
        A = NDSparseArray{Float64, 2}((2, 2))  # Use Float64 array for fractional results
        A[1, 1] = 6.0
        A[2, 2] = 9.0

        mul!(A, 0.5)

        @test A[1, 1] == 3.0
        @test A[2, 2] == 4.5
        @test nnz(A) == 2
    end

    @testset "Chain operations" begin
        A = NDSparseArray{Int, 2}((3, 3))
        A[1, 1] = 10
        A[2, 2] = 5

        B = NDSparseArray{Int, 2}((3, 3))
        B[1, 1] = 2
        B[3, 3] = 8

        # Chain operations
        add!(A, B)
        sub!(A, 3)
        mul!(A, 2)

        @test A[1, 1] == 18  # (10 + 2 - 3) * 2 = 18
        @test A[2, 2] == 4   # (5 + 0 - 3) * 2 = 4
        @test A[3, 3] == 10  # (0 + 8 - 3) * 2 = 10
        @test nnz(A) == 3
    end

    @testset "Operations with empty arrays" begin
        A = NDSparseArray{Int, 2}((2, 2))
        B = NDSparseArray{Int, 2}((2, 2))

        add!(A, B)
        @test nnz(A) == 0

        sub!(A, B)
        @test nnz(A) == 0

        # Add some values then test with empty
        A[1, 1] = 5
        add!(A, B)
        @test A[1, 1] == 5
        @test nnz(A) == 1
    end

    @testset "Operations preserve sparsity" begin
        A = NDSparseArray{Int, 2}((100, 100))
        A[1, 1] = 5
        A[50, 50] = 10
        A[100, 100] = 15

        # Operations should not create dense arrays
        add!(A, 0)
        @test nnz(A) == 3

        sub!(A, 0)
        @test nnz(A) == 3

        mul!(A, 1)
        @test nnz(A) == 3

        # Verify values are still correct
        @test A[1, 1] == 5
        @test A[50, 50] == 10
        @test A[100, 100] == 15
    end

    @testset "Memory efficiency" begin
        A = NDSparseArray{Int, 2}((1000, 1000))
        # Add a few sparse elements
        A[1, 1] = 5
        A[500, 500] = 10
        A[1000, 1000] = 15

        original_memory = length(A.data)

        # Operations should not significantly increase memory usage
        add!(A, 100)
        @test length(A.data) == original_memory

        mul!(A, 2)
        @test length(A.data) == original_memory

        # Verify large array is still sparse
        @test nnz(A) == 3
        @test A[1, 1] == 210    # (5 + 100) * 2
        @test A[500, 500] == 220  # (10 + 100) * 2
        @test A[1000, 1000] == 230  # (15 + 100) * 2
    end

    @testset "Higher dimensional arrays" begin
        # Test 3D arrays
        A = NDSparseArray{Int, 3}((2, 2, 2))
        A[1, 1, 1] = 5
        A[2, 2, 2] = 10

        B = NDSparseArray{Int, 3}((2, 2, 2))
        B[1, 1, 1] = 2
        B[1, 2, 1] = 7

        add!(A, B)

        @test A[1, 1, 1] == 7
        @test A[2, 2, 2] == 10
        @test A[1, 2, 1] == 7
        @test nnz(A) == 3

        # Test scalar operations on 3D
        mul!(A, 2)
        @test A[1, 1, 1] == 14
        @test A[2, 2, 2] == 20
        @test A[1, 2, 1] == 14
    end

    @testset "Generic scalar types" begin
        # Test with custom numeric type that supports +, -, *
        struct CustomNumber
            value::Float64
        end
        Base.:+(a::CustomNumber, b::CustomNumber) = CustomNumber(a.value + b.value)
        Base.:-(a::CustomNumber, b::CustomNumber) = CustomNumber(a.value - b.value)
        Base.:*(a::CustomNumber, b::CustomNumber) = CustomNumber(a.value * b.value)
        Base.convert(::Type{CustomNumber}, x::CustomNumber) = x
        Base.zero(::Type{CustomNumber}) = CustomNumber(0.0)

        A = NDSparseArray{CustomNumber, 2}((2, 2))
        A[1, 1] = CustomNumber(5.0)
        A[2, 2] = CustomNumber(10.0)

        add!(A, CustomNumber(3.0))

        @test A[1, 1].value == 8.0
        @test A[2, 2].value == 13.0
        @test nnz(A) == 2

        # Test with rational numbers
        B = NDSparseArray{Rational{Int}, 2}((2, 2))
        B[1, 1] = 1//2
        B[2, 2] = 3//4

        add!(B, 1//4)

        @test B[1, 1] == 3//4
        @test B[2, 2] == 1//1
        @test nnz(B) == 2

        # Test with complex numbers and complex scalar
        C = NDSparseArray{Complex{Int}, 2}((2, 2))
        C[1, 1] = 2 + 3im
        C[2, 2] = 1 + 1im

        add!(C, 1 + 2im)

        @test C[1, 1] == 3 + 5im
        @test C[2, 2] == 2 + 3im
        @test nnz(C) == 2

        # Test multiplication with rational
        mul!(B, 2//3)

        @test B[1, 1] == 1//2  # (3//4) * (2//3) = 1//2
        @test B[2, 2] == 2//3  # (1//1) * (2//3) = 2//3
        @test nnz(B) == 2
    end

    @testset "Generic type error handling" begin
        # Test that incompatible types still give sensible errors
        A = NDSparseArray{Int, 2}((2, 2))
        A[1, 1] = 5

        # Test with incompatible scalar type that can't convert
        struct IncompatibleType
            value::String
        end

        incompatible = IncompatibleType("test")
        @test_throws MethodError add!(A, incompatible)
    end
end

end
