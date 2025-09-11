module TestAdvancedTypePromotion

using NDimensionalSparseArrays
using Test

@testset "Advanced Type Promotion" begin
    @testset "Custom number types" begin
        # Test promotion with Rational types
        A = NDSparseArray{Rational{Int}, 2}((2, 2))
        A[1, 1] = 1//3
        A[2, 2] = 2//5

        B = NDSparseArray{Float64, 2}((2, 2))
        B[1, 1] = 0.5
        B[1, 2] = 0.25

        # Addition should promote to Float64
        C = A + B
        @test eltype(C) == Float64
        @test C[1, 1] ≈ 1/3 + 0.5
        @test C[2, 2] ≈ 2/5
        @test C[1, 2] ≈ 0.25

        # Subtraction should also promote correctly
        D = A - B
        @test eltype(D) == Float64
        @test D[1, 1] ≈ 1/3 - 0.5
    end

    @testset "BigFloat and BigInt promotion" begin
        A = NDSparseArray{BigFloat, 2}((2, 2))
        A[1, 1] = BigFloat("0.5")
        A[2, 1] = BigFloat("1.5")

        B = NDSparseArray{Float64, 2}((2, 2))
        B[1, 1] = 0.25
        B[2, 2] = 2.0

        C = A + B
        @test eltype(C) == BigFloat
        @test C[1, 1] == BigFloat("0.75")
        @test C[2, 1] == BigFloat("1.5")
        @test C[2, 2] == BigFloat("2.0")

        # Test with BigInt
        A_int = NDSparseArray{BigInt, 2}((2, 2))
        A_int[1, 1] = BigInt(10)^50  # Very large number

        B_int = NDSparseArray{Int, 2}((2, 2))
        B_int[1, 1] = 1
        B_int[2, 2] = 100

        C_int = A_int + B_int
        @test eltype(C_int) == BigInt
        @test C_int[1, 1] == BigInt(10)^50 + 1
        @test C_int[2, 2] == BigInt(100)
    end

    @testset "Complex number promotion" begin
        A = NDSparseArray{Complex{Int}, 2}((2, 2))
        A[1, 1] = 3 + 4im
        A[2, 1] = 1 + 0im

        B = NDSparseArray{Float64, 2}((2, 2))
        B[1, 1] = 2.5
        B[1, 2] = 1.5

        C = A + B
        @test eltype(C) == Complex{Float64}
        @test C[1, 1] == (3 + 4im) + 2.5
        @test C[2, 1] == 1.0 + 0im
        @test C[1, 2] == 1.5 + 0.0im

        # Test Complex with Complex
        D = NDSparseArray{Complex{Float32}, 2}((2, 2))
        D[1, 1] = Complex{Float32}(1.0, 2.0)
        D[2, 2] = Complex{Float32}(0.5, 0.0)

        E = A + D
        @test eltype(E) == Complex{Float32}  # Promotion keeps the more specific type
    end

    @testset "Zero result detection with different types" begin
        # Test with floating point precision issues
        A = NDSparseArray{Float64, 2}((2, 2))
        A[1, 1] = 1.0 + 2.0 * eps(1.0)

        B = NDSparseArray{Float64, 2}((2, 2))
        B[1, 1] = 1.0 + eps(1.0)

        C = A - B
        # Should store the small difference, not treat as zero
        @test hasindex(C, 1, 1)
        @test C[1, 1] ≈ eps(1.0)

        # Test with exactly equal values
        D = NDSparseArray{Float64, 2}((2, 2))
        D[1, 1] = 1.0

        E = NDSparseArray{Float64, 2}((2, 2))
        E[1, 1] = 1.0

        F = D - E
        @test !hasindex(F, 1, 1)  # Should be exactly zero, not stored

        # Test with complex numbers
        A_complex = NDSparseArray{Complex{Float64}, 2}((2, 2))
        A_complex[1, 1] = 1.0 + 2.0im

        B_complex = NDSparseArray{Complex{Float64}, 2}((2, 2))
        B_complex[1, 1] = 1.0 + 2.0im

        C_complex = A_complex - B_complex
        @test !hasindex(C_complex, 1, 1)  # Should be exactly zero
    end

    @testset "Scalar multiplication type promotion" begin
        A = NDSparseArray{Int, 2}((2, 2))
        A[1, 1] = 5
        A[2, 2] = 10

        # Multiply by Float64
        B = A * 2.5
        @test eltype(B) == Float64
        @test B[1, 1] == 12.5
        @test B[2, 2] == 25.0

        # Multiply by Complex
        C = A * (1 + 2im)
        @test eltype(C) == Complex{Int}
        @test C[1, 1] == 5 + 10im
        @test C[2, 2] == 10 + 20im

        # Multiply by Rational
        D = A * (3//4)
        @test eltype(D) == Rational{Int}
        @test D[1, 1] == 15//4
        @test D[2, 2] == 15//2

        # Test commutativity
        E = 2.5 * A
        @test E == B
        @test eltype(E) == Float64
    end

    @testset "Mixed dimensional promotion" begin
        # Test that operations only work with same dimensions
        A = NDSparseArray{Int, 2}((2, 2))
        A[1, 1] = 5

        B = NDSparseArray{Int, 3}((2, 2, 1))
        B[1, 1, 1] = 3

        # This should throw an error (either DimensionMismatch or BoundsError)
        @test_throws Exception A + B
        @test_throws Exception A - B
    end

    @testset "Promotion with unusual numeric types" begin
        # Test with different integer types
        A = NDSparseArray{Int8, 2}((2, 2))
        A[1, 1] = Int8(100)

        B = NDSparseArray{Int64, 2}((2, 2))
        B[1, 1] = Int64(200)

        C = A + B
        @test eltype(C) == Int64  # Should promote to larger type
        @test C[1, 1] == 300

        # Test with UInt
        D = NDSparseArray{UInt8, 2}((2, 2))
        D[1, 1] = UInt8(50)

        E = NDSparseArray{Int16, 2}((2, 2))
        E[1, 1] = Int16(75)

        F = D + E
        @test eltype(F) == Int16  # Actual promotion behavior 
        @test F[1, 1] == 125
    end
end

end
