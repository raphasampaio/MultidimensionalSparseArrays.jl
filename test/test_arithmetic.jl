module TestArithmetic

using MultidimensionalSparseArrays
using Test

@testset "Arithmetic Operations" begin
    @testset "Addition" begin
        A = SparseArray{Int, 2}((3, 3))
        A[1, 1] = 5
        A[2, 2] = 10

        B = SparseArray{Int, 2}((3, 3))
        B[1, 1] = 3
        B[3, 3] = 7

        C = A + B
        @test C[1, 1] == 8
        @test C[2, 2] == 10
        @test C[3, 3] == 7
        @test !hasindex(C, 1, 2)
        @test nnz(C) == 3

        # Test type promotion
        D = SparseArray{Float64, 2}((2, 2))
        D[1, 1] = 2.5
        E = SparseArray{Int, 2}((2, 2))
        E[1, 1] = 3

        F = D + E
        @test eltype(F) == Float64
        @test F[1, 1] == 5.5

        # Test dimension mismatch
        G = SparseArray{Int, 2}((2, 3))
        @test_throws DimensionMismatch A + G
    end

    @testset "Subtraction" begin
        A = SparseArray{Int, 2}((3, 3))
        A[1, 1] = 10
        A[2, 2] = 15

        B = SparseArray{Int, 2}((3, 3))
        B[1, 1] = 3
        B[2, 2] = 5
        B[3, 3] = 7

        C = A - B
        @test C[1, 1] == 7
        @test C[2, 2] == 10
        @test C[3, 3] == -7
        @test !hasindex(C, 1, 2)
        @test nnz(C) == 3

        # Test result with zeros
        D = SparseArray{Int, 2}((2, 2))
        D[1, 1] = 5
        E = SparseArray{Int, 2}((2, 2))
        E[1, 1] = 5

        F = D - E
        @test !hasindex(F, 1, 1)  # Zero results should not be stored
        @test nnz(F) == 0
    end

    @testset "Scalar Multiplication" begin
        A = SparseArray{Float64, 2}((3, 3))
        A[1, 1] = 2.0
        A[2, 3] = -1.5

        # Test right multiplication
        B = A * 3
        @test B[1, 1] == 6.0
        @test B[2, 3] == -4.5
        @test !hasindex(B, 1, 2)
        @test nnz(B) == 2

        # Test left multiplication
        C = 3 * A
        @test C[1, 1] == 6.0
        @test C[2, 3] == -4.5

        # Test multiplication by zero
        D = A * 0
        @test nnz(D) == 0
        @test !hasindex(D, 1, 1)  # Zero results should not be stored

        # Test type promotion
        E = A * 2
        @test eltype(E) == Float64

        F = A * (1 + 0im)
        @test eltype(F) == Complex{Float64}
    end

    @testset "Mixed Type Operations" begin
        A = SparseArray{Int, 2}((2, 2))
        A[1, 1] = 5

        B = SparseArray{Float64, 2}((2, 2))
        B[1, 1] = 2.5
        B[2, 2] = 1.0

        C = A + B
        @test eltype(C) == Float64
        @test C[1, 1] == 7.5
        @test C[2, 2] == 1.0
    end
end

end
