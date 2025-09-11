module TestToDense

using NDimensionalSparseArrays
using Test

@testset "to_dense Function" begin
    @testset "Basic functionality" begin
        A = NDSparseArray{Int, 2}((3, 3))
        A[1, 1] = 5
        A[2, 3] = 10
        A[3, 1] = -2

        dense = to_dense(A)
        @test dense == [5 0 0; 0 0 10; -2 0 0]
        @test eltype(dense) == Int
        @test size(dense) == size(A)
    end

    @testset "Different types" begin
        B = NDSparseArray{Float64, 2}((2, 2))
        B[1, 2] = 3.14

        dense_float = to_dense(B)
        @test dense_float[1, 2] == 3.14
        @test dense_float[1, 1] == 0.0
        @test eltype(dense_float) == Float64
    end

    @testset "Complex numbers" begin
        C = NDSparseArray{Complex{Int}, 2}((2, 2))
        C[1, 1] = 1 + 2im

        dense_complex = to_dense(C)
        @test dense_complex[1, 1] == 1 + 2im
        @test dense_complex[2, 2] == 0 + 0im
        @test eltype(dense_complex) == Complex{Int}
    end

    @testset "Higher dimensions" begin
        D = NDSparseArray{Int, 3}((2, 2, 2))
        D[1, 1, 1] = 42
        D[2, 2, 2] = 24

        dense_3d = to_dense(D)
        @test dense_3d[1, 1, 1] == 42
        @test dense_3d[2, 2, 2] == 24
        @test all(dense_3d[i, j, k] == 0 for i in 1:2, j in 1:2, k in 1:2 if (i, j, k) ∉ [(1, 1, 1), (2, 2, 2)])
    end

    @testset "Empty array" begin
        E = NDSparseArray{Int, 2}((3, 3))
        dense_empty = to_dense(E)
        @test all(dense_empty .== 0)
        @test size(dense_empty) == (3, 3)
    end

    @testset "Single element arrays" begin
        F = NDSparseArray{Float32, 1}((1,))
        F[1] = 5.5f0

        dense_single = to_dense(F)
        @test dense_single == [5.5f0]
        @test eltype(dense_single) == Float32
    end

    @testset "Zero-dimensional arrays" begin
        G = NDSparseArray{Int, 0}(())
        G[] = 99

        dense_0d = to_dense(G)
        @test dense_0d[] == 99
        @test ndims(dense_0d) == 0
    end

    @testset "Large sparse arrays" begin
        H = NDSparseArray{Int, 2}((100, 100))
        H[1, 1] = 1
        H[50, 50] = 2
        H[100, 100] = 3

        dense_large = to_dense(H)
        @test dense_large[1, 1] == 1
        @test dense_large[50, 50] == 2
        @test dense_large[100, 100] == 3
        @test count(!iszero, dense_large) == 3
        @test size(dense_large) == (100, 100)
    end
end

end
