module TestConstruction

using MultidimensionalSparseArrays
using Test

@testset "Construction" begin
    # Basic construction
    A = SparseArray{Float64, 2}((3, 4))
    @test size(A) == (3, 4)
    @test eltype(A) == Float64
    @test nnz(A) == 0
    @test A.default_value == 0.0

    # Construction with custom default value
    B = SparseArray{Int, 2}((2, 2), -1)
    @test B.default_value == -1
    @test B[1, 1] == -1  # Should return default value

    # Convenience constructors
    C = SparseArray{Float64}((3, 3))
    @test size(C) == (3, 3)
    @test eltype(C) == Float64

    D = SparseArray{Int}(2, 3, 4)
    @test size(D) == (2, 3, 4)
    @test eltype(D) == Int

    # Construction from dense array
    dense = [1 0 3; 0 0 0; 2 0 4]
    E = SparseArray(dense)
    @test size(E) == (3, 3)
    @test E[1, 1] == 1
    @test E[1, 2] == 0
    @test E[1, 3] == 3
    @test E[2, 1] == 0
    @test E[3, 1] == 2
    @test E[3, 3] == 4
    @test nnz(E) == 4  # Only non-zero elements stored
end

end
