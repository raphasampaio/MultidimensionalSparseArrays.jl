module TestConstruction

using NDimensionalSparseArrays
using Test

@testset "Construction" begin
    # Basic construction
    A = SparseArray{Float64, 2}((3, 4))
    @test size(A) == (3, 4)
    @test eltype(A) == Float64
    @test nnz(A) == 0

    # Test that unset indices throw errors
    @test_throws BoundsError A[1, 1]

    # Test basic functionality
    A[1, 1] = 5.0
    @test A[1, 1] == 5.0
    @test nnz(A) == 1
    @test hasindex(A, 1, 1)
    @test !hasindex(A, 1, 2)

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
    @test !hasindex(E, 1, 2)  # Zero values not stored
    @test E[1, 3] == 3
    @test !hasindex(E, 2, 1)
    @test E[3, 1] == 2
    @test E[3, 3] == 4
    @test nnz(E) == 4  # Only non-zero elements stored

    # Array-like constructor with undef
    F = SparseArray{Float64, 2}(undef, 2, 3)
    @test size(F) == (2, 3)
    @test eltype(F) == Float64
    @test nnz(F) == 0
    # Array starts empty - no values stored

    G = SparseArray{Int, 3}(undef, (2, 2, 2))
    @test size(G) == (2, 2, 2)
    @test eltype(G) == Int
    @test nnz(G) == 0
end

end
