module TestMultidimensionalArrays

using NDimensionalSparseArrays
using Test

@testset "Multidimensional Arrays" begin
    # 3D array
    A = SparseArray{Float64, 3}((2, 2, 2))
    A[1, 1, 1] = 1.0
    A[2, 2, 2] = 8.0

    @test A[1, 1, 1] == 1.0
    @test A[2, 2, 2] == 8.0
    @test_throws BoundsError A[1, 2, 1]  # Unset index throws error
    @test nnz(A) == 2

    # 4D array
    B = SparseArray{Int, 4}((2, 2, 2, 2))
    B[1, 1, 1, 1] = 100
    @test B[1, 1, 1, 1] == 100
    @test nnz(B) == 1
end

end
