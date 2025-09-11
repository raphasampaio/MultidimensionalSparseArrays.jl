
module TestEqualityAndCopying

using NDimensionalSparseArrays
using Test

@testset "Equality and Copying" begin
    A = NDSparseArray{Int, 2}((3, 3))
    A[1, 1] = 5
    A[2, 2] = 10

    B = NDSparseArray{Int, 2}((3, 3))
    B[1, 1] = 5
    B[2, 2] = 10

    @test A == B

    # Different values
    B[1, 1] = 6
    @test A != B

    # Different sizes
    C = NDSparseArray{Int, 2}((2, 2))
    @test A != C

    # Copy
    D = copy(A)
    @test A == D
    @test A.data !== D.data  # Different objects

    # Modify copy shouldn't affect original
    D[3, 3] = 15
    @test A != D
    @test !hasindex(A, 3, 3)  # A doesn't have this index
    @test D[3, 3] == 15
end

end
