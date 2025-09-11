module TestDeleteAdvanced

using NDimensionalSparseArrays
using Test

@testset "delete! Advanced Cases" begin
    @testset "Multiple deletions" begin
        A = NDSparseArray{Int, 2}((3, 3))
        A[1, 1] = 5

        # Test deleting same index multiple times
        delete!(A, 1, 1)
        @test !hasindex(A, 1, 1)

        # Should not error when deleting non-existent element
        delete!(A, 1, 1)
        @test !hasindex(A, 1, 1)
    end

    @testset "Linear indexing delete" begin
        A = NDSparseArray{Int, 2}((3, 3))
        A[2, 2] = 10
        A[3, 1] = 15

        # Test delete with CartesianIndex
        delete!(A, CartesianIndex(2, 2))
        @test !hasindex(A, 2, 2)
        @test hasindex(A, 3, 1)  # Other elements should remain
    end

    @testset "Return value" begin
        A = NDSparseArray{Int, 2}((3, 3))
        A[1, 1] = 5

        # Test that delete! returns the array
        result = delete!(A, 1, 1)
        @test result === A

        # Test with CartesianIndex variant
        A[2, 2] = 10
        result2 = delete!(A, CartesianIndex(2, 2))
        @test result2 === A
    end

    @testset "Delete from various array sizes" begin
        # 1D array
        A1 = NDSparseArray{Int, 1}((5,))
        A1[3] = 7
        delete!(A1, 3)
        @test !hasindex(A1, 3)

        # 3D array
        A3 = NDSparseArray{Int, 3}((2, 2, 2))
        A3[1, 2, 1] = 42
        delete!(A3, 1, 2, 1)
        @test !hasindex(A3, 1, 2, 1)

        # 4D array
        A4 = NDSparseArray{Int, 4}((2, 2, 2, 2))
        A4[1, 1, 2, 2] = 100
        delete!(A4, CartesianIndex(1, 1, 2, 2))
        @test !hasindex(A4, 1, 1, 2, 2)
    end

    @testset "Delete with bounds checking" begin
        A = NDSparseArray{Int, 2}((3, 3))
        A[1, 1] = 5

        # These should throw BoundsError
        @test_throws BoundsError delete!(A, 0, 1)
        @test_throws BoundsError delete!(A, 1, 4)
        @test_throws BoundsError delete!(A, CartesianIndex(4, 1))

        # Original element should still exist
        @test hasindex(A, 1, 1)
    end

    @testset "Delete preserves other elements" begin
        A = NDSparseArray{String, 2}((3, 3))
        A[1, 1] = "first"
        A[1, 2] = "second"
        A[2, 1] = "third"
        A[3, 3] = "fourth"

        original_nnz = nnz(A)
        delete!(A, 1, 2)

        @test nnz(A) == original_nnz - 1
        @test !hasindex(A, 1, 2)
        @test A[1, 1] == "first"
        @test A[2, 1] == "third"
        @test A[3, 3] == "fourth"
    end

    @testset "Delete all elements" begin
        A = NDSparseArray{Int, 2}((2, 2))
        A[1, 1] = 1
        A[1, 2] = 2
        A[2, 1] = 3
        A[2, 2] = 4

        delete!(A, 1, 1)
        delete!(A, 1, 2)
        delete!(A, 2, 1)
        delete!(A, 2, 2)

        @test nnz(A) == 0
        @test length(A.data) == 0
    end
end

end
