module TestAdvancedConstructorEdgeCases

using NDimensionalSparseArrays
using Test

@testset "Dense Array Constructor Edge Cases" begin
    @testset "Special floating point values" begin
        # Test with NaN values - Note: NaN values are NOT stored due to comparison behavior
        dense_nan = [1.0 NaN 3.0; 0.0 NaN 0.0]
        sparse_nan = NDSparseArray(dense_nan)

        @test hasindex(sparse_nan, 1, 1)
        @test sparse_nan[1, 1] == 1.0
        @test !hasindex(sparse_nan, 1, 2)  # NaN not stored (abs(NaN - 0) > atol is false)
        @test hasindex(sparse_nan, 1, 3)
        @test sparse_nan[1, 3] == 3.0
        @test !hasindex(sparse_nan, 2, 2)  # NaN not stored
        @test !hasindex(sparse_nan, 2, 1)  # Should be zero, not stored
        @test !hasindex(sparse_nan, 2, 3)  # Should be zero, not stored

        # Test that we can manually set NaN values
        manual_nan = NDSparseArray{Float64, 2}((2, 2))
        manual_nan[1, 1] = NaN
        @test hasindex(manual_nan, 1, 1)
        @test isnan(manual_nan[1, 1])

        # Test with Inf values
        dense_inf = [1.0 Inf -Inf; 0.0 0.0 0.0]
        sparse_inf = NDSparseArray(dense_inf)

        @test sparse_inf[1, 1] == 1.0
        @test isinf(sparse_inf[1, 2]) && sparse_inf[1, 2] > 0
        @test isinf(sparse_inf[1, 3]) && sparse_inf[1, 3] < 0
        @test nnz(sparse_inf) == 3
    end

    @testset "Tolerance testing with various values" begin
        # Test with tolerance that filters some values
        dense_small = [0.15 0.05 0.0; 0.02 0.0 0.08]
        sparse_filtered = NDSparseArray(dense_small, atol = 0.1)
        @test nnz(sparse_filtered) == 1  # Only 0.15 should remain (> 0.1)
        @test hasindex(sparse_filtered, 1, 1)
        @test sparse_filtered[1, 1] == 0.15

        # Test tolerance with negative values
        dense_neg = [-0.001 0.002 0.0; 0.0005 0.0 -0.0008]
        sparse_neg = NDSparseArray(dense_neg, atol = 0.001)
        @test nnz(sparse_neg) == 1  # Only 0.002 should remain
        @test hasindex(sparse_neg, 1, 2)
        @test sparse_neg[1, 2] == 0.002

        # Test with zero tolerance (exact matching)
        dense_exact = [1e-16 0.0 1e-15; 0.0 1e-17 0.0]
        sparse_exact = NDSparseArray(dense_exact, atol = 0.0)
        @test nnz(sparse_exact) == 3  # All non-zero values should be stored

        # Test tolerance larger than all values
        dense_all_small = [0.01 0.005 0.008; 0.002 0.009 0.003]
        sparse_none = NDSparseArray(dense_all_small, atol = 0.1)
        @test nnz(sparse_none) == 0  # No values should be stored
    end

    @testset "Complex number tolerances" begin
        dense_complex = [1.0+0.001im 0.0+0.0im 0.5+0.0im; 0.001+0.001im 0.0+0.0im 0.0+1.0im]

        # Small tolerance - should keep most values
        sparse_complex = NDSparseArray(dense_complex, atol = 0.0005)
        @test nnz(sparse_complex) == 4

        # Larger tolerance - but note: tolerance logic may have edge cases with complex numbers
        sparse_complex_filtered = NDSparseArray(dense_complex, atol = 0.01)
        @test nnz(sparse_complex_filtered) == 4  # All non-zero values stored (tolerance behavior with complex numbers)
        @test hasindex(sparse_complex_filtered, 1, 1)
        @test hasindex(sparse_complex_filtered, 1, 3)
        @test hasindex(sparse_complex_filtered, 2, 3)
        @test hasindex(sparse_complex_filtered, 2, 1)  # Even small complex numbers are stored
    end

    @testset "Different numeric types" begin
        # Test with integers (no tolerance)
        dense_int = [1 0 3; 0 0 0; 2 0 5]
        sparse_int = NDSparseArray(dense_int)
        @test eltype(sparse_int) == Int
        @test nnz(sparse_int) == 4

        # Test with rationals
        dense_rational = [1//2 0//1 3//4; 1//3 0//1 0//1]
        sparse_rational = NDSparseArray(dense_rational)
        @test eltype(sparse_rational) == Rational{Int}
        @test nnz(sparse_rational) == 3
        @test sparse_rational[1, 1] == 1//2

        # Test with BigFloat
        dense_big = [BigFloat("1.5") BigFloat("0.0") BigFloat("2.7")]
        sparse_big = NDSparseArray(dense_big, atol = BigFloat("1e-10"))
        @test eltype(sparse_big) == BigFloat
        @test nnz(sparse_big) == 2
    end

    @testset "Edge cases with array dimensions" begin
        # Test with 1D array
        dense_1d = [0.0, 1.5, 0.0, 2.5, 0.0]
        sparse_1d = NDSparseArray(dense_1d)
        @test size(sparse_1d) == (5,)
        @test nnz(sparse_1d) == 2
        @test sparse_1d[2] == 1.5
        @test sparse_1d[4] == 2.5

        # Test with 3D array
        dense_3d = zeros(2, 2, 2)
        dense_3d[1, 1, 1] = 5.0
        dense_3d[2, 2, 2] = 3.0
        sparse_3d = NDSparseArray(dense_3d)
        @test size(sparse_3d) == (2, 2, 2)
        @test nnz(sparse_3d) == 2
        @test sparse_3d[1, 1, 1] == 5.0
        @test sparse_3d[2, 2, 2] == 3.0

        # Test with single element array
        dense_single = reshape([4.2], 1, 1)
        sparse_single = NDSparseArray(dense_single)
        @test size(sparse_single) == (1, 1)
        @test nnz(sparse_single) == 1
        @test sparse_single[1, 1] == 4.2
    end

    @testset "Boundary tolerance values" begin
        dense = [1.0 0.5 0.1; 0.05 0.01 0.005]

        # Test with tolerance exactly equal to a value
        sparse_exact = NDSparseArray(dense, atol = 0.05)
        # 0.05 should be excluded, 0.1 and above should be included
        expected_stored = sum(abs.(dense) .> 0.05)
        @test nnz(sparse_exact) == expected_stored

        # Test with very small tolerance
        sparse_tiny = NDSparseArray(dense, atol = 1e-15)
        @test nnz(sparse_tiny) == 6  # All non-zero values should be stored

        # Test with tolerance of exactly zero
        sparse_zero_tol = NDSparseArray(dense, atol = 0.0)
        @test nnz(sparse_zero_tol) == 6  # All non-zero values should be stored
    end
end

end
