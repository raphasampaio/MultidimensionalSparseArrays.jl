module TestFindingFunctions

using MultidimensionalSparseArrays
using Test

@testset "Finding Functions" begin
    @testset "findnz" begin
        A = SparseArray{Int, 2}((3, 4))
        A[1, 1] = 5
        A[2, 3] = -2
        A[3, 4] = 10
        
        indices, values = findnz(A)
        
        @test length(indices) == 3
        @test length(values) == 3
        @test length(indices) == length(values)
        
        # Check that all stored elements are found
        @test CartesianIndex(1, 1) in indices
        @test CartesianIndex(2, 3) in indices
        @test CartesianIndex(3, 4) in indices
        
        @test 5 in values
        @test -2 in values
        @test 10 in values
        
        # Test with empty array
        B = SparseArray{Float64, 2}((2, 2))
        indices_empty, values_empty = findnz(B)
        @test length(indices_empty) == 0
        @test length(values_empty) == 0
    end
    
    @testset "findall with functions" begin
        A = SparseArray{Int, 2}((3, 3))
        A[1, 1] = 5
        A[1, 2] = -3
        A[2, 2] = 10
        A[3, 3] = 0  # This should not be stored (equals default)
        
        # Find positive values
        pos_indices = findall(x -> x > 0, A)
        @test CartesianIndex(1, 1) in pos_indices
        @test CartesianIndex(2, 2) in pos_indices
        @test !(CartesianIndex(1, 2) in pos_indices)  # -3 is not positive
        
        # Find negative values
        neg_indices = findall(x -> x < 0, A)
        @test CartesianIndex(1, 2) in neg_indices
        @test length(neg_indices) == 1
        
        # Find zeros (should include default values)
        zero_indices = findall(x -> x == 0, A)
        # Should include all positions except stored non-zero values
        expected_zeros = 9 - 3  # 9 total positions - 3 stored non-zero values
        @test length(zero_indices) == expected_zeros
        @test !(CartesianIndex(1, 1) in zero_indices)
        @test !(CartesianIndex(1, 2) in zero_indices)
        @test !(CartesianIndex(2, 2) in zero_indices)
        
        # Find values equal to specific number
        five_indices = findall(x -> x == 5, A)
        @test length(five_indices) == 1
        @test CartesianIndex(1, 1) in five_indices
        
        # Test with empty result
        huge_indices = findall(x -> x > 1000, A)
        @test length(huge_indices) == 0
    end
    
    @testset "findall with different predicates" begin
        A = SparseArray{Float64, 2}((2, 3))
        A[1, 1] = 2.5
        A[1, 3] = -1.5
        A[2, 2] = 0.0  # Should not be stored
        
        # Find non-zero values (should only find stored values that aren't default)
        nonzero_indices = findall(!iszero, A)
        @test length(nonzero_indices) == 2
        @test CartesianIndex(1, 1) in nonzero_indices
        @test CartesianIndex(1, 3) in nonzero_indices
        
        # Find values greater than 1
        gt_one_indices = findall(x -> x > 1, A)
        @test length(gt_one_indices) == 1
        @test CartesianIndex(1, 1) in gt_one_indices
        
        # Find values using isfinite (all should be finite)
        finite_indices = findall(isfinite, A)
        @test length(finite_indices) == 6  # All positions
        
        # Test with complex predicate
        complex_indices = findall(x -> abs(x) > 1.0 && x < 0, A)
        @test length(complex_indices) == 1
        @test CartesianIndex(1, 3) in complex_indices
    end
    
    @testset "findall performance with sparse arrays" begin
        # Create a large sparse array
        A = SparseArray{Int, 2}((100, 100))
        
        # Add only a few non-zero elements
        A[10, 10] = 100
        A[50, 50] = 200
        A[90, 90] = 300
        
        # Find large values - should be efficient since we only check stored values
        # and then decide about default values
        large_indices = findall(x -> x > 150, A)
        @test length(large_indices) == 2
        @test CartesianIndex(50, 50) in large_indices
        @test CartesianIndex(90, 90) in large_indices
        
        # Find zeros - this tests the default value case
        zero_indices = findall(x -> x == 0, A)
        @test length(zero_indices) == 10000 - 3  # Total minus stored non-zeros
    end
    
    @testset "Edge cases for finding functions" begin
        # Test with 1D array
        A = SparseArray{Int, 1}((5,))
        A[2] = 42
        A[4] = -7
        
        indices, values = findnz(A)
        @test length(indices) == 2
        @test CartesianIndex(2) in indices
        @test CartesianIndex(4) in indices
        
        positive_indices = findall(x -> x > 0, A)
        @test length(positive_indices) == 1
        @test CartesianIndex(2) in positive_indices
        
        # Test with 3D array
        B = SparseArray{Float64, 3}((2, 2, 2))
        B[1, 1, 1] = 1.0
        B[2, 2, 2] = 8.0
        
        indices_3d, values_3d = findnz(B)
        @test length(indices_3d) == 2
        @test CartesianIndex(1, 1, 1) in indices_3d
        @test CartesianIndex(2, 2, 2) in indices_3d
        
        # Test with custom default value
        C = SparseArray{Int, 2}((2, 2), -1)  # Default is -1
        C[1, 1] = 5
        C[2, 2] = -1  # Should not be stored since it equals default
        
        default_indices = findall(x -> x == -1, C)
        @test length(default_indices) == 3  # Three positions have default value
        @test !(CartesianIndex(1, 1) in default_indices)
    end
end

end