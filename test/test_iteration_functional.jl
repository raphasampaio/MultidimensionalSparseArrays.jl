module TestIterationFunctional

using NDimensionalSparseArrays
using Test

@testset "Iteration and Functional Programming" begin
    @testset "Basic Iteration Over Stored Values" begin
        A = NDSparseArray{Int, 2}((3, 4))
        A[1, 2] = 5
        A[2, 1] = 10
        A[3, 4] = -3

        # Test collect returns only stored values
        values = collect(A)
        @test length(values) == 3
        @test Set(values) == Set([5, 10, -3])

        # Test iteration order is consistent
        values1 = collect(A)
        values2 = collect(A)
        @test values1 == values2

        # Test that iteration doesn't include unset values
        for val in A
            @test val in [5, 10, -3]
        end
    end

    @testset "Empty Array Iteration" begin
        A = NDSparseArray{Float64, 2}((3, 3))

        values = collect(A)
        @test length(values) == 0
        @test values == Float64[]

        # Test iteration over empty array
        count = 0
        for val in A
            count += 1
        end
        @test count == 0
    end

    @testset "Iteration with Different Types" begin
        # Complex numbers
        A = NDSparseArray{Complex{Float64}, 2}((2, 2))
        A[1, 1] = 1.0 + 2.0im
        A[2, 2] = -3.0 + 4.0im

        values = collect(A)
        @test length(values) == 2
        @test (1.0 + 2.0im) in values
        @test (-3.0 + 4.0im) in values

        # Rational numbers
        B = NDSparseArray{Rational{Int}, 2}((2, 2))
        B[1, 2] = 1//3
        B[2, 1] = 2//5

        values = collect(B)
        @test length(values) == 2
        @test (1//3) in values
        @test (2//5) in values

        # String values
        C = NDSparseArray{String, 2}((2, 2))
        C[1, 1] = "hello"
        C[2, 2] = "world"

        values = collect(C)
        @test length(values) == 2
        @test "hello" in values
        @test "world" in values
    end

    @testset "Stored Indices Iteration" begin
        A = NDSparseArray{Int, 3}((2, 3, 2))
        A[1, 1, 1] = 10
        A[2, 3, 1] = 20
        A[1, 2, 2] = 30

        indices = collect(stored_indices(A))
        @test length(indices) == 3
        @test CartesianIndex(1, 1, 1) in indices
        @test CartesianIndex(2, 3, 1) in indices
        @test CartesianIndex(1, 2, 2) in indices

        # Test that we can access values using these indices
        for idx in stored_indices(A)
            @test hasindex(A, idx)
            @test A[idx] in [10, 20, 30]
        end
    end

    @testset "Stored Values Iteration" begin
        A = NDSparseArray{Float64, 2}((3, 3))
        A[1, 1] = 1.5
        A[2, 3] = 2.7
        A[3, 1] = 0.0  # Explicitly stored zero

        values = collect(stored_values(A))
        @test length(values) == 3
        @test 1.5 in values
        @test 2.7 in values
        @test 0.0 in values

        # Test iteration order consistency
        values1 = collect(stored_values(A))
        values2 = collect(stored_values(A))
        @test values1 == values2
    end

    @testset "Stored Pairs Iteration" begin
        A = NDSparseArray{Int, 2}((2, 3))
        A[1, 2] = 42
        A[2, 1] = 17
        A[2, 3] = -5

        pairs = collect(stored_pairs(A))
        @test length(pairs) == 3

        # Convert to dictionary for easier testing
        dict = Dict(pairs)
        @test dict[CartesianIndex(1, 2)] == 42
        @test dict[CartesianIndex(2, 1)] == 17
        @test dict[CartesianIndex(2, 3)] == -5

        # Test that we can reconstruct the array
        B = NDSparseArray{Int, 2}((2, 3))
        for (idx, val) in stored_pairs(A)
            B[idx] = val
        end
        @test A == B
    end

    @testset "Functional Programming Operations" begin
        A = NDSparseArray{Int, 2}((3, 3))
        A[1, 1] = 2
        A[1, 3] = 4
        A[2, 2] = 6
        A[3, 1] = 8

        # Test map-like operations over stored values
        squared_values = [val^2 for val in stored_values(A)]
        @test Set(squared_values) == Set([4, 16, 36, 64])

        # Test filter-like operations
        even_indices = [idx for (idx, val) in stored_pairs(A) if val % 2 == 0]
        @test length(even_indices) == 4  # All values are even

        large_values = [val for val in stored_values(A) if val > 4]
        @test Set(large_values) == Set([6, 8])

        # Test reduce-like operations
        sum_values = sum(stored_values(A))
        @test sum_values == 20  # 2 + 4 + 6 + 8

        max_value = maximum(stored_values(A))
        @test max_value == 8

        min_value = minimum(stored_values(A))
        @test min_value == 2
    end

    @testset "Enumerate with Stored Values" begin
        A = NDSparseArray{String, 2}((2, 2))
        A[1, 1] = "first"
        A[2, 2] = "second"

        enumerated = collect(enumerate(stored_values(A)))
        @test length(enumerated) == 2

        # Check that enumeration provides correct indices
        for (i, val) in enumerate(stored_values(A))
            @test i in [1, 2]
            @test val in ["first", "second"]
        end
    end

    @testset "Zip Operations" begin
        A = NDSparseArray{Int, 2}((2, 2))
        A[1, 1] = 1
        A[2, 2] = 4

        B = NDSparseArray{Int, 2}((2, 2))
        B[1, 1] = 10
        B[2, 2] = 40

        # Test zipping stored values
        pairs = collect(zip(stored_values(A), stored_values(B)))
        @test length(pairs) == 2

        # Note: order might not be guaranteed, so test membership
        @test (1, 10) in pairs || (4, 40) in pairs

        # Test zipping indices with values from different arrays
        indices_A = collect(stored_indices(A))
        values_B = collect(stored_values(B))

        if length(indices_A) == length(values_B)
            zipped = collect(zip(indices_A, values_B))
            @test length(zipped) == 2
        end
    end

    @testset "Generator Expressions" begin
        A = NDSparseArray{Int, 2}((3, 3))
        A[1, 1] = 1
        A[1, 3] = 3
        A[3, 1] = 5
        A[3, 3] = 7

        # Test generator over stored values
        doubled = (2 * val for val in stored_values(A))
        doubled_list = collect(doubled)
        @test Set(doubled_list) == Set([2, 6, 10, 14])

        # Test generator over stored pairs
        shifted_indices = (CartesianIndex(idx.I .+ 1) for (idx, val) in stored_pairs(A) if val > 2)
        # This would shift indices by 1, but we need to be careful about bounds
        shifted_list = collect(shifted_indices)
        @test length(shifted_list) == 3  # values 3, 5, 7 are > 2
    end

    @testset "Iteration State Consistency" begin
        A = NDSparseArray{Int, 2}((3, 3))
        A[1, 1] = 10
        A[2, 2] = 20
        A[3, 3] = 30

        # Test that multiple iterations give same result
        first_iteration = collect(A)
        second_iteration = collect(A)
        third_iteration = collect(A)

        @test first_iteration == second_iteration
        @test second_iteration == third_iteration

        # Test that modifying array during iteration doesn't break existing iterators
        # (This is more of a safety check)
        values_before = collect(A)
        A[1, 2] = 15  # Add new element
        values_after = collect(A)

        @test length(values_after) == length(values_before) + 1
        @test 15 in values_after
    end

    @testset "Large Array Iteration Performance" begin
        # Test that iteration is efficient for large sparse arrays
        A = NDSparseArray{Int, 2}((1000, 1000))

        # Add only a few elements to large array
        for i in 1:10
            A[i*100, i*100] = i
        end

        # Iteration should only visit stored elements
        values = collect(A)
        @test length(values) == 10
        @test Set(values) == Set(1:10)

        # Time complexity should be O(stored elements), not O(total elements)
        # This is more of a smoke test
        @test nnz(A) == 10
        @test length(collect(stored_indices(A))) == 10
    end
end

end
