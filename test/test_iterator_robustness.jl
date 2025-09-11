module TestIteratorRobustness

using NDimensionalSparseArrays
using Test

@testset "Iterator Robustness" begin
    @testset "Basic iteration" begin
        A = NDSparseArray{Int, 2}((3, 3))
        A[1, 1] = 1
        A[2, 2] = 4
        A[3, 1] = 9

        # Collect all values through iteration
        values = collect(A)
        @test length(values) == 3
        @test sort(values) == [1, 4, 9]

        # Test that iteration order is consistent
        first_pass = collect(A)
        second_pass = collect(A)
        @test first_pass == second_pass
    end

    @testset "Empty array iteration" begin
        B = NDSparseArray{Int, 2}((2, 2))

        # Empty array should have no iterations
        @test iterate(B) === nothing
        # Don't test iterate(B, nothing) as it's not valid for empty arrays
        @test collect(B) == Int[]
        @test length(collect(B)) == 0
    end

    @testset "Single element iteration" begin
        C = NDSparseArray{String, 1}((5,))
        C[3] = "hello"

        values = collect(C)
        @test length(values) == 1
        @test values[1] == "hello"

        # Manual iteration
        iter_result = iterate(C)
        @test iter_result !== nothing
        val, state = iter_result
        @test val == "hello"

        # Next iteration should be nothing
        @test iterate(C, state) === nothing
    end

    @testset "Iteration after modifications" begin
        D = NDSparseArray{Int, 2}((3, 3))
        D[1, 1] = 10
        D[2, 2] = 20

        # Get initial state
        iter_state = iterate(D)
        @test iter_state !== nothing
        val, state = iter_state
        @test val in [10, 20]

        # Modify array by adding element
        D[3, 3] = 30

        # Continuing iteration should work (though order may vary)
        remaining_values = [val]
        next_iter = iterate(D, state)
        while next_iter !== nothing
            push!(remaining_values, next_iter[1])
            next_iter = iterate(D, next_iter[2])
        end

        # Should have collected all values
        @test length(remaining_values) >= 2  # At least the ones from before modification
        @test 10 in remaining_values || 20 in remaining_values
    end

    @testset "Iteration with deletions" begin
        E = NDSparseArray{Int, 2}((3, 3))
        E[1, 1] = 100
        E[2, 2] = 200
        E[3, 3] = 300

        original_values = sort(collect(E))
        @test original_values == [100, 200, 300]

        # Delete one element
        delete!(E, 2, 2)

        # Iteration should now return only remaining elements
        remaining_values = sort(collect(E))
        @test remaining_values == [100, 300]
        @test length(remaining_values) == 2
    end

    @testset "Large array iteration efficiency" begin
        # Test that iteration works efficiently with many elements
        F = NDSparseArray{Int, 2}((100, 100))

        # Add many elements
        for i in 1:50
            F[i, i] = i^2
        end

        @test nnz(F) == 50

        # Iteration should return all values
        values = collect(F)
        @test length(values) == 50
        @test sort(values) == [i^2 for i in 1:50]

        # Test manual iteration doesn't fail
        count = 0
        iter_state = iterate(F)
        while iter_state !== nothing
            count += 1
            iter_state = iterate(F, iter_state[2])
        end
        @test count == 50
    end

    @testset "Iteration with different types" begin
        # Test with floating point
        G = NDSparseArray{Float64, 2}((2, 2))
        G[1, 1] = 3.14
        G[2, 2] = 2.71

        float_values = collect(G)
        @test length(float_values) == 2
        @test 3.14 in float_values
        @test 2.71 in float_values

        # Test with complex numbers
        H = NDSparseArray{Complex{Int}, 2}((2, 2))
        H[1, 1] = 1 + 2im
        H[2, 2] = 3 + 4im

        complex_values = collect(H)
        @test length(complex_values) == 2
        @test (1 + 2im) in complex_values
        @test (3 + 4im) in complex_values

        # Test with strings
        I = NDSparseArray{String, 2}((2, 2))
        I[1, 2] = "first"
        I[2, 1] = "second"

        string_values = collect(I)
        @test length(string_values) == 2
        @test "first" in string_values
        @test "second" in string_values
    end

    @testset "Iterator state consistency" begin
        J = NDSparseArray{Int, 3}((2, 2, 2))
        J[1, 1, 1] = 1
        J[1, 2, 1] = 2
        J[2, 1, 2] = 3
        J[2, 2, 2] = 4

        # Multiple manual iterations should give same results
        manual_values1 = Int[]
        iter_state = iterate(J)
        while iter_state !== nothing
            push!(manual_values1, iter_state[1])
            iter_state = iterate(J, iter_state[2])
        end

        manual_values2 = Int[]
        iter_state = iterate(J)
        while iter_state !== nothing
            push!(manual_values2, iter_state[1])
            iter_state = iterate(J, iter_state[2])
        end

        @test sort(manual_values1) == sort(manual_values2)
        @test sort(manual_values1) == [1, 2, 3, 4]
    end

    @testset "Zero-dimensional array iteration" begin
        K = NDSparseArray{Int, 0}(())

        # Empty 0-d array
        @test iterate(K) === nothing
        @test collect(K) == Int[]

        # 0-d array with value
        K[] = 42
        values = collect(K)
        @test length(values) == 1
        @test values[1] == 42
    end
end

end
