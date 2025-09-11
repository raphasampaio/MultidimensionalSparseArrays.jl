module TestDisplay

using NDimensionalSparseArrays
using Test

@testset "Display and Show Methods" begin
    @testset "Empty Array Display" begin
        A = SparseArray{Float64, 2}((3, 3))

        # Test full representation using show with text/plain MIME
        str = sprint(io -> show(io, "text/plain", A))
        @test occursin("SparseArray{Float64, 2}", str)
        @test occursin("0 stored entries", str)
        @test occursin("100.0% (9 zeros)", str)

        # Test compact display
        io = IOBuffer()
        show(IOContext(io, :compact => true), A)
        compact_str = String(take!(io))
        @test compact_str == "(3, 3) SparseArray{Float64, 2}"
    end

    @testset "Small Array Display" begin
        A = SparseArray{Int, 2}((2, 3))
        A[1, 1] = 5
        A[2, 2] = 10
        A[1, 3] = -2

        str = string(A)
        @test occursin("3 stored entries", str)
        @test occursin("50.0% (3 zeros)", str)
        @test occursin("CartesianIndex(1, 1)  =>  5", str)
        @test occursin("CartesianIndex(2, 2)  =>  10", str)
        @test occursin("CartesianIndex(1, 3)  =>  -2", str)
    end

    @testset "Large Array Display (>10 entries)" begin
        A = SparseArray{Int, 2}((5, 5))

        # Add more than 10 entries
        for i in 1:5, j in 1:3
            A[i, j] = i * 10 + j
        end

        str = string(A)
        @test occursin("15 stored entries", str)
        @test occursin("(5 more entries)", str)
        @test occursin("⋮", str)

        # Should only show first 10 entries
        lines = split(str, '\n')
        entry_lines = filter(line -> occursin("=>", line), lines)
        @test length(entry_lines) == 10
    end

    @testset "Different Number Types Display" begin
        # Complex numbers
        A = SparseArray{Complex{Float64}, 2}((2, 2))
        A[1, 1] = 1.0 + 2.0im
        A[2, 2] = -3.0 - 4.0im

        str = string(A)
        @test occursin("SparseArray{ComplexF64, 2}", str)
        @test occursin("1.0 + 2.0im", str)
        @test occursin("-3.0 - 4.0im", str)

        # Rational numbers
        B = SparseArray{Rational{Int}, 3}((2, 2, 2))
        B[1, 1, 1] = 1//2
        B[2, 2, 2] = 3//4

        str = string(B)
        @test occursin("SparseArray{Rational{Int64}, 3}", str)
        @test occursin("1//2", str)
        @test occursin("3//4", str)
    end

    @testset "High Dimensional Array Display" begin
        A = SparseArray{Float64, 4}((2, 2, 2, 2))
        A[1, 1, 1, 1] = 1.0
        A[2, 2, 2, 2] = 2.0

        str = string(A)
        @test occursin("SparseArray{Float64, 4}", str)
        @test occursin("CartesianIndex(1, 1, 1, 1)", str)
        @test occursin("CartesianIndex(2, 2, 2, 2)", str)
        @test occursin("87.5% (14 zeros)", str)  # 16 total - 2 stored = 14 zeros
    end

    @testset "Single Dimension Array Display" begin
        A = SparseArray{Int, 1}((5,))
        A[2] = 42
        A[4] = -17

        str = string(A)
        @test occursin("SparseArray{Int64, 1}", str)
        @test occursin("CartesianIndex(2,)", str)
        @test occursin("CartesianIndex(4,)", str)
        @test occursin("60.0% (3 zeros)", str)
    end

    @testset "Zero Values Display" begin
        A = SparseArray{Float64, 2}((3, 3))
        A[1, 1] = 0.0
        A[2, 2] = 0.0
        A[3, 3] = 5.0

        str = string(A)
        @test occursin("3 stored entries", str)
        @test occursin("0.0", str)
        @test occursin("5.0", str)
        @test occursin("66.67% (6 zeros)", str)
    end

    @testset "Edge Case: Very Large Sparse Array" begin
        A = SparseArray{Int, 2}((1000, 1000))
        A[500, 500] = 42

        str = string(A)
        @test occursin("1 stored entries", str)
        @test occursin("100.0% (999999 zeros)", str)
        @test occursin("CartesianIndex(500, 500)  =>  42", str)
    end

    @testset "Sparsity Calculation Precision" begin
        # Test precise sparsity calculation
        A = SparseArray{Int, 2}((3, 4))  # 12 total elements
        A[1, 1] = 1
        A[2, 2] = 2
        # 2 stored, 10 zeros -> 10/12 = 83.33%

        str = string(A)
        @test occursin("83.33% (10 zeros)", str)
    end
end

end
