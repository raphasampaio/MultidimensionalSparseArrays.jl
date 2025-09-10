module TestStrings

using Test
using Banners

@testset "Strings" begin
    @testset "string_to_matrix" begin
        # Test simple string
        result = Banners.string_to_matrix("ABC\nDEF")
        @test length(result) == 2
        @test result[1] == ['A', 'B', 'C']
        @test result[2] == ['D', 'E', 'F']

        # Test empty string
        result = Banners.string_to_matrix("")
        @test length(result) == 1
        @test result[1] == Char[]

        # Test single line
        result = Banners.string_to_matrix("Hello")
        @test length(result) == 1
        @test result[1] == ['H', 'e', 'l', 'l', 'o']

        # Test lines with different lengths
        result = Banners.string_to_matrix("A\nBCD\nEF")
        @test length(result) == 3
        @test result[1] == ['A']
        @test result[2] == ['B', 'C', 'D']
        @test result[3] == ['E', 'F']
    end

    @testset "matrix_to_string" begin
        # Test simple matrix
        matrix = [['A', 'B'], ['C', 'D']]
        result = Banners.matrix_to_string(matrix)
        @test result == "AB\nCD"

        # Test empty matrix
        matrix = Vector{Vector{Char}}()
        result = Banners.matrix_to_string(matrix)
        @test result == ""

        # Test matrix with single row
        matrix = [['H', 'e', 'l', 'l', 'o']]
        result = Banners.matrix_to_string(matrix)
        @test result == "Hello"

        # Test matrix with mixed types (AnnotatedString)
        matrix = [["A", "B"], ["C", "D"]]
        result = Banners.matrix_to_string(matrix)
        @test result == "AB\nCD"
    end

    @testset "Round trip conversion" begin
        original = "Hello\nWorld\n!!!"
        matrix = Banners.string_to_matrix(original)
        converted = Banners.matrix_to_string(matrix)
        @test converted == original
    end
end
end
