module TestAqua

using Aqua
using MultidimensionalSparseArrays
using Test

@testset "Aqua" begin
    Aqua.test_ambiguities(MultidimensionalSparseArrays, recursive = false)
    Aqua.test_all(MultidimensionalSparseArrays, ambiguities = false)
    return nothing
end

end
