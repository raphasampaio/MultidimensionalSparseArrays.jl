module TestAqua

using Aqua
using NDimensionalSparseArrays
using Test

@testset "Aqua" begin
    Aqua.test_ambiguities(NDimensionalSparseArrays, recursive = false)
    Aqua.test_all(NDimensionalSparseArrays, ambiguities = false)
    return nothing
end

end
