module TestAqua

using Aqua
using Banners
using Test

@testset "Aqua" begin
    Aqua.test_ambiguities(Banners, recursive = false)
    Aqua.test_all(Banners, ambiguities = false)
    return nothing
end

end
