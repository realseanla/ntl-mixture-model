using Test
using TestSetExtensions

include("tests.jl")

@testset ExtendedTestSet "Changepoint" begin
    Tests.changepoint_test()
end