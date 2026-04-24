module TimeIntegratorsTests

using Test

@testset verbose=true "AdvectionBasedTests" begin
    include("AdvectionTests.jl")
end
@testset verbose=true "ConvergenceTests" begin
    include("ConvergenceTests.jl")
end
@testset verbose=true "StabilityTests" begin
    include("StabilityTests.jl")
end

end
