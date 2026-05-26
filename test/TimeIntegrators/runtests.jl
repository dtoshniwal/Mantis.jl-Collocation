module TimeIntegratorsTests

using Test

@testset verbose=true "Advection-Based Test" begin
    include("AdvectionTests.jl")
end
@testset verbose=true "Convergence Tests" begin
    include("ConvergenceTests.jl")
end
@testset verbose=true "Amplification Factors Single-Step Multi-Stage" begin
    include("StabilityTests.jl")
end

end
