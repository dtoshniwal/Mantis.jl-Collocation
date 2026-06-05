module TimeIntegrationStabilityTests

import SparseArrays
import LinearAlgebra
using Mantis
using Test

# We can tests (some) numerical schemes by performing a numerical Von Neumann analysis over
# one time step and then check if the computed eigenmode is the same as predicted by theory.
# The theoretical stability functions are exact, so we can match the value to machine
# precision.
# Because the eigenmodes are expressed as complex numbers, this test also ensures that we
# can use more general number types than just Float64.

# We use the advection equation in 1D with a simple finite difference spatial
# discretisation to verify stability in the case where we solve for multiple variables.

# Consider the advection equation on a 1D domain (x_L, x_R) as
# dc/dt = Ac
# where c is the vector of unknowns, and A is the (N-1, N-1) discretisation matrix.

const dx = 0.01
const L = 1.0
const velocity = 0.1
const dt = dx / velocity  # gives CFL = 1
const grid = 0.0:dx:L
const A =
    (velocity / dx) *
    SparseArrays.spdiagm(0 => ones(length(grid)), -1 => -1 .* ones(length(grid)-1))
# Set periodic boundary conditions.
A[1, end] = -(velocity / dx)

function discretised_advection_equation(c, t)
    return -A * c
end

c_0 = sinpi.(2.0 .* grid)

linear_advection_ode = TimeIntegrators.define_explicit_ode(discretised_advection_equation)

# Amplification factor analysis ------------------------------------------------------------
const k = 2 * pi  # wavenumber
const ck_0 = exp.(im .* k .* grid)

# From the spatial discretisation:
const lambda_k = - velocity * (1 - exp(-im * k * dx)) / dx

const z_k = dt * lambda_k

# Known stability functions:
# Note that these functions hold for rk methods of the same order if they have the same
# number of stages. This means that, for example, HEUN2 and RALSTON2 have the same
# stability function because they are both 2-stage RK methods of order 2.
function exact_stability(z, order)
    if order == 1
        return 1 + z
    elseif order == 2
        return 1 + z + 1/2 * z^2
    elseif order == 3
        return 1 + z + 1/2 * z^2 + 1/6 * z^3
    elseif order == 4
        return 1 + z + 1/2 * z^2 + 1/6 * z^3 + 1/24 * z^4
    end
end

const integrators = (
    TimeIntegrators.FORWARD_EULER,
    TimeIntegrators.EXPLICIT_MIDPOINT,
    TimeIntegrators.HEUN2,
    TimeIntegrators.RALSTON2,
    TimeIntegrators.HEUN3,
    TimeIntegrators.RK3,
    TimeIntegrators.RALTSON3,
    TimeIntegrators.VDHW3,
    TimeIntegrators.SSPRK3,
    TimeIntegrators.RK4,
    TimeIntegrators.RK4_3_8,
    TimeIntegrators.RALTSON4,
)

@testset "Explicit Integrators" verbose = true begin
    foreach(integrators) do scheme
        ck_n = TimeIntegrators.initialize_scheme(ck_0, scheme)
        TimeIntegrators.time_integrate!(ck_n, linear_advection_ode, 0.0, dt)

        # Pick just one factor to test (away from the boundary condition).
        amplifaction_factor_scheme = (TimeIntegrators.get_solution(ck_n) ./ ck_0)[14]
        @test isapprox(
            exact_stability(z_k, TimeIntegrators.get_order(scheme)),
            amplifaction_factor_scheme,
            rtol=1e-15,
        )
    end
end

# Known stability functions for diagonally implicit schemes. Here, the stability functions
# are usually not the same per order, so we specify them per integrator.
const gamma4 = (1 + TimeIntegrators._α_DIRK4) / 2
const diagonally_implicit_integrators = (
    (TimeIntegrators.BACKWARD_EULER, z -> 1 / (1-z)),
    (TimeIntegrators.RADAU_IA_1, z -> 1 / (1-z)),
    (TimeIntegrators.IMPLICIT_MIDPOINT, z -> (1 + 1/2 * z) / (1 - 1/2*z)),
    (
        TimeIntegrators.DIRK2,
        z ->
            (
                1 +
                (1 - 2*TimeIntegrators._α_DIRK2)z +
                (TimeIntegrators._α_DIRK2^2 - 2*TimeIntegrators._α_DIRK2 + 1/2) * z^2
            ) / (1 - TimeIntegrators._α_DIRK2*z)^2,
    ),
    (
        TimeIntegrators.DIRK3,
        z ->
            (
                1 +
                (1 - 2*(1/2 + sqrt(3)/6))z +
                ((1/2 + sqrt(3)/6)^2 - 2*(1/2 + sqrt(3)/6) + 1/2) * z^2
            ) / (1 - (1/2 + sqrt(3)/6)*z)^2,
    ),
    (
        TimeIntegrators.DIRK4,
        z ->
            (
                1 +
                (1 - 3*gamma4)z +
                (3*gamma4^2 - 3*gamma4 + 1/2) * z^2 +
                (-gamma4^3 + 3*gamma4^2 - 3/2*gamma4 + 1/6) * z^3
            ) / (1 - gamma4*z)^3,
    ),
)

implicit_linear_advection_ode = TimeIntegrators.define_implicit_linear(
    nothing, -A, nothing, (c, t) -> -A * c
)

@testset "Diagonally Implicit Integrators" verbose = true begin
    foreach(diagonally_implicit_integrators) do (scheme, exact_stability_function)
        ck_n = TimeIntegrators.initialize_scheme(ck_0, scheme)
        TimeIntegrators.time_integrate!(ck_n, implicit_linear_advection_ode, 0.0, dt)

        # Pick just one factor to test (away from the boundary condition).
        amplifaction_factor_scheme = (TimeIntegrators.get_solution(ck_n) ./ ck_0)[84]
        @test isapprox(
            exact_stability_function(z_k), amplifaction_factor_scheme, rtol=1e-15
        )
    end
end

# const implicit_integrators = (
#     (TimeIntegrators.RADAU_IA_3, z -> (1 + z/3) / (1 - 2*z/3 - (z^2)/6)),
#     (TimeIntegrators.GAUSS_LEGENDRE_4, z -> (1 + z/2 + (z^2)/12) / (1 - z/2 + (z^2)/12)),
# )

# function large_solve(x, λ, t; num_stages)
#     bigA = SparseArrays.blockdiag([-A for i in 1:num_stages]...)
#     return (LinearAlgebra.I - λ * bigA) \ x
# end
# function large_eval(c; num_stages)
#     bigA = SparseArrays.blockdiag([-A for i in 1:num_stages]...)
#     return -bigA * c
# end
# implicit_linear_advection_ode = TimeIntegrators.define_implicit_ode(large_solve, large_eval)
# # TimeIntegrators.define_implicit_linear(
# #     nothing, -A, nothing, (c, t) -> -A * c
# # )

# @testset "Amplification Factors Implicit Integrators" verbose = true begin
#     foreach(implicit_integrators) do (scheme, exact_stability_function)
#         @show scheme
#         ck_n = TimeIntegrators.initialize_scheme(ck_0, scheme)
#         TimeIntegrators.time_integrate!(ck_n, implicit_linear_advection_ode, 0.0, dt; num_stages=TimeIntegrators.get_num_stages(scheme))

#         # Pick just one factor to test (away from the boundary condition).
#         amplifaction_factor_scheme = (TimeIntegrators.get_solution(ck_n) ./ ck_0)#[84]
#         @show amplifaction_factor_scheme
#         @test isapprox(
#             exact_stability_function(z_k), amplifaction_factor_scheme[84], rtol=1e-15
#         )
#     end
# end

end
