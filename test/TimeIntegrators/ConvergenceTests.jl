module TimeIntegrationConvergenceTests

using Mantis
using Test
import LinearAlgebra

# We can check the correctness of the implemented schemes by computing their rate of
# convergence and checking if this matches the expected rate. Here, we use a simple ODE
# with one unknown.
#
# Consider the ODE:
# dy/dt = lambda y,  y(t=0) = 1.0
# which has exact solution y(t) = exp(lambda t).

const lambda = -4
const t_final = 1

y_0 = 1.0
function exact_sol(t)
    return exp(lambda * t)
end

# Fully explicit ---------------------------------------------------------------------------
function test_ode_explicit_func(yn, t)
    return lambda * yn
end
test_ode_explicit = TimeIntegrators.define_explicit_ode(test_ode_explicit_func)

function implicitSolve(x, h, t)
    return (LinearAlgebra.I - h * lambda) \ x
end
test_ode_implicit = TimeIntegrators.define_implicit_ode(implicitSolve, x -> lambda * x)

test_ode_imex = TimeIntegrators.define_imex_ode(
    (yn, t) -> 0.5 * lambda * yn,  # Explcit evaluation
    (x, h, t) -> (LinearAlgebra.I - 0.5 * h * lambda) \ x,  # Implicit solver
    x -> 0.5 * lambda * x  # Implicit evaluate
)

const explicit_integrators = (
    # Single-step, multi-stage
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

@testset "Convergence Rates Single-Step Explicit Integrators" verbose = true begin
    foreach(explicit_integrators) do scheme
        errors = zeros(8)
        dts = zeros(length(errors))
        dt = 0.2
        for i in eachindex(errors)
            y_n = TimeIntegrators.initializeScheme([y_0], scheme)
            dt = dt / 2
            dts[i] = dt
            for t in 0.0:dt:t_final-dt
                TimeIntegrators.timeIntegrate!(y_n, test_ode_explicit, t, dt)
            end

            errors[i] = abs(exact_sol(t_final) - TimeIntegrators.get_solution(y_n)[1])
        end
        rates = [log(errors[i]/errors[i+1])/(log(dts[i]/dts[i+1])) for i in eachindex(errors)[1:end-1]]

        # The rate is computed to 2 decimal places.
        @test isapprox(rates[end], TimeIntegrators.get_order(scheme), rtol=1e-2)
    end
end

# const explicit_multi_step_integrators = (
#     # Multi-step, single-stage
#     (TimeIntegrators.AB1, TimeIntegrators.FORWARD_EULER),
#     (TimeIntegrators.AB2, TimeIntegrators.HEUN2),
#     (TimeIntegrators.AB3, TimeIntegrators.HEUN3),
#     (TimeIntegrators.AB4, TimeIntegrators.RK4),
#     (TimeIntegrators.ARK4, TimeIntegrators.RK4),
#     #(TimeIntegrators.AB5, TimeIntegrators.FORWARD_EULER),
# )
# @testset "Convergence Rates Multi-Step Explicit Integrators" verbose = true begin
#     foreach(explicit_multi_step_integrators) do (scheme, startup_scheme)
#         errors = zeros(8)
#         dts = zeros(length(errors))
#         dt = 0.2
#         for i in eachindex(errors)
#             y_n = TimeIntegrators.initializeScheme([y_0], scheme, startup_scheme)
#             dt = dt / 2
#             dts[i] = dt
#             for t in 0.0:dt:t_final-dt
#                 TimeIntegrators.timeIntegrate!(y_n, test_ode_explicit, t, dt)
#             end

#             errors[i] = abs(exact_sol(t_final) - TimeIntegrators.get_solution(y_n)[1])
#         end
#         rates = [log(errors[i]/errors[i+1])/(log(dts[i]/dts[i+1])) for i in eachindex(errors)[1:end-1]]

#         # The rate is computed to 2 decimal places.
#         @test isapprox(rates[end], TimeIntegrators.get_order(scheme), rtol=1e-2)
#     end
# end

# Fully implicit ---------------------------------------------------------------------------
const implicit_integrators = (
    # Single-step, multi-stage
    TimeIntegrators.BACKWARD_EULER,
    TimeIntegrators.RADAU_IA_1,
    TimeIntegrators.IMPLICIT_MIDPOINT,
    TimeIntegrators.DIRK2,
    TimeIntegrators.RADAU_IA_3,
    TimeIntegrators.DIRK3,
    TimeIntegrators.DIRK4,
    TimeIntegrators.GAUSS_LEGENDRE_4,
    TimeIntegrators.GAUSS_LEGENDRE_6,
    # # Multi-step, single-stage
    # TimeIntegrators.AM1,
    # TimeIntegrators.AM2,
    # TimeIntegrators.AM3,
    # TimeIntegrators.AM4,
    # TimeIntegrators.BD1,
    # TimeIntegrators.BD2,
    # TimeIntegrators.BD3,
    # TimeIntegrators.BD4,
)

@testset "Convergence Rates Implicit Integrators" verbose = true begin
    foreach(implicit_integrators) do scheme
        errors = zeros(8)
        dts = zeros(length(errors))
        dt = 0.2
        #@show scheme
        for i in eachindex(errors)
            y_n = TimeIntegrators.initializeScheme([y_0], scheme)
            dt = dt / 2
            dts[i] = dt
            for t in 0.0:dt:t_final-dt
                TimeIntegrators.timeIntegrate!(y_n, test_ode_implicit, t, dt)
            end

            errors[i] = abs(exact_sol(t_final) - TimeIntegrators.get_solution(y_n)[1])
            #@show TimeIntegrators.get_solution(y_n)
        end
        #@show errors
        rates = [log(errors[i]/errors[i+1])/(log(dts[i]/dts[i+1])) for i in eachindex(errors)[1:end-1]]
        #@show rates

        # The rate is computed to 2 decimal places.
        if TimeIntegrators.get_order(scheme) > 4
            # For high-order methods, we reach machine precision so the rate bottoms out.
            # We can pick an earlier rate to check correctness.
            test_rate = rates[3]
        else
            test_rate = rates[end]
        end

        @test isapprox(test_rate, TimeIntegrators.get_order(scheme), rtol=1e-2)
    end
end

# IMEX -------------------------------------------------------------------------------------
const imex_integrators = (
    # Single-step, multi-stage
    TimeIntegrators.BACKWARD_FORWARD_EULER,
    TimeIntegrators.MIDPOINT_IMEX,
    # TimeIntegrators.RK3_IMEX,
    # # Multi-step, single-stage
    # TimeIntegrators.CNAB2,
    # TimeIntegrators.SSSS3,
)

@testset "Convergence Rates IMEX Integrators" verbose = true begin
    foreach(imex_integrators) do scheme
        errors = zeros(8)
        dts = zeros(length(errors))
        dt = 0.2
        for i in eachindex(errors)
            y_n = TimeIntegrators.initializeScheme([y_0], scheme)
            dt = dt / 2
            dts[i] = dt
            for t in 0.0:dt:t_final-dt
                TimeIntegrators.timeIntegrate!(y_n, test_ode_imex, t, dt)
            end

            errors[i] = abs(exact_sol(t_final) - TimeIntegrators.get_solution(y_n)[1])
            @show TimeIntegrators.get_solution(y_n)
        end
        #@show errors
        rates = [log(errors[i]/errors[i+1])/(log(dts[i]/dts[i+1])) for i in eachindex(errors)[1:end-1]]
        #@show rates

        # The rate is computed to 2 decimal places.
        @test isapprox(rates[end], TimeIntegrators.get_order(scheme), rtol=1e-2)
    end
end


end
