"""
    timeIntegrate(
        y_n::TimeIntegrationSolution,
        ode::TimeIntegrationOperators,
        t::Float64,
        dt::Float64;
        kwargs...,
    )

Perform a single time integration step using the given time integration scheme and ODE
system operators.

# See also
[`timeIntegrate!`](@ref)

# Arguments
- `y_n::TimeIntegrationSolution`: The current solution vector.
- `ode::TimeIntegrationOperators`: The ODE system operators.
- `t::Float64`: The current time.
- `dt::Float64`: The time step.
- `kwargs...`: Additional arguments passed to the ODE system.

# Returns
- `TimeIntegrationSolution`: The updated solution vector after one time step.
"""
function timeIntegrate(
    y_n::TimeIntegrationSolution,
    ode::TimeIntegrationOperators,
    t::Float64,
    dt::Float64;
    kwargs...,
)
    y_n = deepcopy(y_n)
    timeIntegrate!(y_n, ode, t, dt; kwargs...)
    return y_n
end

"""
    timeIntegrate(
        y1_n::T1,
        y2_n::T2,
        ode::TimeIntegrationOperators,
        t::Float64,
        dt::Float64;
        ϵ::Float64=1e-3,
        tol::Float64=2e-3,
        max_step_change::Float64=0.5,
        error_evaluation::Function=norm,
        kwargs...,
    ) where {T1 <: TimeIntegrationSolution, T2 <: TimeIntegrationSolution}

Perform a single time integration step by calculating two solution with different schemes
to adaptive determine the next time step size.

# Arguments
- `y1_n::T1`: Solution vector with higher order scheme.
- `y2_n::T2`: Solution vector with lower order scheme
- `ode::TimeIntegrationOperators`: ODE system
- `t::Float64`: Current time
- `dt::Float64`: Current time step

# Keyword Arguments
- `ϵ::Float64=1e-3`: Desired error tolerance
- `tol::Float64=2e-3`: Error tolerance for the adaptive time step
- `max_step_change::Float64=0.5`: Maximum allowed change of the time step
- `error_evaluation::Function=norm`: Error evaluation function
- `kwargs...`: Additional arguments which are passed to the ODE system

# Returns
- `y_lower::TimeIntegrationSolution`: Solution using the higher order scheme.
- `y_higher::TimeIntegrationSolution`: Solution using the lower order scheme.
- `dt_new::Float64`: New time step.
- `::Float64`: Updated time.
"""
function timeIntegrate(
    y1_n::T1,
    y2_n::T2,
    ode::TimeIntegrationOperators,
    t::Float64,
    dt::Float64;
    ϵ::Float64=1e-3,
    tol::Float64=2e-3,
    max_step_change::Float64=0.5,
    error_evaluation::Function=norm,
    kwargs...,
) where {T1 <: TimeIntegrationSolution, T2 <: TimeIntegrationSolution}
    # calculate the solution with the higher order scheme
    y_higher = timeIntegrate(y1_n, ode, t, dt; kwargs...)
    # calculate the solution with the lower order scheme
    y_lower = timeIntegrate(y2_n, ode, t, dt; kwargs...)
    # calculate the error
    error = error_evaluation(get_solution(y_higher) - get_solution(y_lower))
    y_lower.solution[:, 1] = get_solution(y_higher)

    # from the PhD thesis "Control of Error and Convergence in ODE Solvers" by Kjell
    # Gustafsson (1992)
    dt_predicted(dt, ϵ) = dt * clamp(
        (ϵ / error)^(1 / y1_n.scheme.order), max_step_change, 1 / max_step_change
    )

    dt_new = dt_predicted(dt, ϵ)

    if error > tol
        # If the error is too high, reject the solution and try again with a smaller time
        # step
        return timeIntegrate(
            y1_n,
            y2_n,
            ode,
            t,
            dt_new;
            ϵ=ϵ,
            tol=tol,
            max_step_change=max_step_change,
            error_evaluation=error_evaluation,
            kwargs...,
        )
    end

    return y_lower, y_higher, dt_new, t + dt
end

"""
    timeIntegrate!(
        y_n::TimeIntegrationSolution,
        ode::TimeIntegrationOperators,
        t::Float64,
        dt::Float64;
        kwargs...,
    )

Perform a single, in-place time integration step using the given time integration scheme
and ODE system operators.

# See also
[`timeIntegrate`](@ref)

# Arguments
- `y_n::TimeIntegrationSolution`: The current solution vector.
- `ode::TimeIntegrationOperators`: The ODE system operators.
- `t::Float64`: The current time.
- `dt::Float64`: The time step.
- `kwargs...`: Additional arguments passed to the ODE system.

# Returns (in-place)
- `TimeIntegrationSolution`: The updated solution vector after one time step.
"""
function timeIntegrate!(
    y_n::TimeIntegrationSolution{T, S},
    ode::TimeIntegrationOperators,
    t::Float64,
    dt::Float64;
    kwargs...,
) where {T, S <: AbstractTimeIntegrator}
    remaining_startup_steps = get_remaining_startup_steps(y_n)
    if remaining_startup_steps > 0
        num_steps = length(y_n.scheme.time_levels.step_values)
        num_G = length(y_n.scheme.time_levels.step_derivatives_implicit)
        num_F = length(y_n.scheme.time_levels.step_derivatives_explicit)

        y_n_startup = y_n.startup_solution

        # We do have to advance the solution using the startup scheme, to ensure that the
        # solution remains at the expected time level, even if some of the startup steps
        # are only needed to initialise the step derivatives.
        ynm1 = get_solution(y_n_startup)
        timeIntegrate_!(y_n_startup, y_n.startup_scheme, ode, t, dt; kwargs...)

        # Then we initialise the required solutions and stage derivatives. Note these are
        # all if statements, as we may need all of them at the same time. The first
        # condition in the if statements always check if we ever need the specific type of
        # initialisation, while the second condition ensures that we are at the correct
        # time to initialise the solution or stage derivatives.
        if num_steps > 0 && remaining_startup_steps - num_steps <= 0
            index_steps = num_steps + (remaining_startup_steps - num_steps)
            y_n.solution[:, index_steps] = get_solution(y_n_startup)
        end
        if num_G > 0 && remaining_startup_steps - num_G <= 0
            index_implicit = num_steps + num_G + (remaining_startup_steps - num_G)
            y_n.solution[:, index_implicit] .= ode.implicitEvaluate(get_solution(y_n_startup); kwargs...) .* dt
        end
        if num_F > 0 && remaining_startup_steps - num_F <= 0
            index_explicit = num_steps + num_G + num_F + (remaining_startup_steps - num_F)
            y_n.solution[:, index_explicit] .= ode.explicitEvaluate(ynm1, t; kwargs...) .* dt
        end

        t += dt
        y_n.remaining_startup_steps = y_n.remaining_startup_steps - 1

        return nothing
    else
        timeIntegrate_!(y_n, get_scheme(y_n), ode, t, dt; kwargs...)

        return nothing
    end
end

function timeIntegrate!(
    y_n::TimeIntegrationSolution{T, Nothing}, # No startup scheme
    ode::TimeIntegrationOperators,
    t::Float64,
    dt::Float64;
    kwargs...,
) where {T}
    timeIntegrate_!(y_n, get_scheme(y_n), ode, t, dt; kwargs...)
end

"""
    timeIntegrate_!(
        y_n::TimeIntegrationSolution,
        scheme::Explicit{num_stages, num_steps},
        ode::TimeIntegrationOperators,
        t::Float64,
        dt::Float64;
        kwargs...,
    ) where {num_steps, num_stages}

Perform a single time integration step using the given Explicit time integration scheme and
ODE system operators.

# Arguments
- `y_n::TimeIntegrationSolution`: The current solution vector.
- `scheme::Explicit{num_stages, num_steps}`: The Explicit time integration scheme.
- `ode::TimeIntegrationOperators`: The ODE system operators.
- `t::Float64`: The current time.
- `dt::Float64`: The time step.
- `kwargs...`: Additional arguments passed to the ODE system.

# Returns (in-place)
- `TimeIntegrationSolution{num_steps}`: The updated solution vector after one time step.
"""
function timeIntegrate_!(
    y_n::TimeIntegrationSolution,
    scheme::Explicit{num_stages, num_steps},
    ode::TimeIntegrationOperators,
    t::Float64,
    dt::Float64;
    kwargs...,
) where {num_steps, num_stages}
    N = get_num_variables(y_n)
    y_nm1 = get_solution(y_n)
    F = get_F_allocated(y_n)
    F .= 0.0
    Yi = Vector{eltype(y_n)}(undef, N)
    @views yⁿ = y_n.solution_alocated[:, :]
    # Ti = 0.0
    @inbounds for i in 1:num_stages
        # Calculate the stage value Yi
        Yi .= F * scheme.A[i, :] * dt
        @inbounds for j in 1:num_steps
            Yi .+= scheme.U[i, j] * y_nm1[:,j]
        end
        # for j in 1:num_steps
        #     Ti += scheme.A[i, j] * dt
        #     Ti += scheme.U[i, j] * t_nm1[j]
        # end

        # Calculate the stage derivative Fᵢ
        F[:, i] .= ode.explicitEvaluate(Yi, t + scheme.C[i] * dt; kwargs...)
        # F[:, i] .= ode.explicitEvaluate(Yi, Ti; kwargs...)
    end

    # Optimisation when both the last stages are the same as the first step, so we can skip
    # the calculation of the final solution
    start_index = 1
    if scheme.U[end, :] == scheme.V[1, :] && scheme.A[end, :] == scheme.B[1, :]
        yⁿ[:, 1] .= Yi
        start_index = 2
    end

    @inbounds for i in start_index:num_steps
        @views yⁿ[:, i] .= F * view(scheme.B, i, :) * dt
        @views yⁿ[:, i] .+= y_nm1 * scheme.V[i, :]
    end

    # swich the pointers of y_n.solution and y_n.solution_alocated
    y_n.solution, y_n.solution_alocated = y_n.solution_alocated, y_n.solution

    return nothing
end

"""
    timeIntegrate_!(
        y_n::TimeIntegrationSolution,
        scheme::DiagonallyImplicit{num_stages, num_steps},
        ode::TimeIntegrationOperators,
        t::Float64,
        dt::Float64;
        kwargs...,
    ) where {num_steps, num_stages}

Perform a single time integration step using the given DiagonallyImplicit time integration scheme and
ODE system operators.

# Arguments
- `y_n::TimeIntegrationSolution{num_steps}`: The current solution vector.
- `scheme::DiagonallyImplicit{num_stages,num_steps}`: The DiagonallyImplicit time integration scheme.
- `ode::TimeIntegrationOperators`: The ODE system operators.
- `t::Float64`: The current time.
- `dt::Float64`: The time step.
- `kwargs...`: Additional arguments passed to the ODE system.

# Returns (in-place)
- `TimeIntegrationSolution{num_steps}`: The updated solution vector after one time step.
"""
function timeIntegrate_!(
    y_n::TimeIntegrationSolution,
    scheme::DiagonallyImplicit{num_stages, num_steps},
    ode::TimeIntegrationOperators,
    t::Float64,
    dt::Float64;
    kwargs...,
) where {num_steps, num_stages}
    N = get_num_variables(y_n)
    y_nm1 = get_solution(y_n)
    G = get_G_allocated(y_n)
    G .= 0.0
    Yᵢ = Vector{eltype(y_n)}(undef, N)
    xᵢ = Vector{eltype(y_n)}(undef, N)
    @views yⁿ = y_n.solution_alocated[:, :]

    @inbounds for i in 1:num_stages
        # calculate the temporary value xᵢ
        xᵢ .= dt * G[:, 1:i] * scheme.A[i, 1:i]
        xᵢ .+= y_nm1 * scheme.U[i, :]

        # Calculate the stage value Yᵢ
        # solve (Yᵢ - aᵢᵢᴵᴹ * h * g(Yᵢ,t)) = xᵢ
        Yᵢ = ode.implicitSolve(xᵢ, scheme.A[i, i] * dt, t + scheme.C[i] * dt; kwargs...)
        # Calculate the stage derivative Gᵢ
        let a_ii = scheme.A[i, i]
            if a_ii != 0.0
                G[:, i] .= (Yᵢ - xᵢ) / (a_ii * dt)
            else
                G[:, i] .= 0.0
            end
        end
    end

    # Optimisation when both the last stages are the same as the first step, so we can skip
    # the calculation of the final solution
    start_index = 1
    if scheme.U[end, :] == scheme.V[1, :] && scheme.A[end, :] == scheme.B[1, :]
        yⁿ[:, 1] .= Yᵢ
        start_index = 2
    end

    @inbounds for i in start_index:num_steps
        @views yⁿ[:, i] .= dt * G * scheme.B[i, :]
        @views yⁿ[:, i] .+= y_nm1 * scheme.V[i, :]
    end

    y_n.solution, y_n.solution_alocated = y_n.solution_alocated, y_n.solution

    return nothing
end

"""
    timeIntegrate_!(
        y_n::TimeIntegrationSolution,
        scheme::Implicit{num_stages, num_steps},
        ode::TimeIntegrationOperators,
        t::Float64,
        dt::Float64;
        kwargs...,
    ) where {num_steps, num_stages}

Perform a single time integration step using the given Implicit time integration scheme and
ODE system operators.

# Arguments
- `y_n::TimeIntegrationSolution{num_steps}`: The current solution vector.
- `scheme::Implicit{num_stages,num_steps}`: The Implicit time integration scheme.
- `ode::TimeIntegrationOperators`: The ODE system operators.
- `t::Float64`: The current time.
- `dt::Float64`: The time step.
- `kwargs...`: Additional arguments passed to the ODE system.

# Returns (in-place)
- `TimeIntegrationSolution{num_steps}`: The updated solution vector after one time step.
"""
function timeIntegrate_!(
    y_n::TimeIntegrationSolution,
    scheme::Implicit{num_stages, num_steps},
    ode::TimeIntegrationOperators,
    t::Float64,
    dt::Float64;
    kwargs...,
) where {num_steps, num_stages}
    N = get_num_variables(y_n)
    y_nm1 = get_solution(y_n)
    G = get_G_allocated(y_n)
    G .= 0.0
    Yᵢ = Vector{eltype(y_n)}(undef, N)
    xᵢ = Vector{eltype(y_n)}(undef, N)
    @views yⁿ = y_n.solution_alocated[:, :]
    xi = reduce(vcat, y_nm1 for i in 1:num_stages)

    # newAblocks = ntuple(N) do
    #     SparseArrays.sparse(scheme.A * dt)
    # end
    # newA = SparseArrays.blockdiag([SparseArrays.sparse(scheme.A * dt) for i in 1:N]...)

    # sparse_block = SparseArrays.sparse(Matrix(scheme.A * dt))
    # newA = SparseArrays.blockdiag((sparse_block for i in 1:N)...)
    # Y = ode.implicitSolve(xi, newA, t + scheme.C[1] * dt; kwargs...)
    Y = ode.implicitSolve(xi, scheme.A * dt, t + scheme.C[1] * dt; kwargs...)

    @inbounds for i in 1:num_steps
        allG = ode.implicitEvaluate(Y; kwargs...)
        for n in 1:N
            @views yⁿ[n, i] = dt * dot(allG[(n-1)*num_stages+1:end], scheme.B[i, :])
            @views yⁿ[n, i] += sum(y_nm1[n] .* scheme.V[i, :])
        end
    end

    y_n.solution, y_n.solution_alocated = y_n.solution_alocated, y_n.solution

    return nothing
end

"""
    timeIntegrate_!(
        y_n::TimeIntegrationSolution,
        scheme::IMEX{num_stages, num_steps},
        ode::TimeIntegrationOperators,
        t::Float64,
        dt::Float64;
        kwargs...,
    ) where {num_steps, num_stages}

Perform a single time integration step using the given IMEX time integration scheme and ODE
system operators.

# Arguments
- `y_n::TimeIntegrationSolution{num_steps}`: The current solution vector.
- `scheme::IMEX{num_stages,num_steps}`: The IMEX time integration scheme.
- `ode::TimeIntegrationOperators`: The ODE system operators.
- `t::Float64`: The current time.
- `dt::Float64`: The time step.
- `kwargs...`: Additional arguments passed to the ODE system.

# Returns (in-place)
- `TimeIntegrationSolution{num_steps}`: The updated solution vector after one time step.
"""
function timeIntegrate_!(
    y_n::TimeIntegrationSolution,
    scheme::IMEX{num_stages, num_steps},
    ode::TimeIntegrationOperators,
    t::Float64,
    dt::Float64;
    kwargs...,
) where {num_steps, num_stages}
    N = get_num_variables(y_n)
    y_nm1 = get_solution(y_n)
    F = get_F_allocated(y_n)
    G = get_G_allocated(y_n)
    F .= 0.0
    G .= 0.0
    Yᵢ = Vector{eltype(y_n)}(undef, N)
    xᵢ = Vector{eltype(y_n)}(undef, N)
    @views yⁿ = y_n.solution_alocated[:, :]

    @inbounds for i in 1:num_stages
        # calculate the temporary value xᵢ
        xᵢ .= dt * G[:, 1:i] * scheme.A_IM[i, 1:i]
        xᵢ .+= dt * F[:, 1:i] * scheme.A_EX[i, 1:i]
        xᵢ .+= y_nm1 * scheme.U[i, :]

        # Calculate the stage value Yᵢ
        # solve (Yᵢ - aᵢᵢᴵᴹ * h * g(Yᵢ,t)) = xᵢ
        Yᵢ .= ode.implicitSolve(
            xᵢ, scheme.A_IM[i, i] * dt, t + scheme.C_IM[i] * dt; kwargs...
        )
        # Calculate the stage derivative Fᵢ
        F[:, i] .= ode.explicitEvaluate(Yᵢ, t + scheme.C_EX[i] * dt; kwargs...)
        # Calculate the stage derivative Gᵢ
        let a_ii = scheme.A_IM[i, i]
            if a_ii != 0.0
                G[:, i] .= (Yᵢ - xᵢ) / (a_ii * dt)
            else
                G[:, i] .= 0.0
            end
        end
    end

    start_index = 1
    if scheme.U[end, :] == scheme.V[1, :] &&
        scheme.A_IM[end, :] == scheme.B_IM[1, :] &&
        scheme.A_EX[end, :] == scheme.B_EX[1, :]
        yⁿ[:, 1] .= Yᵢ
        start_index = 2
    end

    @inbounds for i in start_index:num_steps
        @views yⁿ[:, i] .= dt * G * scheme.B_IM[i, :]
        @views yⁿ[:, i] .+= dt * F * scheme.B_EX[i, :]
        @views yⁿ[:, i] .+= y_nm1 * scheme.V[i, :]
    end

    y_n.solution, y_n.solution_alocated = y_n.solution_alocated, y_n.solution

    return nothing
end
