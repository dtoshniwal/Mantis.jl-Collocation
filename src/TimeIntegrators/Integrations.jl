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
    y_n::TimeIntegrationSolution,
    ode::TimeIntegrationOperators,
    t::Float64,
    dt::Float64;
    kwargs...,
)
    if !(isnothing(get_startup_scheme(y_n)))
        # Calculate the step-derivatives for the current step for the multi-step scheme.
        num_startup_steps = maximum(y_n.scheme.time_levels)
        if num_startup_steps >= 0 && get_remaining_startup_steps(y_n) == num_startup_steps
            calc_step_derivatives!(y_n, ode, dt, t; kwargs...)
        end

        # Check if we still need to use the startup scheme.
        if get_remaining_startup_steps(y_n) >= 0
            if y_n.scheme.time_levels == y_n.startup_scheme.time_levels
                throw(ArgumentError("The scheme is not a startup scheme"))
            end
            startup_scheme = get_startup_scheme(y_n)
            y_old = y_n
            y_n = initializeScheme(get_solution(y_old), y_old.startup_scheme)
            shift_steps!(y_old)
            timeIntegrate_!(y_n, startup_scheme, ode, t, dt; kwargs...)
            sol = get_solution(y_n)
            y_n = y_old
            y_n.solution[:, 1] = sol
            calc_step_derivatives!(y_n, ode, dt, t + dt; kwargs...)
        end
    else
        timeIntegrate_!(y_n, get_scheme(y_n), ode, t, dt; kwargs...)
    end
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
    Yi = Vector{Float64}(undef, N)
    @views yⁿ = y_n.solution_alocated[:, :]

    @inbounds for i in 1:num_stages
        # Calculate the stage value Yi
        Yi .= F * scheme.A[i, :] * dt
        Yi .+= y_n.solution * scheme.U[i, :]

        # Calculate the stage derivative Fᵢ
        F[:, i] .= ode.explicitEvaluate(Yi, t + scheme.C[i] * dt; kwargs...)
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
    return y_n.solution, y_n.solution_alocated = y_n.solution_alocated, y_n.solution
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
    Yᵢ = Vector{Float64}(undef, N)
    xᵢ = Vector{Float64}(undef, N)
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
    Yᵢ = Vector{Float64}(undef, N)
    xᵢ = Vector{Float64}(undef, N)
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

    return y_n.solution, y_n.solution_alocated = y_n.solution_alocated, y_n.solution
end

"""
    shift_steps!(y_n::TimeIntegrationSolution)

Shift the time levels of the solution vector assuming y_n is at index 1 and y_{n-num_steps} is at index num_steps.
Example:
- time levels:  [0 1 2 | 0 1 | 0 1]
- input ->      [y₀ 0 0 | 0 0 | 0 0]
- t1 init ->    [y₀ 0 0 | G₀ 0 | F₀ 0]
- t1 shift ->   [0 y₀ 0 | 0 G₀ | 0 F₀]
- t1 integrate->[y₁ y₀ 0 | 0 G₀ | 0 F₀]
- t1 step derv->[y₁ y₀ 0 | G₁ G₀ | F₁ F₀]
- t2 shift ->   [0 y₁ y₀ | 0 G₁ | G₁ F₁]
- t2 integrate->[y₂ y₁ y₀ | 0 G₁ | G₁ F₁]
- t2 step derv->[y₂ y₁ y₀ | G₂ G₁ | F₂ F₁]

# Arguments
- `y_n::TimeIntegrationSolution`: The current solution vector.

# Returns (in-place)
- `TimeIntegrationSolution`: The updated solution vector after shifting the time levels.
"""
function shift_steps!(y_n::TimeIntegrationSolution)
    return y_n.solution = circshift(y_n.solution, (0, 1))
end

"""
    calc_step_derivatives!(y_n::TimeIntegrationSolution, ode::TimeIntegrationOperators, dt::Float64, t::Float64; kwargs...)

Calculate the stage derivatives for the current step for the multi-step scheme.

# Arguments
- `y_n::TimeIntegrationSolution`: The current solution vector.
- `ode::TimeIntegrationOperators`: The ODE system operators.
- `dt::Float64`: The time step.
- `t::Float64`: The current time.
- `kwargs...`: Additional arguments passed to the user defined ODE system.

# Returns (in-place)
- `TimeIntegrationSolution`: The updated solution vector after calculating the stage derivatives.
"""
function calc_step_derivatives!(
    y_n::TimeIntegrationSolution,
    ode::TimeIntegrationOperators,
    dt::Float64,
    t::Float64;
    kwargs...,
)
    curr_step_values = y_n.solution[:, 1]
    N_step_values = size(y_n.scheme.time_levels.step_values, 1)
    N_step_derivatives_implicit = size(y_n.scheme.time_levels.step_derivatives_implicit, 1)

    # Calculate stage derivatives implicit (G)
    if !isempty(y_n.scheme.time_levels.step_derivatives_implicit)
        start_index = N_step_values + 1
        y_n.solution[:, start_index] = (
            ode.implicitSolve(curr_step_values, dt, t; kwargs...) - curr_step_values
        )
    end

    # Calculate stage derivatives explicit (F)
    if !isempty(y_n.scheme.time_levels.step_derivatives_explicit)
        start_index = N_step_values + N_step_derivatives_implicit + 1
        y_n.solution[:, start_index] =
            ode.explicitEvaluate(curr_step_values, t; kwargs...) * dt
    end

    return y_n.remaining_startup_steps -= 1
end
