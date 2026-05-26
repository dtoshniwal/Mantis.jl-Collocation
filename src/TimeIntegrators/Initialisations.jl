"""
    initializeScheme(
        y0::Matrix{Float64}, scheme::AbstractTimeIntegrator{num_stages, num_steps}
    ) where {num_stages, num_steps}

Creates the TimeIntegrationSolution object with a initialized y_0 vector.

# Arguments
- `y0::Matrix{Float64}`: The initial value of the solution matrix.
- `scheme::AbstractTimeIntegrator{num_stages, num_steps}`: The time integration scheme.

# Returns
- `TimeIntegrationSolution{num_steps}`: The initialized solution vector.
"""
function initializeScheme(
    y0::Matrix{T}, scheme::AbstractTimeIntegrator{num_stages, num_steps}
) where {T, num_stages, num_steps}
    return TimeIntegrationSolution(y0, scheme, nothing, 0)
end

"""
    initializeScheme(
        y0::Vector{T}, scheme::AbstractTimeIntegrator{num_stages, num_steps}
    ) where {T, num_stages, num_steps}

For a single-step scheme (multi-stage).

# Arguments
- `y0::Vector{T}`: The initial value of the solution vector.
- `scheme::AbstractTimeIntegrator{num_stages, num_steps}`: The time integration scheme.
"""
function initializeScheme(
    y0::Vector{T}, scheme::AbstractTimeIntegrator{num_stages, num_steps}
) where {T, num_stages, num_steps}
    if maximum(scheme.time_levels) != 0
        throw(ArgumentError("The scheme is not a single-step scheme"))
    end

    # Set the solution vector which has time level 0 to the initial value.
    yn = zeros(T, length(y0), num_steps)
    yn[:, 1] .= y0

    return initializeScheme(yn, scheme)
end

"""
    initializeScheme(
        y0::Vector{Float64},
        scheme::AbstractTimeIntegrator{s1, num_steps},
        startup_scheme::AbstractTimeIntegrator{s2, 1},
    ) where {s1, num_steps, s2}

For a multi-step scheme, where also stage derivatives have to be initialized.

# Arguments
- `y0::Vector{Float64}`: The initial value of the solution vector.
- `scheme::AbstractTimeIntegrator{s1,num_steps}`: The time integration scheme.
- `startup_scheme::AbstractTimeIntegrator{s2,1}`: The startup scheme.

# Returns
- `TimeIntegrationSolution{num_steps}`: The initialized solution vector.
"""
function initializeScheme(
    y0::Vector{T},
    scheme::AbstractTimeIntegrator{num_stages_scheme, num_steps},
    startup_scheme::AbstractTimeIntegrator{num_stages_startup, 1},
) where {T, num_stages_scheme, num_steps, num_stages_startup}
    if num_steps == 1
        throw(
            ArgumentError(
                "The scheme is a single-step scheme, so no startup_scheme is needed.",
            ),
        )
    end

    total_length = length(scheme.time_levels.step_values) +
        length(scheme.time_levels.step_derivatives_explicit) +
        length(scheme.time_levels.step_derivatives_implicit)

    # Set the solution vector which has time level 0 to the initial value.
    yn = zeros(T, length(y0), total_length)
    yn[:, length(scheme.time_levels.step_values)] .= y0

    n_startup_steps = max(
        length(scheme.time_levels.step_values) - 1,
        length(scheme.time_levels.step_derivatives_explicit),
        length(scheme.time_levels.step_derivatives_implicit)
    )

    sol_startup = initializeScheme(y0, startup_scheme)

    return TimeIntegrationSolution(yn, scheme, startup_scheme, n_startup_steps, sol_startup)
end
