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
    return TimeIntegrationSolution(y0, scheme, nothing, -1)
end

"""
    initializeScheme(
        step_values::Matrix{Float64},
        step_derivative_explicit::Matrix{Float64},
        step_derivative_implicit::Matrix{Float64},
        scheme::AbstractTimeIntegrator{num_stages, num_steps},
        startup_scheme::AbstractTimeIntegrator{s2, 1},
    ) where {num_stages, num_steps, s2}

Creates the TimeIntegrationSolution object with a (partialy) initialized y_0 matrix. Also
has to know the startup scheme to initialize the (remaining) time levels, and the number of
remaining startup steps to fully intitialize the solution vector.

# Arguments
- `step_values::Matrix{Float64}`: The initial value of the solution matrix.
- `step_derivative_explicit::Matrix{Float64}`: The initial value of the explicit step
    derivative matrix.
- `step_derivative_implicit::Matrix{Float64}`: The initial value of the implicit step
    derivative matrix.
- `scheme::AbstractTimeIntegrator{num_stages, num_steps}`: The time integration scheme.
- `startup_scheme::AbstractTimeIntegrator{s2,1}`: The startup scheme.

# Returns
- `TimeIntegrationSolution{num_steps}`: The initialized solution vector.
"""
function initializeScheme(
    step_values::Matrix{Float64},
    step_derivative_explicit::Matrix{Float64},
    step_derivative_implicit::Matrix{Float64},
    scheme::AbstractTimeIntegrator{num_stages, num_steps},
    startup_scheme::AbstractTimeIntegrator{s2, 1},
) where {num_stages, num_steps, s2}
    # size of y_0 should be equal to (N, the number of external points)
    # where N is the size of the ODE system
    N = size(step_values, 1)
    if size(step_values)[2] > num_steps ||
        size(step_derivative_explicit)[2] > num_steps ||
        size(step_derivative_implicit)[2] > num_steps
        throw(
            ArgumentError(
                LazyString(
                    "The size of the input vector is not compatible with the scheme",
                    ", the actual sizes are: ",
                    size(step_values),
                    size(step_derivative_explicit),
                    size(step_derivative_implicit),
                    " , but the expected size is:  ( $N, $num_steps )",
                )
            ),
        )
    end

    # extend the step and derivative values to the size of the scheme
    step_values_extended = hcat(step_values, zeros(N, num_steps - size(step_values, 2)))
    step_derivative_explicit_extended = hcat(
        step_derivative_explicit, zeros(N, num_steps - size(step_derivative_explicit, 2))
    )
    step_derivative_implicit_extended = hcat(
        step_derivative_implicit, zeros(N, num_steps - size(step_derivative_implicit, 2))
    )
    y0 = hcat(
        step_values_extended,
        step_derivative_explicit_extended,
        step_derivative_implicit_extended,
    )

    return TimeIntegrationSolution(y0, scheme, startup_scheme, n_startup_steps)
end

"""
    initializeScheme(
        y0::Matrix{Float64},
        scheme::AbstractTimeIntegrator{num_stages, num_steps},
        startup_scheme::AbstractTimeIntegrator{s2, 1},
        n_startup_steps::Int,
    ) where {num_stages, num_steps, s2}

Creates the TimeIntegrationSolution object with a (partialy) initialized y_0 matrix. Also
has to know the startup scheme to initialize the (remaining) time levels, and the number of
remaining startup steps to fully intitialize the solution vector.

# Arguments
- `y0::Matrix{Float64}`: The initial value of the solution matrix.
- `scheme::AbstractTimeIntegrator{num_stages, num_steps}`: The time integration scheme.
- `startup_scheme::AbstractTimeIntegrator{s2,1}`: The startup scheme.
- `n_startup_steps::Int`: The number of remaining startup steps.

# Returns
- `TimeIntegrationSolution{num_steps}`: The initialized solution vector.
"""
function initializeScheme(
    y0::Matrix{Float64},
    scheme::AbstractTimeIntegrator{num_stages, num_steps},
    startup_scheme::AbstractTimeIntegrator{s2, 1},
    n_startup_steps::Int,
) where {num_stages, num_steps, s2}
    # size of y0 should be equal to (N, the number of external points)
    # where N is the size of the ODE system
    N = size(y0, 1)
    if size(y0)[2] != num_steps
        throw(
            ArgumentError(
                LazyString(
                    "The size of the input vector is not compatible with the scheme",
                    ", the actual size is: ",
                    size(y0),
                    ", expected:  (",
                    N,
                    ", ",
                    num_steps,
                   ")",
                )
            ),
        )
    end

    return TimeIntegrationSolution(y0, scheme, startup_scheme, n_startup_steps)
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
    y0::Vector{Float64},
    scheme::AbstractTimeIntegrator{num_stages_scheme, num_steps},
    startup_scheme::AbstractTimeIntegrator{num_stages_startup, 1},
) where {num_stages_scheme, num_steps, num_stages_startup}
    if num_steps == 1
        throw(
            ArgumentError(
                "The scheme is a single-step scheme, so no startup_scheme is needed.",
            ),
        )
    end

    # Set the solution vector which has time level 0 to the initial value.
    N = length(y0)
    yn = zeros(N, num_steps)
    yn[:, 1] .= y0

    # Check if the scheme has step derivatives, and set the number of startup steps
    # accordingly.
    if !isempty(scheme.time_levels.step_derivatives_implicit) || !isempty(scheme.time_levels.step_derivatives_explicit)
        n_startup_steps = maximum(scheme.time_levels)
    else
        n_startup_steps = -1
    end

    return initializeScheme(yn, scheme, startup_scheme, n_startup_steps)
end
