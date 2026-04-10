abstract type AbstractTimeIntegrator{num_stages, num_steps} end

function get_num_stages(
    ::AbstractTimeIntegrator{num_stages, num_steps}
) where {num_stages, num_steps}
    return num_stages
end
function get_num_steps(
    ::AbstractTimeIntegrator{num_stages, num_steps}
) where {num_stages, num_steps}
    return num_steps
end

"""
    TimeIntegrationOperators{F1<:Function,F2<:Function}

explicitEvaluate::(Vector{Float64} -> Vector{Float64}) \\
Function that solves F = f(Y) which has to be defined if Aᵉˣ ≠ 0 \\
where F and Y are the input arguments and f is the output argument \\

implicitSolve::((Vector{Float64}, Float64) -> Vector{Float64}) \\
Function that solves the equation Y - λ g(Y) = x which has to be defined if Aᴵᴹ ≠ 0  \\
for 𝐘 ∈ ℝᴺ , given as input the vector 𝐗 ∈ ℝᴺ and the scalar λ ∈ ℝᴺ.\\
In case g is a linear operator, a direct solution method can be through the inverse operator (I − λg)⁻¹, where I is the identity function.
"""
struct TimeIntegrationOperators{EF, IF}
    explicitEvaluate::EF
    implicitSolve::IF

    function TimeIntegrationOperators(
        explicit_evaluate::Union{Nothing, Function},
        implicit_solve::Union{Nothing, Function},
    )
        new{typeof(explicit_evaluate), typeof(implicit_solve)}(
            explicit_evaluate, implicit_solve
        )
    end
end

function define_explicit_ode(explicit_evaluate::Function)
    return TimeIntegrationOperators(explicit_evaluate, nothing)
end
function define_implicit_ode(implicit_solve::Function)
    return TimeIntegrationOperators(nothing, implicit_solve)
end
function define_imex_ode(explicit_evaluate::Function, implicit_solve::Function)
    return TimeIntegrationOperators(explicit_evaluate, implicit_solve)
end

"""
    define_implicit_linear(M::Matrix{Float64}, K::Matrix{Float64}, F::Vector{Float64})

Function that defines the implicit solver for a linear operator of the form Mx - λKx = F.

The mass and stiffness matrices `M` and `K` can be either `Matrix{Float64}` or `nothing`,
and the forcing vector `F` can be either `Vector{Float64}` or `nothing`. At least `K` or
`F` should be defined.

# Arguments
- `M`: Mass matrix
- `K`: Stiffness matrix
- `F`: Forcing vector

# Returns
- `::TimeIntegrationOperators` : Function that solves the equation M ẏ = K y + F
"""
function define_implicit_linear(
    M::AbstractMatrix{T}, K::AbstractMatrix{T}, F::AbstractVector{T}
) where {T}
    return define_implicit_ode((x, λ, t) -> (M - λ * K) \ (M * x + λ * F))
end
function define_implicit_linear(M::Nothing, K::Nothing, F::AbstractVector{T}) where {T}
    return define_implicit_ode((x, λ, t) -> x + λ * F)
end
function define_implicit_linear(M::Nothing, K::AbstractMatrix{T}, F::Nothing) where {T}
    return define_implicit_ode((x, λ, t) -> (I - λ * K) \ x)
end
function define_implicit_linear(
    M::Nothing, K::AbstractMatrix{T}, F::AbstractVector{T}
) where {T}
    return define_implicit_ode((x, λ, t) -> (I - λ * K) \ (x + λ * F))
end
function define_implicit_linear(
    M::AbstractMatrix{T}, K::AbstractMatrix{T}, F::Nothing
) where {T}
    return define_implicit_ode((x, λ, t) -> (M - λ * K) \ (M * x))
end
function define_implicit_linear(
    M::AbstractMatrix{T}, K::Nothing, F::AbstractVector{T}
) where {T}
    return define_implicit_ode((x, λ, t) -> x + λ * (M \ F))
end
function define_implicit_linear(M::AbstractMatrix{T}, K::Nothing, F::Nothing) where {T}
    throw(ArgumentError("At least one of the matrices K or F has to be defined"))
end
function define_implicit_linear(M::Nothing, K::Nothing, F::Nothing)
    throw(ArgumentError("At least one of the matrices K or F has to be defined"))
end

"""
    define_implicit_linear(func::Function)

Function that defines the implicit solver for a linear operator of the form Mx - λKx = F
The matrix has to be defined 'nothing' if it is not present. Example func(t) = (nothing, K, F)

# Arguments
- func(t::Float64)::Tuple{Union{Matrix{Float64},Nothing}, Union{Matrix{Float64},Nothing}, Union{Vector{Float64},Nothing}} : Function that defines the matrices M, K and F

# Returns
- implicitSolve::Function : Function that solves the equation M ẏ = K y + F
"""
function define_implicit_linear(func::Function)
    function implicitSolveMKF(x::Vector{Float64}, λ::Float64, t::Float64)::Vector{Float64}
        M, K, F = func(t)
        return (M - λ * K) \ (M * x + λ * F)
    end
    function implicitSolveMK(x::Vector{Float64}, λ::Float64, t::Float64)::Vector{Float64}
        M, K, _ = func(t)
        return (M - λ * K) \ (M * x)
    end
    function implicitSolveKF(x::Vector{Float64}, λ::Float64, t::Float64)::Vector{Float64}
        _, K, F = func(t)
        return (I - λ * K) \ (x + λ * F)
    end
    function implicitSolveMF(x::Vector{Float64}, λ::Float64, t::Float64)::Vector{Float64}
        M, _, F = func(t)
        return x + λ * (M \ F)
    end
    M, K, F = func(0.0)
    if F === nothing && K !== nothing && M !== nothing
        return define_implicit_ode(implicitSolveMK)
    elseif F !== nothing && K === nothing && M !== nothing
        return define_implicit_ode(implicitSolveMF)
    elseif F !== nothing && K !== nothing && M === nothing
        return define_implicit_ode(implicitSolveKF)
    else
        return define_implicit_ode(implicitSolveMKF)
    end
end

"""
    struct TimeLevels

time levels of the scheme which are used to determine the structure of the input/output vector y associated to the scheme

# Fields
- `step_values::Vector{Int}`: step values of the scheme
- `step_derivatives_implicit::Vector{Int}`: step derivatives of the implicit part of the scheme
- `step_derivatives_explicit::Vector{Int}`: step derivatives of the explicit part of the scheme
"""
struct TimeLevels
    step_values::Vector{Int}
    step_derivatives_implicit::Vector{Int}
    step_derivatives_explicit::Vector{Int}
end

"""
    struct IMEX{num_stages, num_steps} <: AbstractTimeIntegrator{num_stages, num_steps}

Implicit-Explicit time integration scheme

# Fields
- `A_IM::SMatrix{num_stages,num_stages,Float64}`: Implicit matrix A of size sxs
- `A_EX::SMatrix{num_stages,num_stages,Float64}`: Explicit matrix A of size sxs
- `B_IM::SMatrix{num_steps,num_stages,Float64}`: Implicit matrix B of size rxs
- `B_EX::SMatrix{num_steps,num_stages,Float64}`: Explicit matrix B of size rxs
- `U::SMatrix{num_stages,num_steps,Float64}`: Matrix U of size sxr
- `V::SMatrix{num_steps,num_steps,Float64}`: Matrix V of size rxr
- `C_IM::SVector{num_stages,Float64}`: time Vector C of size num_stages, indicates at what time the stage is evaluated
- `C_EX::SVector{num_stages,Float64}`: time Vector C of size num_stages, indicates at what time the stage is evaluated
- `time_levels::TimeLevels`: reflects the structure of the input/output vector y associated to the scheme
- `order::Int`: order of the scheme

# Type parameters
- `num_stages`: number of stages of the scheme
- `num_steps`: number of external steps of the scheme
"""
struct IMEX{num_stages, num_steps, NT, AA, AE, EE} <: AbstractTimeIntegrator{num_stages, num_steps}
    A_IM::SMatrix{num_stages, num_stages, NT, AA} # of size sxs
    A_EX::SMatrix{num_stages, num_stages, NT, AA} # of size sxs
    B_IM::SMatrix{num_steps, num_stages, NT, AE}  # of size rxs
    B_EX::SMatrix{num_steps, num_stages, NT, AE}  # of size rxs
    U::SMatrix{num_stages, num_steps, NT, AE} # of size sxr
    V::SMatrix{num_steps, num_steps, NT, EE} # of size rxr
    C_IM::SVector{num_stages, NT}  # of size num_stages, indicates at what time the stage is evaluated
    C_EX::SVector{num_stages, NT}  # of size num_stages, indicates at what time the stage is evaluated
    # reflects the structure of the input/output vector y associated to the scheme
    time_levels::TimeLevels
    order::Int

    function IMEX(
        A_IM::SMatrix{num_stages, num_stages, NT, AA},
        A_EX::SMatrix{num_stages, num_stages, NT, AA},
        B_IM::SMatrix{num_steps, num_stages, NT, AE},
        B_EX::SMatrix{num_steps, num_stages, NT, AE},
        U::SMatrix{num_stages, num_steps, NT, AE},
        V::SMatrix{num_steps, num_steps, NT, EE},
        C_IM::SVector{num_stages, NT},
        C_EX::SVector{num_stages, NT},
        time_levels::TimeLevels,
        order::Int,
    ) where {num_stages, num_steps, NT, AA, AE, EE}
        return new{num_stages, num_steps, NT, AA, AE, EE}(
            A_IM, A_EX, B_IM, B_EX, U, V, C_IM, C_EX, time_levels, order
        )
    end
end

"""
    struct Explicit{num_stages, num_steps} <: AbstractTimeIntegrator{num_stages,num_steps}

Explicit time integration scheme

# Fields
- `A::SMatrix{num_stages,num_stages,Float64}`: matrix A of size sxs
- `B::SMatrix{num_steps,num_stages,Float64}`: matrix B of size rxs
- `U::SMatrix{num_stages,num_steps,Float64}`: matrix U of size sxr
- `V::SMatrix{num_steps,num_steps,Float64}`: matrix V of size rxr
- `C::SVector{num_stages,Float64}`: time Vector C of size num_stages, indicates at what time the stage is evaluated
- `time_levels::TimeLevels`: reflects the structure of the input/output vector y associated to the scheme
- `order::Int`: order of the scheme

# Type parameters
- `num_stages`: number of stages of the scheme
- `num_steps`: number of external steps of the scheme
"""
struct Explicit{num_stages, num_steps, NT, AA, AE, EE} <:
       AbstractTimeIntegrator{num_stages, num_steps}
    A::SMatrix{num_stages, num_stages, NT, AA} # of size sxs
    B::SMatrix{num_steps, num_stages, NT, AE}  # of size rxs
    U::SMatrix{num_stages, num_steps, NT, AE} # of size sxr
    V::SMatrix{num_steps, num_steps, NT, EE} # of size rxr
    C::SVector{num_stages, NT}  # of size num_stages, indicates at what time the stage is evaluated
    time_levels::TimeLevels
    order::Int

    function Explicit(
        A::SMatrix{num_stages, num_stages, NT, AA},
        B::SMatrix{num_steps, num_stages, NT, AE},
        U::SMatrix{num_stages, num_steps, NT, AE},
        V::SMatrix{num_steps, num_steps, NT, EE},
        C::SVector{num_stages, NT},
        time_levels::TimeLevels,
        order::Int,
    ) where {num_stages, num_steps, NT, AA, AE, EE}
        return new{num_stages, num_steps, NT, AA, AE, EE}(A, B, U, V, C, time_levels, order)
    end
end

"""
    struct Implicit{num_stages, num_steps} <: AbstractTimeIntegrator{num_stages, num_steps}

Implicit time integration scheme

# Fields
- `A::SMatrix{num_stages,num_stages,Float64}`: matrix A of size sxs
- `B::SMatrix{num_steps,num_stages,Float64}`: matrix B of size rxs
- `U::SMatrix{num_stages,num_steps,Float64}`: matrix U of size sxr
- `V::SMatrix{num_steps,num_steps,Float64}`: matrix V of size rxr
- `C::SVector{num_stages,Float64}`: time Vector C of size num_stages, indicates at what time the stage is evaluated
- `time_levels::TimeLevels`: reflects the structure of the input/output vector y associated to the scheme
- `order::Int`: order of the scheme

# Type parameters
- `num_stages`: amount of stages of the scheme
- `num_steps`: amount of external steps of the scheme
"""
struct Implicit{num_stages, num_steps, NT, AA, AE, EE} <: AbstractTimeIntegrator{num_stages, num_steps}
    A::SMatrix{num_stages, num_stages, NT, AA} # of size sxs
    B::SMatrix{num_steps, num_stages, NT, AE}  # of size rxs
    U::SMatrix{num_stages, num_steps, NT, AE} # of size sxr
    V::SMatrix{num_steps, num_steps, NT, EE} # of size rxr
    C::SVector{num_stages, NT}  # of size num_stages, indicates at what time the stage is evaluated
    time_levels::TimeLevels
    order::Int

    function Implicit(
        A::SMatrix{num_stages, num_stages, NT, AA},
        B::SMatrix{num_steps, num_stages, NT, AE},
        U::SMatrix{num_stages, num_steps, NT, AE},
        V::SMatrix{num_steps, num_steps, NT, EE},
        C::SVector{num_stages, NT},
        time_levels::TimeLevels,
        order::Int,
    ) where {num_stages, num_steps, NT, AA, AE, EE}
        return new{num_stages, num_steps, NT, AA, AE, EE}(A, B, U, V, C, time_levels, order)
    end
end

"""
    TimeIntegrationSolution{T, S, NT}

Solution of the time integrator.

# Fields
- `N::Int`: Number varables in the system
- `solution::Matrix{NT}`: y, of size (N, num_steps)
- `scheme::T<:AbstractTimeIntegrator`: time integration scheme
- `startup_scheme::S<:Union{Nothing,AbstractTimeIntegrator}`: startup scheme
- `remaining_startup_steps::Int`: remaining startup steps
- `solution_alocated::Matrix{NT}`: used as pre-allocated memory for calculations
- `F_alocated::Matrix{NT}`: used as pre-allocated memory for calculations
- `G_alocated::Matrix{NT}`: used as pre-allocated memory for calculations
"""
mutable struct TimeIntegrationSolution{T, S, NT}
    N::Int
    solution::Matrix{NT}
    scheme::T
    startup_scheme::S
    remaining_startup_steps::Int
    solution_alocated::Matrix{NT}
    F_alocated::Matrix{NT}
    G_alocated::Matrix{NT}

    """
        TimeIntegrationSolution(
            solution::Matrix{NT},
            scheme::AbstractTimeIntegrator,
            startup_scheme::Union{Nothing, AbstractTimeIntegrator},
            remaining_startup_steps::Int,
        ) where {NT}

    Create a TimeIntegrationSolution

    # Arguments
    - `solution::Matrix{NT}`: y, of size (N, num_steps). Its `eltype` will dictate the
        number type used in the TimeIntegrationSolution.
    - `scheme::AbstractTimeIntegrator`: time integration scheme
    - `startup_scheme::Union{Nothing,AbstractTimeIntegrator}`: startup scheme
    - `remaining_startup_steps::Int`: remaining startup steps
    """
    function TimeIntegrationSolution(
        solution::Matrix{NT},
        scheme::AbstractTimeIntegrator{num_stages, num_steps},
        startup_scheme::Union{Nothing, AbstractTimeIntegrator},
        remaining_startup_steps::Int,
    ) where {NT, num_stages, num_steps}
        return new{typeof(scheme), typeof(startup_scheme), NT}(
            size(solution, 1),
            solution,
            scheme,
            startup_scheme,
            remaining_startup_steps,
            similar(solution),
            zeros(NT, size(solution, 1), num_stages),
            zeros(NT, size(solution, 1), num_stages),
        )
    end
end

function get_num_variables(sol::TimeIntegrationSolution)
    return sol.N
end
function get_solution(sol::TimeIntegrationSolution{T, S, NT}) where {T, S, NT}
    # The type annotation is needed to prevent Julia from converting a single-column matrix
    # to a vector, which would break matrix multiplication later on.
    return sol.solution::Matrix{NT}
end
function get_scheme(sol::TimeIntegrationSolution)
    return sol.scheme
end
function get_startup_scheme(sol::TimeIntegrationSolution)
    return sol.startup_scheme
end
function get_remaining_startup_steps(sol::TimeIntegrationSolution)
    return sol.remaining_startup_steps
end
function get_F_allocated(sol::TimeIntegrationSolution)
    return sol.F_alocated
end
function get_G_allocated(sol::TimeIntegrationSolution)
    return sol.G_alocated
end

### Overloading for convenience
Base.:(==)(tl::TimeLevels, tl2::TimeLevels) =
    tl.step_values == tl2.step_values &&
    tl.step_derivatives_implicit == tl2.step_derivatives_implicit &&
    tl.step_derivatives_explicit == tl2.step_derivatives_explicit
Base.:(!=)(tl::TimeLevels, tl2::TimeLevels) = !(tl == tl2)

# return a scalar max of all elements in the TimeLevels struct
Base.maximum(tl::TimeLevels) = maximum([
    maximum(tl.step_values; init=0),
    maximum(tl.step_derivatives_implicit; init=0),
    maximum(tl.step_derivatives_explicit; init=0),
])
