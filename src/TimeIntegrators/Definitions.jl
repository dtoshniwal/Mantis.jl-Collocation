"""
    AbstractTimeIntegrator{num_stages, num_steps}

Supertype for all time integrators.

# Type parameters
- `num_stages`: The number of stages for a multi-step scheme (such as the Runge-Kutta
    family). Since every scheme is at least a single-stage scheme, `num_stages` >= 1.
- `num_steps`: The number of steps for a multi-step scheme (such as the Adams-Bashforth
    family). Since every scheme is at least a single-step scheme, `num_steps` >= 1.
"""
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
    TimeIntegrationOperators{EF, IF}

Defines the ODE-specific operators used in the time integration.

Makes a distiction between the explicit and implicit operators. `EF` and `IF` and the types
of the explicit and implicit functions, respectively, which are `Nothing` if not defined.
Note that at least one of `EF` and `IF` must be a function.

# Fields
- `explicitEvaluate::EF`: A function that evaluates the explicit part of the ODE, that is,
    the function that evaluates F = f(y). See the manual section on [TimeIntegrators](@ref)
    for the terminology. Note that the function must take in a `Vector`, and return a
    `Vector` (with the same `eltype`).
- `implicitSolve::IF`: A function that solves the implicit part of the ODE, that is, the
    function that solves the equation Y - λ g(Y) = x. See the manual section on
    [TimeIntegrators](@ref) for the terminology. In case g is a linear operator, a direct
    solution method can be through the inverse operator (I − λg)⁻¹, where I is the identity
    function.

# Constructors
- `define_explicit_ode(explicit_evaluate::Function)`: For fully explicit ODEs.
- `define_implicit_ode(implicit_solve::Function)`: For fully implicit ODEs.
- `define_imex_ode(explicit_evaluate::Function, implicit_solve::Function)`: For IMEX ODEs.
"""
struct TimeIntegrationOperators{EF, IF, IE}
    explicitEvaluate::EF
    implicitSolve::IF
    implicitEvaluate::IE

    function TimeIntegrationOperators(
        explicit_evaluate::Union{Nothing, Function},
        implicit_solve::Union{Nothing, Function},
        implicit_evaluate::Union{Nothing, Function},
    )
        new{typeof(explicit_evaluate), typeof(implicit_solve), typeof(implicit_evaluate)}(
            explicit_evaluate, implicit_solve, implicit_evaluate
        )
    end
end

"""
    define_explicit_ode(explicit_evaluate::Function)

Creates a [`TimeIntegrationOperators`](@ref) object for an explicit ODE.
"""
function define_explicit_ode(explicit_evaluate::Function)
    return TimeIntegrationOperators(explicit_evaluate, nothing, nothing)
end
"""
    define_implicit_ode(implicit_solve::Function)

Creates a [`TimeIntegrationOperators`](@ref) object for an implicit ODE. This function is
called by the [`define_implicit_linear`](@ref) functions when defining an implicit ODE.
"""
function define_implicit_ode(implicit_solve::Function, implicit_evaluate::Function)
    return TimeIntegrationOperators(nothing, implicit_solve, implicit_evaluate)
end
"""
    define_imex_ode(explicit_evaluate::Function, implicit_solve::Function)

Creates a [`TimeIntegrationOperators`](@ref) object for an IMEX ODE.
"""
function define_imex_ode(explicit_evaluate::Function, implicit_solve::Function, implicit_evaluate::Function)
    return TimeIntegrationOperators(explicit_evaluate, implicit_solve, implicit_evaluate)
end

"""
    define_implicit_linear(M::AbstractMatrix{T}, K::AbstractMatrix{T}, F::AbstractVector{T}) where {T}
    define_implicit_linear(M::Nothing, K::Nothing, F::AbstractVector{T}) where {T}
    define_implicit_linear(M::Nothing, K::AbstractMatrix{T}, F::Nothing) where {T}
    define_implicit_linear(M::Nothing, K::AbstractMatrix{T}, F::AbstractVector{T}) where {T}
    define_implicit_linear(M::AbstractMatrix{T}, K::AbstractMatrix{T}, F::Nothing) where {T}
    define_implicit_linear(M::AbstractMatrix{T}, K::Nothing, F::AbstractVector{T}) where {T}

Function that defines the implicit solver for a linear operator of the form Mx - λKx = F.
At least `K` or `F` should be defined.

# Arguments
- `M`: Mass matrix
- `K`: Stiffness matrix
- `F`: Forcing vector

# Returns
- `::TimeIntegrationOperators`: See [`TimeIntegrationOperators{EF, IF}`](@ref).
"""
function define_implicit_linear(
    M::AbstractMatrix{T}, K::AbstractMatrix{T}, F::AbstractVector{T}, g
) where {T}
    return define_implicit_ode((x, λ, t) -> (M - λ * K) \ (M * x + λ * F), g)
end
function define_implicit_linear(M::Nothing, K::Nothing, F::AbstractVector{T}, g) where {T}
    return define_implicit_ode((x, λ, t) -> x + λ * F, g)
end
function define_implicit_linear(M::Nothing, K::AbstractMatrix{T}, F::Nothing, g) where {T}
    return define_implicit_ode((x, λ, t) -> (I - λ * K) \ x, g)
end
function define_implicit_linear(M::Nothing, K, F::Nothing, g)
    return define_implicit_ode((x, λ, t) -> (I - λ * K) \ x, g)
end
function define_implicit_linear(
    M::Nothing, K::AbstractMatrix{T}, F::AbstractVector{T}, g
) where {T}
    return define_implicit_ode((x, λ, t) -> (I - λ * K) \ (x + λ * F), g)
end
function define_implicit_linear(
    M::AbstractMatrix{T}, K::AbstractMatrix{T}, F::Nothing, g
) where {T}
    return define_implicit_ode((x, λ, t) -> (M - λ * K) \ (M * x), g)
end
function define_implicit_linear(
    M::AbstractMatrix{T}, K::Nothing, F::AbstractVector{T}, g
) where {T}
    return define_implicit_ode((x, λ, t) -> x + λ * (M \ F), g)
end
function define_implicit_linear(M::AbstractMatrix{T}, K::Nothing, F::Nothing, g) where {T}
    throw(ArgumentError("At least one of the matrices K or F has to be defined"))
end
function define_implicit_linear(M::Nothing, K::Nothing, F::Nothing, g)
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
function define_implicit_linear(func::Function, g)
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
        return define_implicit_ode(implicitSolveMK, g)
    elseif F !== nothing && K === nothing && M !== nothing
        return define_implicit_ode(implicitSolveMF, g)
    elseif F !== nothing && K !== nothing && M === nothing
        return define_implicit_ode(implicitSolveKF, g)
    else
        return define_implicit_ode(implicitSolveMKF, g)
    end
end

"""
    TimeLevels

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
    IMEX{num_stages, num_steps, NT, AA, AE, EE} <: AbstractTimeIntegrator{num_stages, num_steps}

Implicit-Explicit (IMEX) time integration scheme.

# Fields
- `A_IM::SMatrix{num_stages, num_stages, Float64}`: Implicit matrix A.
- `A_EX::SMatrix{num_stages, num_stages, Float64}`: Explicit matrix A.
- `B_IM::SMatrix{num_steps, num_stages, Float64}`: Implicit matrix B.
- `B_EX::SMatrix{num_steps, num_stages, Float64}`: Explicit matrix B.
- `U::SMatrix{num_stages, num_steps, Float64}`: Matrix U.
- `V::SMatrix{num_steps, num_steps, Float64}`: Matrix V.
- `C_IM::SVector{num_stages, Float64}`: time Vector C of size num_stages, indicates at what time the stage is evaluated
- `C_EX::SVector{num_stages, Float64}`: time Vector C of size num_stages, indicates at what time the stage is evaluated
- `time_levels::TimeLevels`: reflects the structure of the input/output vector y associated to the scheme
- `order::Int`: order of the scheme

# Type parameters
- `num_stages`: number of stages of the scheme
- `num_steps`: number of external steps of the scheme
"""
struct IMEX{num_stages, num_steps, NT, AA, AE, EE} <:
    AbstractTimeIntegrator{num_stages, num_steps}
    A_IM::SMatrix{num_stages, num_stages, NT, AA}
    A_EX::SMatrix{num_stages, num_stages, NT, AA}
    B_IM::SMatrix{num_steps, num_stages, NT, AE}
    B_EX::SMatrix{num_steps, num_stages, NT, AE}
    U::SMatrix{num_stages, num_steps, NT, AE}
    V::SMatrix{num_steps, num_steps, NT, EE}
    C_IM::SVector{num_stages, NT}
    C_EX::SVector{num_stages, NT}
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
    Explicit{num_stages, num_steps, NT, AA, AE, EE} <: AbstractTimeIntegrator{num_stages, num_steps}

Explicit time integration scheme.

!!! note "Explicit time integrators are explicit in the ODE sense"
    Following [Vos2011](@cite), the explicit time integrators in this framework are
    considered explicit integrators when applied to ODEs. When applied to PDEs using a
    Galerkin method, one still has to solve a linear system. This can be referred to as an
    indirect explicit method in this case.

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
    A::SMatrix{num_stages, num_stages, NT, AA}
    B::SMatrix{num_steps, num_stages, NT, AE}
    U::SMatrix{num_stages, num_steps, NT, AE}
    V::SMatrix{num_steps, num_steps, NT, EE}
    C::SVector{num_stages, NT}
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
    struct DiagonallyImplicit{num_stages, num_steps} <: AbstractTimeIntegrator{num_stages, num_steps}

DiagonallyImplicit time integration scheme

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
struct DiagonallyImplicit{num_stages, num_steps, NT, AA, AE, EE} <: AbstractTimeIntegrator{num_stages, num_steps}
    A::SMatrix{num_stages, num_stages, NT, AA}
    B::SMatrix{num_steps, num_stages, NT, AE}
    U::SMatrix{num_stages, num_steps, NT, AE}
    V::SMatrix{num_steps, num_steps, NT, EE}
    C::SVector{num_stages, NT}
    time_levels::TimeLevels
    order::Int

    function DiagonallyImplicit(
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
    A::SMatrix{num_stages, num_stages, NT, AA}
    B::SMatrix{num_steps, num_stages, NT, AE}
    U::SMatrix{num_stages, num_steps, NT, AE}
    V::SMatrix{num_steps, num_steps, NT, EE}
    C::SVector{num_stages, NT}
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

function get_order(scheme::AbstractTimeIntegrator)
    return scheme.order
end


"""
    TimeIntegrationSolution{T, S, NT}

Solution of the time integrator.

# Constructors
- `TimeIntegrationSolution(
    solution::Matrix{NT},
    scheme::AbstractTimeIntegrator,
    startup_scheme::Union{Nothing, AbstractTimeIntegrator},
    remaining_startup_steps::Int,
) where {NT}`: General constructor. Note that the eltype of the solution matrix will
    dictate the number type used in the `TimeIntegrationSolution`.

# Fields
- `N::Int`: Number varables in the system.
- `solution::Matrix{NT}`: Of size (`N`, num_steps).
- `scheme::T<:AbstractTimeIntegrator`: The time integration scheme.
- `startup_scheme::S<:Union{Nothing, AbstractTimeIntegrator}`: The startup scheme, if
    desired.
- `remaining_startup_steps::Int`: remaining startup steps
- `solution_alocated::Matrix{NT}`: Pre-allocated memory for calculations.
- `F_alocated::Matrix{NT}`: Pre-allocated memory for calculations.
- `G_alocated::Matrix{NT}`: Pre-allocated memory for calculations.
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

Base.eltype(::Type{TimeIntegrationSolution{T, S, NT}}) where {T, S, NT} = NT

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
