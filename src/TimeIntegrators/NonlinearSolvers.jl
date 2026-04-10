"""
    define_newton_solver_ode(ode; eps::Float64=1e-15, iter::Int64=10, jacobian::Function)

Function that defines the Newton solver for a given ODE

# Arguments
- `ode::Function`: Function that defines the ODE

# Keyword Arguments
- `eps::Float64=1e-15`: Tolerance for the Newton solver
- `iter::Int=10`: Maximum number of iterations
- `jacobian::Function`: Jacobian of the ODE with signature
    func(y::Vector{Float64}, t::Float64)::Matrix{Float64}
"""
function define_newton_solver_ode(
    ode; eps::Float64=1e-15, iter::Int=10, jacobian::Function
)
    return define_implicit_ode(
        (x, λ, t) -> _newton_solve(ode, jacobian, x, λ, t; eps=eps, iter=iter)
    )
end

"""
    define_picard_solver_ode(ode; eps::Float64=1e-15, iter::Int64=10)

Function that defines the Picard solver for a given ODE

# Arguments
- `ode::Function` : Function that defines the ODE

# Keyword Arguments
- `eps::Float64=1e-15` : Tolerance for the Picard solver
- `iter::Int64=10` : Maximum number of iterations
"""
function define_picard_solver_ode(ode; eps::Float64=1e-15, iter::Int64=10)
    return define_implicit_ode(
        (x, λ, t) -> _picard_solve(ode, x, λ, t; eps=eps, iter=iter)
    )
end

"""
    define_fixed_point_relaxation_ode(ode; eps::Float64=1e-15, iter::Int64=10, omega::Float64=0.8)

Function that defines the Fixed Point Relaxation solver for a given ODE

# Arguments
- ode::Function : Function that defines the ODE

# Keyword Arguments
- eps::Float64=1e-15 : Tolerance for the Fixed Point Relaxation solver
- iter::Int64=10 : Maximum number of iterations
- omega::Float64=0.8 : Relaxation parameter
"""
function define_fixed_point_relaxation_ode(
    ode; eps::Float64=1e-15, iter::Int64=10, omega::Float64=0.8
)
    return define_implicit_ode(
        (x, λ, t) -> _fixed_point_relaxation(ode, x, λ, t; eps=eps, iter=iter, omega=omega)
    )
end

function _newton_solve(
    f::Function,
    jacobian::Function,
    y::Vector{Float64},
    λ::Float64,
    t::Float64;
    eps::Float64=1e-6,
    iter::Int64=10,
)
    yi = copy(y)
    for i in 1:iter
        residual = yi - y - λ * f(yi, t)
        if norm(residual) < eps
            return yi
        end
        J::Matrix{Float64} = jacobian(yi, t)
        diff = -(I - λ * J) \ residual
        yi += diff
    end
    return yi
end

function _picard_solve(
    f::Function,
    y::Vector{Float64},
    λ::Float64,
    t::Float64;
    eps::Float64=1e-6,
    iter::Int64=10,
)
    yi = copy(y)
    for i in 1:iter
        residual = yi - y - λ * f(yi, t)
        if norm(residual) < eps
            return yi
        end
        yi = y + λ * f(yi, t)
    end

    return yi
end

# https://mooseframework.inl.gov/bison/syntax/Executioner/FixedPointAlgorithms/
function _fixed_point_relaxation(
    f::Function,
    y::Vector{Float64},
    λ::Float64,
    t::Float64;
    eps::Float64=1e-16,
    iter::Int64=10,
    omega::Float64=0.8,
)
    yi = copy(y)
    for i in 1:iter
        residual = yi - y - λ * f(yi, t)
        if norm(residual) < eps
            return yi
        end
        yi = yi + omega * (y + λ * f(yi, t) - yi)
    end

    return yi
end
