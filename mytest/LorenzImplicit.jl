using Mantis
using GLMakie

dt::Float64 = 0.02
σ::Float64 = 10
ρ::Float64 = 28
β::Float64 = 8/3

y₀ = [2.0, 1.0, 2.5]
# dt::Float64 = 0.003

function g(sol, t)
    x, y, z = sol
    return [σ * (y - x), x * (ρ - z) - y, x * y - β * z]
end

function J(sol, t)
    x, y, z = sol
    return [
        -σ σ 0.0;
        ρ-z -1.0 -x;
        y x -β
    ]
end

ode = TimeIntegrators.define_newton_solver_ode(g; jacobian=J, iter=50, eps=1e-14)
# method = ti.CRANK_NICOLSON
# method = ti.GAUSS_LEGENDRE_6
method = TimeIntegrators.DIRK3
method2 = TimeIntegrators.IMPLICIT_MIDPOINT

y1_n = TimeIntegrators.initialize_scheme(y₀, method)
y2_n = TimeIntegrators.initialize_scheme(y₀, method2)

fig = Figure()
ax = Axis3(fig[1, 1]; title="t = 0.0", limits=(-30, 30, -30, 30, 0, 60), viewmode=:fit)

t = 0.0
display(fig)
for i in 1:1000
    global y1_n, y2_n, dt, t
    y1_n, y2_n, dt, t = TimeIntegrators.time_integrate(y1_n, y2_n, ode, t, dt)
    y = TimeIntegrators.get_solution(y1_n)
    scatter!(ax, y[1], y[2], y[3]; color=(:blue, 0.2), fxaa=true, transparency=true)

    ax.title = "t = $(round(t,digits=1))"

    yield()
end
