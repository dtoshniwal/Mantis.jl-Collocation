using Mantis
using GLMakie

m1, m2, m3 = 1.0, 1.0, 1.0

function get_forces(x)
    # x = [x1, y1, x2, y2, x3, y3]
    force = zeros(6)
    for i in 1:2:5
        for j in 1:2:5
            if i != j
                r = sqrt((x[i] - x[j])^2 + (x[i + 1] - x[j + 1])^2)
                force[i] += (x[j] - x[i]) / r^3 # x component
                force[i + 1] += (x[j + 1] - x[i + 1]) / r^3 # y component
            end
        end
    end
    return force
end

function f_velocity(vel, t; pos)
    force = get_forces(pos)
    return force
end
function f_position(pos, t; vel)
    return vel
end

ode_velocity = TimeIntegrators.define_explicit_ode(f_velocity)
ode_position = TimeIntegrators.define_explicit_ode(f_position)

dt = 0.05
T = 10.0
t0 = 0.0

# Broucke–Hénon–Hadjidemetriou family
function initialize()
    x = zeros(6)
    x[1] = 0.97000436
    x[2] = -0.24308753
    x[3] = -0.97000436
    x[4] = 0.24308753
    x[5] = 0.0
    x[6] = 0.0

    v = zeros(6)
    v[1] = -0.4662036850
    v[2] = -0.4323657300
    v[3] = -0.4662036850
    v[4] = -0.4323657300
    v[5] = 0.93240737
    v[6] = 0.86473146

    return x, v
end

x₀, v₀ = initialize()

x = copy(x₀)
v = copy(v₀)

method = TimeIntegrators.FORWARD_EULER
x_n = TimeIntegrators.initialize_scheme(x, method)
v_n = TimeIntegrators.initialize_scheme(v, method)

xs = TimeIntegrators.get_solution(x_n)[1:2:end]
ys = TimeIntegrators.get_solution(x_n)[2:2:end]

fps = 30
function animate_solution(
    xs, ys, x_n, v_n, T, t0, dt, fps, ode_velocity, ode_position, trail_length=30
)
    N_t = round(Int, T / dt)

    fig = Figure()
    ax = Axis(fig[1, 1]; title="t = 0.0", limits=(-1.2, 1.2, -1.2, 1.2))
    ax2 = Axis(fig[1, 2]; title="t = 0.0", limits=(0.0, 10.0, -0.4, 0.4))

    x_last = TimeIntegrators.get_solution(x_n)

    trail_x = [copy(xs) for _ in 1:trail_length]
    trail_y = [copy(ys) for _ in 1:trail_length]
    trail_1 = Point2f.(getindex.(trail_x, 1), getindex.(trail_y, 1))
    trail_2 = Point2f.(getindex.(trail_x, 2), getindex.(trail_y, 2))
    trail_3 = Point2f.(getindex.(trail_x, 3), getindex.(trail_y, 3))
    trail1 = Observable(trail_1)
    trail2 = Observable(trail_2)
    trail3 = Observable(trail_3)
    alphas = LinRange(0.0, 1.0, trail_length)
    colors = [(:black, i) for i in alphas]
    scatter!(ax, trail1; color=colors)
    scatter!(ax, trail2; color=colors)
    scatter!(ax, trail3; color=colors)

    energies = Float64[]
    drift = Observable(Float64[])
    ts = Observable(Float64[])
    lines!(ax2, ts, drift; color=:blue)
    display(fig)
    for i in 1:N_t
        t = t0 + dt * i
        TimeIntegrators.time_integrate!(
            v_n, ode_velocity, t, dt; pos=TimeIntegrators.get_solution(x_n)
        )
        TimeIntegrators.time_integrate!(
            x_n, ode_position, t+dt/2, dt; vel=TimeIntegrators.get_solution(v_n)
        )
        x = TimeIntegrators.get_solution(x_n)
        v = TimeIntegrators.get_solution(v_n)

        x_true = (x_last + x) / 2
        xs = x_true[1:2:end]
        ys = x_true[2:2:end]

        push!(trail_x, copy(xs))
        push!(trail_y, copy(ys))
        if length(trail_x) > trail_length
            popfirst!(trail_x)
            popfirst!(trail_y)
        end
        trail_1 = Point2f.(getindex.(trail_x, 1), getindex.(trail_y, 1))
        trail_2 = Point2f.(getindex.(trail_x, 2), getindex.(trail_y, 2))
        trail_3 = Point2f.(getindex.(trail_x, 3), getindex.(trail_y, 3))
        trail1[] = trail_1
        trail2[] = trail_2
        trail3[] = trail_3

        energy = 0.0
        for i in 1:2:5
            for j in 1:2:5
                if i != j
                    r = sqrt((x_true[i] - x_true[j])^2 + (x_true[i + 1] - x_true[j + 1])^2)
                    PE = -1.0 / r
                    energy += PE
                end
                KE = 0.5 * (v[i]^2 + v[i + 1]^2)
                energy += KE
            end
        end
        push!(energies, energy)
        drift[] = energies .- sum(energies)/i
        ts[] = LinRange(t0, T, N_t)[1:i]

        ax.title = "t = $(round(t,digits=1))"
        ax2.title = "t = $(round(t,digits=1))"
        x_last = x
        yield()
        sleep(1/fps) # refreshes the display!
    end

    return fig
end

animate_solution(xs, ys, x_n, v_n, T, t0, dt, fps, ode_velocity, ode_position)
