# # Three-body problem

# ## [Background knowledge](@id ThreeBodyBackground)
# The [three-body problem](https://en.wikipedia.org/wiki/Three-body_problem) is a classical
# physics problem where three bodies are orbiting each other and are thus interacting based
# on Newton's law of universal gravitation. From a computational point of view, the goal is
# to compute the trajecteries of all three bodies given the initial position and velocity
# of each body.
#
# The problem is an application of the Newton's second law and Newton's law of universal
# gravitation, that is, we can define the following system of equations:
# ```math
# \begin{cases}
#   \frac{\partial^2 x_1}{\partial t^2} = -Gm_2 \frac{\mathbf{x}_1 - \mathbf{x}_2}{||\mathbf{x}_1 - \mathbf{x}_2||^3} -Gm_3 \frac{\mathbf{x}_1 - \mathbf{x}_3}{||\mathbf{x}_1 - \mathbf{x}_3||^3}\;, \\
#   \frac{\partial^2 x_2}{\partial t^2} = -Gm_3 \frac{\mathbf{x}_2 - \mathbf{x}_3}{||\mathbf{x}_1 - \mathbf{x}_3||^3} -Gm_1 \frac{\mathbf{x}_2 - \mathbf{x}_1}{||\mathbf{x}_2 - \mathbf{x}_1||^3}\;, \\
#   \frac{\partial^2 x_3}{\partial t^2} = -Gm_1 \frac{\mathbf{x}_3 - \mathbf{x}_1}{||\mathbf{x}_1 - \mathbf{x}_1||^3} -Gm_2 \frac{\mathbf{x}_3 - \mathbf{x}_2}{||\mathbf{x}_3 - \mathbf{x}_2||^3}\;,
# \end{cases}
# ```
# where ``\mathbf{x}_i`` is the position of the ``i``-th body, and G is the gravitational
# constant (which will be set to ``1`` below).
#
# Instead of using one second order ODE for each body, we can reformulate the problem in
# terms of two ODEs for each body as
# ```math
# \begin{cases}
#   \frac{\partial v_i}{\partial t} = -G \sum_{j=1, j \neq i}^3 \frac{\mathbf{x}_i - \mathbf{x}_j}{||\mathbf{x}_i - \mathbf{x}_j||^3}\; \\
#   \frac{\partial x_i}{\partial t} = v_i\;.
# \end{cases}
# ```
# This system of two ODEs will be used in the implementation below. In this example, we
# will consider a 2D version of this problem.


# ## [Implementation](@id ThreeBodyImplementation)
# In this section, we will implement the above three-body problem using the
# `TimeIntegrators` module within `Mantis`.
#
# ### [Packages](@id ThreeBodyPackages)
# In this example, we will use `Mantis` for the integration, `GLMakie` to create an
# animation, and `Printf` to easily create a nicely formatted title for the animation. So,
# we are `using` these three packages.

using Mantis
using GLMakie
using Printf

# ### [Test case](@id ThreeBodyTestCase)
# In this example, we will consider the 'figure 8' periodic solution as described in
# [Chenciner2000](@cite). This means that the three bodies each have the same mass (we use
# the case where the mass is 1), and we define the initial positions and velocities as
# shown in the code. This solution is part of the Broucke–Hénon–Hadjidemetriou family.
m1 = 1.0
m2 = 1.0
m3 = 1.0

## x₀ = [x₁, y₁, x₂, y₂, x₃, y₃] and similarly for v₀.
x₀ = [
    0.97000436,
    -0.24308753,
    -0.97000436,
    0.24308753,
    0.0,
    0.0,
]
v₀ = [
    -0.4662036850,
    -0.4323657300,
    -0.4662036850,
    -0.4323657300,
    0.93240737,
    0.86473146,
]

# ### [Setting up the ODE](@id ThreeBodyODESetup)
# The ODE as described in the [background section](@ref ThreeBodyBackground) needs to be
# implemented. We can do this by creating the `get_forces` function, which will compute the
# gravitation force. We can then define the forcing functions for the position and velocity
# as `f_position` and `f_velocity`. The latter computes and return the force, while the
# former return the velocity. Note that both forcing functions take in a keyword argument,
# as the forces do not depend on ``t`` nor on the variable in the ODE itself.
function get_forces(x)
    ## x = [x1, y1, x2, y2, x3, y3]
    force = zeros(6)
    for i = 1:2:5
        for j = 1:2:5
            if i != j
                r = sqrt((x[i] - x[j])^2 + (x[i+1] - x[j+1])^2)
                force[i] += (x[j] - x[i]) / r^3 ## x component
                force[i+1] += (x[j+1] - x[i+1]) / r^3 ## y component
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

# With these forcing functions, we can define the ODEs by simply calling the
# [`Mantis.TimeIntegrators.define_explicit_ode`](@ref) function.
ode_velocity = TimeIntegrators.define_explicit_ode(f_velocity)
ode_position = TimeIntegrators.define_explicit_ode(f_position)

# ### [Time integration](@id ThreeBodyTimeIntegration)
# We can now setup the time integrator of our choosing. Since we are evolving two ODEs at
# the same time, and we would like to ensure that we preserve the Hamiltonian of the
# system (see [the Wikipedia page](https://en.wikipedia.org/wiki/Three-body_problem) for
# the Hamiltonian), we will stagger the two ODEs in time. Using the forward Euler method
# for each ODE while staggering, we will end up with the so-called symplectic Euler method.
#
# To make this work in `Mantis`, we setup the two solutions `x_n` and `v_n` using the
# previously defined initial conditions and using the forward Euler method in each.
const x_n = TimeIntegrators.initializeScheme(x₀, TimeIntegrators.FORWARD_EULER)
const v_n = TimeIntegrators.initializeScheme(v₀, TimeIntegrators.FORWARD_EULER)

function integrate!(x_n, v_n, tail_x, tail_y, t, tail_length)
    ## Advance both ODEs. Note that the velocity must be updated first, to ensure that the
    ## position update can use the new velocity. This is what makes this the symplectic
    ## Euler scheme.
    TimeIntegrators.timeIntegrate!(
        v_n, ode_velocity, t, dt; pos=TimeIntegrators.get_solution(x_n)
    )
    TimeIntegrators.timeIntegrate!(
        x_n, ode_position, t+dt/2, dt; vel=TimeIntegrators.get_solution(v_n)
    )

    ## Now we update the trails for the visualisation. If there are more entries than the
    ## desired `trail_length`, we remove the oldest position.
    x = TimeIntegrators.get_solution(x_n)

    push!(tail_x, x[1:2:end])
    push!(tail_y, x[2:2:end])
    if length(tail_x) > tail_length
        popfirst!(tail_x)
        popfirst!(tail_y)
    end

    return nothing
end

# Then we can set the desired start time t0 (should be 0.0), the time step size dt and the
# final time T.
const dt = 0.05
const T = 10.0
const t0 = 0.0

# The remaining code is just for the visualisation. We want to create an animation where we
# see the masses move in their orbit with a tail. We set the tail length to be ``30``,
# meaning that we see at most ``30`` points. We will also make them increasingly less
# visible, which is why we define the alphas. The animation is then created using the
# `record` function (from `GLMakie`) and `Makie`'s `Observables`. See the `Makie`
# documentation for more details on creating animations.
const tail_length = 30
tail_x = [copy(TimeIntegrators.get_solution(x_n)[1:2:end]) for _ in 1:tail_length]
tail_y = [copy(TimeIntegrators.get_solution(x_n)[2:2:end]) for _ in 1:tail_length]
alphas = LinRange(0.0, 1.0, tail_length)
colors = [(:black, i) for i in alphas]

time = Observable(t0)

## This part is doing the actual computation at each timestep, everything else is just for
## the animation.
integrator = lift(time) do t
    integrate!(x_n, v_n, tail_x, tail_y, t, 30)
end

tail1 = lift(integrator) do update
    return Point2f.(getindex.(tail_x, 1), getindex.(tail_y, 1))
end
tail2 = lift(integrator) do update
    return Point2f.(getindex.(tail_x, 2), getindex.(tail_y, 2))
end
tail3 = lift(integrator) do update
    return Point2f.(getindex.(tail_x, 3), getindex.(tail_y, 3))
end

fig = scatter(
    tail1;
    color=colors,
    axis = (
        title = @lift("t = $(@sprintf("%0.2f", round($time, digits = 2)))"),
        limits = (-1.2, 1.2, -1.2, 1.2),
        )
)
scatter!(tail2; color=colors)
scatter!(tail3; color=colors)

record(
    fig,
    "three_body_problem.mp4",
    LinRange(dt, T, round(Int, T / dt));
    framerate = 30
) do t
    time[] = t
end

# ```@raw html
# <video autoplay loop muted playsinline controls src="./three_body_problem.mp4" />
# ```
