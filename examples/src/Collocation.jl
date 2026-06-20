# # Collocation

# Most examples in this guide discretise a PDE in weak (Galerkin) form: the equation is
# multiplied by a test form and integrated, and the integrals are evaluated with a quadrature
# rule. `Mantis` also supports collocation, where the strong form of the PDE is enforced
# pointwise at a finite set of points. There is no test space and no integration. Each
# collocation point contributes one row to the linear system, found by evaluating the
# strong-form operator there.
#
# In this example we solve the Poisson problem by collocation. We start with the Greville
# abscissae, which give one collocation point per basis function. That makes the system square
# and lets it carry Dirichlet boundary conditions. We then feed custom collocation points
# through the same assembly pipeline.

# ## Formulation

# We solve the Poisson problem on a domain ``\Omega`` with homogeneous Dirichlet boundary
# conditions,
# ```math
# \begin{alignat*}{2}
#     -\Delta u &= f  \quad &&\text{on}\ \Omega \;, \\
#     u &= 0  \quad &&\text{on}\ \partial\Omega \;.
# \end{alignat*}
# ```
# In `Mantis` the Laplacian of a ``0``-form is the codifferential of its exterior derivative,
# ``\Delta u^0 = \delta\,\mathrm{d}\,u^0``. Evaluated *pointwise*, ``\delta(\mathrm{d}(u^0))``
# returns the analyst's Laplacian ``+\Delta u = \sum_i \partial^2 u/\partial x_i^2``, so the
# Poisson problem ``-\Delta u = f`` is collocated as
# ```math
# \delta(\mathrm{d}(u^0)) = -f \;.
# ```
# We use the manufactured solution ``u = \prod_i \sin(\omega x_i)``, for which
# ``-\Delta u = \mathrm{(manifold\ dim)}\,\omega^2\,u``. The right-hand side field is therefore
# ``-f = -\mathrm{(manifold\ dim)}\,\omega^2\,u``. Writing the forcing in terms of
# `size(x, 2)` (the number of spatial coordinates) keeps the expression independent of the
# dimension, so the same function works for both the 1D and 2D solves below.

using Mantis

ω = 2.0 * pi

function exact_expression(x::Matrix{Float64})
    return [vec(prod(sin.(ω .* x); dims=2))]
end

function forcing_expression(x::Matrix{Float64})
    manifold_dim = size(x, 2)
    y = prod(sin.(ω .* x); dims=2)
    return [vec(@. -manifold_dim * ω * ω * y)]
end

# ## Greville collocation

# We work on the unit interval with maximally smooth cubic B-splines. The strong form needs
# the Laplacian of the basis pointwise, so the basis must be ``C^2`` (or smoother) at the
# collocation points; degree ``p = 3`` with regularity ``k = 2`` satisfies this.

origin       = (0.0,)
box_size     = (1.0,)
num_elements = (16,)
p = (3,)
k = (2,)

B  = FunctionSpaces.create_bspline_space(origin, box_size, num_elements, p, k)
Λ⁰ = Forms.FormSpace(0, B, "u")
geometry = Forms.get_geometry(Λ⁰)

f⁰ = Forms.AnalyticalFormField(0, forcing_expression, geometry, "f")
uₑ = Forms.AnalyticalFormField(0, exact_expression, geometry, "u")

# The collocation points say where the equations are imposed, much as a quadrature rule says
# where the integrals are evaluated in Galerkin assembly. `GrevilleCollocation` builds one
# Greville point per basis function, so the system is square and the ``i``-th point belongs to
# the ``i``-th basis function. This point-to-basis bijection lets the Dirichlet machinery,
# keyed by boundary basis index, reuse the Galerkin code path.

greville_points = Assemblers.GrevilleCollocation(B)

@assert Assemblers.get_num_points(greville_points) == FunctionSpaces.get_num_basis(B)
@assert Assemblers.is_bijective(greville_points)

# The assembly inputs collect the trial form, the forcing, and the collocation points. There
# is no test space, which is the structural difference from `WeakFormInputs`.

inputs = Assemblers.CollocationInputs(Λ⁰, f⁰, greville_points)
u⁰ = Assemblers.get_trial_form(inputs)

# The `CollocationForm` holds the strong-form blocks directly: the left-hand side is the bare
# operator ``\delta(\mathrm{d}(u^0))``, not an integral, and the right-hand side is the forcing
# field ``-f``. Here each is a one-by-one block of forms, matching the single trial form.

collocation_form = Assemblers.CollocationForm(((δ(d(u⁰)),),), ((f⁰,),), inputs)

# Dirichlet conditions are set exactly as in the Galerkin examples. During assembly the rows
# belonging to boundary basis functions are replaced by the identity equation
# ``u_\text{boundary} = \text{value}``.

bc = Forms.set_dirichlet_boundary_conditions(Λ⁰, 0.0)

A, b = Assemblers.assemble(collocation_form, bc)
uₕ = Forms.build_form_field(Λ⁰, vec(A \ b))

# We measure the error with a Gauss-Legendre rule finer than the solution space so that
# quadrature error does not pollute the result.

function l2_error(uₕ, uₑ, geometry, p)
    qrule = Quadrature.tensor_product_rule(p .+ 3, Quadrature.gauss_legendre)
    dΩ = Quadrature.StandardQuadrature(qrule, Geometry.get_num_elements(geometry))
    return Analysis.L2_norm(uₕ - uₑ, dΩ)
end

println("Greville collocation L² error: ", l2_error(uₕ, uₑ, geometry, p))

# ### Convergence study

# Wrapping the solve in a function lets us refine the mesh and check that the error decreases
# at the rate expected of isogeometric collocation. The body repeats the pipeline above.

function solve_poisson_greville(num_elements)
    B = FunctionSpaces.create_bspline_space(origin, box_size, num_elements, p, k)
    Λ⁰ = Forms.FormSpace(0, B, "u")
    geometry = Forms.get_geometry(Λ⁰)

    f⁰ = Forms.AnalyticalFormField(0, forcing_expression, geometry, "f")
    uₑ = Forms.AnalyticalFormField(0, exact_expression, geometry, "u")

    points = Assemblers.GrevilleCollocation(B)
    inputs = Assemblers.CollocationInputs(Λ⁰, f⁰, points)
    u⁰ = Assemblers.get_trial_form(inputs)
    collocation_form = Assemblers.CollocationForm(((δ(d(u⁰)),),), ((f⁰,),), inputs)

    bc = Forms.set_dirichlet_boundary_conditions(Λ⁰, 0.0)
    A, b = Assemblers.assemble(collocation_form, bc)
    uₕ = Forms.build_form_field(Λ⁰, vec(A \ b))

    return l2_error(uₕ, uₑ, geometry, p)
end

errors = [solve_poisson_greville((n,)) for n in (8, 16, 32, 64)]
rates = log2.(errors[1:(end - 1)] ./ errors[2:end])

println("errors: ", errors)
println("rates:  ", rates)

# The error decreases monotonically under refinement, and the rates approach the order
# expected of collocation.

# ## Custom collocation points

# Greville abscissae are a convenient default, but collocation does not require them. The
# `UserCollocation` constructor takes a tensor-product set of points, given per direction in
# parametric coordinates, and routes them through the same assembly pipeline. This lets us
# choose any point distribution, including over-collocation: using more points than basis
# functions, which turns the square system into a rectangular least-squares one.

# Here we place twice as many points as basis functions on a uniform grid over ``[0, 1]``. In
# higher dimensions you would pass one coordinate vector per direction, e.g. `(xs, ys)` for a
# 2D tensor-product grid.

num_basis = FunctionSpaces.get_num_basis(B)
custom_points = (collect(LinRange(0.0, 1.0, 2 * num_basis)),)
user_points = Assemblers.UserCollocation(B, custom_points)

@assert Assemblers.get_num_points(user_points) == 2 * num_basis

# A user-supplied set does not, in general, carry a point-to-basis bijection, so `is_bijective`
# reports `false`.

Assemblers.is_bijective(user_points)

# Building the inputs and the strong-form blocks is identical to the Greville case.

inputs_user = Assemblers.CollocationInputs(Λ⁰, f⁰, user_points)
u⁰_user = Assemblers.get_trial_form(inputs_user)
collocation_form_user = Assemblers.CollocationForm(
    ((δ(d(u⁰_user)),),), ((f⁰,),), inputs_user
)

# The boundary conditions are imposed exactly as in the Greville case: build the same
# `basis index => value` dictionary and pass it to `assemble`. They are enforced strongly
# whether the strong-form collocation gives a square system or, as here with twice as many
# points as basis functions, a rectangular one. For a non-bijective set `assemble` lifts the
# prescribed boundary coefficients to the right-hand side and appends one constraint row per
# boundary basis function. The resulting over-determined system still has one column per basis
# function, and `\` solves it in the least-squares sense.

bc = Forms.set_dirichlet_boundary_conditions(Λ⁰, 0.0)
A_user, b_user = Assemblers.assemble(collocation_form_user, bc)
uₕ_user = Forms.build_form_field(Λ⁰, vec(A_user \ b_user))

println("User over-collocation L² error: ", l2_error(uₕ_user, uₑ, geometry, p))

# ## In higher dimensions

# Nothing in the pipeline is specific to 1D. `GrevilleCollocation` works for any
# tensor-product spline space, and the strong-form blocks are written the same way. We repeat
# the Greville solve on the unit square to confirm this.

origin_2D       = (0.0, 0.0)
box_size_2D     = (1.0, 1.0)
num_elements_2D = (16, 16)
p_2D = (3, 3)
k_2D = (2, 2)

B_2D  = FunctionSpaces.create_bspline_space(origin_2D, box_size_2D, num_elements_2D, p_2D, k_2D)
Λ⁰_2D = Forms.FormSpace(0, B_2D, "u")
geometry_2D = Forms.get_geometry(Λ⁰_2D)

f⁰_2D = Forms.AnalyticalFormField(0, forcing_expression, geometry_2D, "f")
uₑ_2D = Forms.AnalyticalFormField(0, exact_expression, geometry_2D, "u")

points_2D = Assemblers.GrevilleCollocation(B_2D)
inputs_2D = Assemblers.CollocationInputs(Λ⁰_2D, f⁰_2D, points_2D)
u⁰_2D = Assemblers.get_trial_form(inputs_2D)
collocation_form_2D = Assemblers.CollocationForm(((δ(d(u⁰_2D)),),), ((f⁰_2D,),), inputs_2D)

bc_2D = Forms.set_dirichlet_boundary_conditions(Λ⁰_2D, 0.0)
A_2D, b_2D = Assemblers.assemble(collocation_form_2D, bc_2D)
uₕ_2D = Forms.build_form_field(Λ⁰_2D, vec(A_2D \ b_2D))

println("2D Greville collocation L² error: ", l2_error(uₕ_2D, uₑ_2D, geometry_2D, p_2D))
