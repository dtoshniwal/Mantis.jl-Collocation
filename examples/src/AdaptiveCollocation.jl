# # Adaptive collocation

# This example combines the two ideas developed separately in [Collocation](@ref) and
# [Adaptive refinement](@ref). We solve a 1D Poisson problem by strong-form collocation, but on
# a *hierarchical* B-spline space that we refine locally. The exact solution has a sharp peak
# near ``x = 1``; we refine only the elements under that peak and watch the collocation error
# fall.

# The new ingredient is the choice of collocation points. On a single B-spline space the
# Greville abscissae give a natural point set, as in the [Collocation](@ref) example. A locally
# refined hierarchical mesh, however, contains active elements from several refinement levels at
# once, so a single global point set no longer fits. We handle this by choosing a set of
# collocation points *for each level*, and then, for every active element of level ``\ell``,
# keeping the level-``\ell`` points that lie inside it.

# ## A solution with a local feature

# We use the manufactured solution ``u(x) = x^{a}\,(1 - x)`` with ``a = 20``. It vanishes at
# both ends, so the boundary conditions are homogeneous, and it has a narrow peak near
# ``x = a/(a+1) \approx 0.95``. A coarse uniform mesh cannot resolve that peak, which is exactly
# the situation local refinement is meant for.

# As in the [Collocation](@ref) example, the pointwise codifferential ``\delta(\mathrm{d}(u^0))``
# equals the analyst Laplacian ``u''``, so the right-hand side field we collocate is ``u''``
# itself. For this ``u``,
# ```math
# u'' = a\,(a - 1)\,x^{a - 2} - (a + 1)\,a\,x^{a - 1} \;.
# ```

using Mantis

a = 20

function exact_expression(x::Matrix{Float64})
    return [vec(@. x[:, 1]^a * (1 - x[:, 1]))]
end

function forcing_expression(x::Matrix{Float64})
    return [vec(@. a * (a - 1) * x[:, 1]^(a - 2) - (a + 1) * a * x[:, 1]^(a - 1))]
end

# ## The coarse hierarchical space

# We start from a single B-spline space of degree 3 and regularity 2 (so the basis is ``C^2``,
# smooth enough for the pointwise Laplacian) on 5 elements. Wrapping it in a
# `HierarchicalFiniteElementSpace` with no refinement gives the coarsest level of the hierarchy;
# each refined element will later be split into `num_subdivisions = (2,)` children.

p = (3,)
k = (2,)
num_subdivisions = (2,)

B = FunctionSpaces.create_bspline_space((0.0,), (1.0,), (5,), p, k)
H = FunctionSpaces.HierarchicalFiniteElementSpace(B, num_subdivisions, true, false)

# ## Collocation points on a hierarchical mesh

# For each level we take that level's Greville abscissae as the candidate collocation points,
# given in the parametric coordinate of the unit interval. We then walk over the active elements
# of the hierarchical mesh: for an element of level ``\ell`` we keep the level-``\ell`` points
# lying within the element and record them in element-local coordinates. A point sitting on a
# shared element boundary belongs to both neighbours, so we collocate it once.
#
# The result is packaged as a `CollocationPoints` object, the same structure the built-in
# `GrevilleCollocation` produces, so the rest of the pipeline is unchanged. The points carry no
# point-to-basis bijection, so the strong boundary conditions later go through the least-squares
# path described in the [Collocation](@ref) example.

function hierarchical_collocation_points(H)
    num_elements = FunctionSpaces.get_num_elements(H)
    num_levels = FunctionSpaces.get_num_levels(H)

    level_points = [
        FunctionSpaces.get_greville_points(FunctionSpaces.get_space(H, l))[1] for
        l in 1:num_levels
    ]

    element_local_points = Vector{NTuple{1, Vector{Float64}}}(undef, num_elements)
    element_point_ids = Vector{Vector{Int}}(undef, num_elements)
    claimed = Set{Int}()  # parametric coordinates already collocated (rounded to a key)
    num_points = 0

    for element in 1:num_elements
        level, _ = FunctionSpaces.convert_to_element_level_and_level_id(H, element)
        left, right = FunctionSpaces.get_element_vertices(H, element)[1]

        local_coords = Float64[]
        ids = Int[]
        for point in level_points[level]
            (left - 1e-12) <= point <= (right + 1e-12) || continue
            key = round(Int, point * 1e9)
            key in claimed && continue
            push!(claimed, key)
            push!(local_coords, (point - left) / (right - left))
            num_points += 1
            push!(ids, num_points)
        end

        element_local_points[element] = (local_coords,)
        element_point_ids[element] = ids
    end

    return Assemblers.CollocationPoints{1}(
        element_local_points, element_point_ids, num_points, false
    )
end

# ## Solving and measuring the error

# Given a hierarchical space, the solve is the collocation pipeline from before: build the form
# space, the strong-form block ``\delta(\mathrm{d}(u^0))``, the forcing, and the collocation
# points, then assemble with homogeneous Dirichlet conditions and solve. The error is measured
# with a Gauss-Legendre rule on the active mesh.

function solve_collocation(H)
    Λ⁰ = Forms.FormSpace(0, H, "u")
    geometry = Forms.get_geometry(Λ⁰)

    f⁰ = Forms.AnalyticalFormField(0, forcing_expression, geometry, "f")
    uₑ = Forms.AnalyticalFormField(0, exact_expression, geometry, "u")

    points = hierarchical_collocation_points(H)
    inputs = Assemblers.CollocationInputs(Λ⁰, f⁰, points)
    u⁰ = Assemblers.get_trial_form(inputs)
    collocation_form = Assemblers.CollocationForm(((δ(d(u⁰)),),), ((f⁰,),), inputs)

    bc = Forms.set_dirichlet_boundary_conditions(Λ⁰, 0.0)
    A, b = Assemblers.assemble(collocation_form, bc)
    uₕ = Forms.build_form_field(Λ⁰, vec(A \ b))

    qrule = Quadrature.tensor_product_rule(p .+ 5, Quadrature.gauss_legendre)
    dΩ = Quadrature.StandardQuadrature(qrule, Geometry.get_num_elements(geometry))
    error = Analysis.L2_norm(uₕ - uₑ, dΩ)

    return error, Assemblers.get_num_points(points)
end

# ## Refining the last two elements

# Each adaptive step marks the two right-most elements of the finest active level and refines
# them. `refine_space` takes the marked elements per level and returns a new hierarchical space
# whose nested domains have been updated.

function finest_active_level(H)
    for level in FunctionSpaces.get_num_levels(H):-1:1
        isempty(FunctionSpaces.get_level_element_ids(H, level)) || return level
    end
    return 1
end

function refine_last_two(H)
    level = finest_active_level(H)
    element_ids = sort(FunctionSpaces.get_level_element_ids(H, level))
    marked = [Int[] for _ in 1:FunctionSpaces.get_num_levels(H)]
    marked[level] = element_ids[(end - 1):end]
    return FunctionSpaces.refine_space(H, marked)
end

# ## The adaptive loop

# We solve on the coarse mesh, refine the last two elements, solve again, and refine once more.
# The number of active basis functions, the number of collocation points, and the error are
# printed at each step.

for step in 0:2
    error, num_points = solve_collocation(H)
    num_basis = FunctionSpaces.get_num_basis(H)
    num_active = FunctionSpaces.get_num_elements(H)
    println(
        "step $step: active elements = $num_active, basis = $num_basis, ",
        "points = $num_points, L² error = $error",
    )
    global H = refine_last_two(H)
end

# The error drops at every step even though only two elements are refined each time, because the
# refinement tracks the peak of the solution. A uniform mesh would need far more degrees of
# freedom to reach the same accuracy.
