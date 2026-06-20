############################################################################################
#                                   Collocation points                                     #
############################################################################################
"""
    AbstractCollocationPoints{manifold_dim}

Supertype for all collocation-point sets. A collocation-point set plays, for collocation
assembly, the role that an `Quadrature.AbstractGlobalQuadratureRule` plays for Galerkin
assembly: it specifies, per element, the points at which the strong form is evaluated and
the global row index that each point contributes to in the linear system.

# Type parameters
- `manifold_dim`: Dimension of the manifold on which the points live.

# Interface
Concrete subtypes should implement:
- [`get_num_points`](@ref): total number of (distinct) collocation points, i.e. the number
    of rows in the assembled system.
- [`get_num_elements`](@ref): number of elements over which the points are distributed.
- [`get_collocation_points`](@ref): the element-local canonical points of a given element.
- [`get_point_ids`](@ref): the global row indices of those points.
"""
abstract type AbstractCollocationPoints{manifold_dim} end

"""
    get_num_points(points::AbstractCollocationPoints)

Returns the total number of distinct collocation points, i.e. the number of rows in the
assembled collocation system.
"""
function get_num_points(::AbstractCollocationPoints)
    throw(ArgumentError("Method not implemented for this type of collocation points."))
end

function get_num_elements(::AbstractCollocationPoints)
    throw(ArgumentError("Method not implemented for this type of collocation points."))
end

function get_collocation_points(::AbstractCollocationPoints, ::Int)
    throw(ArgumentError("Method not implemented for this type of collocation points."))
end

"""
    get_point_ids(points::AbstractCollocationPoints, element_id::Int)

Returns the global row indices contributed by the collocation points lying in `element_id`,
ordered to match the points returned by [`get_collocation_points`](@ref).
"""
function get_point_ids(::AbstractCollocationPoints, ::Int)
    throw(ArgumentError("Method not implemented for this type of collocation points."))
end

"""
    is_bijective(points::AbstractCollocationPoints)

Returns `true` if there is a one-to-one correspondence between collocation points (rows) and
basis functions (columns), with matching global indices. This holds for Greville collocation
(a square system), and is required for the current Dirichlet boundary-condition handling,
which reuses the Galerkin machinery keyed by boundary basis indices. See
[`GrevilleCollocation`](@ref).
"""
is_bijective(::AbstractCollocationPoints) = false

############################################################################################
#                                        Structure                                         #
############################################################################################
"""
    CollocationPoints{manifold_dim} <: AbstractCollocationPoints{manifold_dim}

A set of collocation points bucketed per element. Within each element the points form a
tensor-product grid (as is the case for Greville abscissae on tensor-product spline spaces),
stored as the per-direction local canonical coordinates in `[0, 1]`.

# Fields
- `element_local_points::Vector{NTuple{manifold_dim, Vector{Float64}}}`: For each element,
    the per-direction local (canonical `[0, 1]`) coordinates of the collocation points in
    that element. An element with no collocation points has empty coordinate vectors.
- `element_point_ids::Vector{Vector{Int}}`: For each element, the global row indices of the
    element's collocation points, ordered to match the iteration order of the
    `Points.CartesianPoints` returned by [`get_collocation_points`](@ref).
- `num_points::Int`: The total number of distinct collocation points (number of rows).
- `bijective::Bool`: Whether point indices coincide with basis-function indices (see
    [`is_bijective`](@ref)).
"""
struct CollocationPoints{manifold_dim} <: AbstractCollocationPoints{manifold_dim}
    element_local_points::Vector{NTuple{manifold_dim, Vector{Float64}}}
    element_point_ids::Vector{Vector{Int}}
    num_points::Int
    bijective::Bool
end

get_num_points(cp::CollocationPoints) = cp.num_points
get_num_elements(cp::CollocationPoints) = length(cp.element_point_ids)
get_point_ids(cp::CollocationPoints, element_id::Int) = cp.element_point_ids[element_id]
is_bijective(cp::CollocationPoints) = cp.bijective

"""
    get_collocation_points(cp::CollocationPoints, element_id::Int)

Returns the element-local canonical collocation points of `element_id` as a
`Points.CartesianPoints`. Should only be called for elements that contain at least one
collocation point (i.e. when `get_point_ids(cp, element_id)` is non-empty).
"""
function get_collocation_points(
    cp::CollocationPoints{manifold_dim}, element_id::Int
) where {manifold_dim}
    return Points.CartesianPoints(cp.element_local_points[element_id]...)
end

############################################################################################
#                                       Constructors                                       #
############################################################################################
"""
    GrevilleCollocation(space::FunctionSpaces.AbstractFESpace)

Construct the Greville-abscissae collocation points for the given finite element `space`.

The Greville abscissae provide exactly one collocation point per basis function, yielding a
square system, and the `i`-th point is associated with the `i`-th basis function. This
point-to-basis bijection (see [`is_bijective`](@ref)) is what lets Dirichlet boundary
conditions reuse the Galerkin machinery keyed by boundary basis indices.

The points are returned in element-local canonical (`[0, 1]^manifold_dim`) coordinates,
bucketed per element. Currently supported for `BSplineSpace` (1D) and `TensorProductSpace`
(any dimension) spaces.

!!! note "Collocation points on element boundaries"
    A Greville point may coincide with an element boundary. It is then assigned to the
    element on whose closure it lies (local coordinate `0` or `1`). Evaluating the strong
    form there assumes the basis is regular enough at that point (e.g. ``C^2`` so that the
    Laplacian is continuous); this holds for maximal-continuity splines.
"""
function GrevilleCollocation(
    space::FunctionSpaces.AbstractFESpace{manifold_dim}
) where {manifold_dim}
    greville = FunctionSpaces.get_greville_points(space)
    return _collocation_from_tensor_points(
        space,
        greville,
        _basis_linear_indices(space),
        true,
        FunctionSpaces.get_num_basis(space),
    )
end

"""
    UserCollocation(
        space::FunctionSpaces.AbstractFESpace{manifold_dim},
        points_per_dim::NTuple{manifold_dim, Vector{Float64}},
    )

Construct a collocation-point set from a user-supplied tensor-product set of points, given in
parametric coordinates per direction. This enables non-Greville point choices, including
over-collocation (more points than basis functions, giving a least-squares system).

The resulting points do not, in general, have a point-to-basis bijection (see
[`is_bijective`](@ref)), so Dirichlet conditions are imposed through the least-squares path of
[`assemble`](@ref) rather than by row replacement.
"""
function UserCollocation(
    space::FunctionSpaces.AbstractFESpace{manifold_dim},
    points_per_dim::NTuple{manifold_dim, Vector{Float64}},
) where {manifold_dim}
    num_points = prod(length, points_per_dim)
    point_lin = LinearIndices(ntuple(k -> length(points_per_dim[k]), manifold_dim))
    return _collocation_from_tensor_points(
        space, points_per_dim, point_lin, false, num_points
    )
end

"""
    HierarchicalCollocation(
        space::FunctionSpaces.HierarchicalFiniteElementSpace{manifold_dim};
        level_points=nothing,
    )

Construct a collocation-point set for a hierarchical (locally refined) finite element `space`,
whose active mesh mixes elements from several refinement levels.

A set of candidate collocation points is chosen for each level of the hierarchy, given per
direction in parametric coordinates. By default these are the Greville abscissae of each
level's space. A custom set may be supplied through `level_points` as a collection with one
entry per level, each entry an `NTuple{manifold_dim, Vector{Float64}}` of per-direction points.

The hierarchical set is then assembled by taking, for every active element of level `ℓ`, the
level-`ℓ` candidate points lying inside that element, in element-local coordinates. A point on a
shared element boundary is collocated once: each interior boundary is owned by the element on
its lower side, and the outer domain boundary by its adjacent element.

The resulting points carry no point-to-basis bijection (see [`is_bijective`](@ref)), so
Dirichlet conditions are imposed through the least-squares path of [`assemble`](@ref), as for
[`UserCollocation`](@ref).
"""
function HierarchicalCollocation(
    space::FunctionSpaces.HierarchicalFiniteElementSpace{manifold_dim};
    level_points=nothing,
) where {manifold_dim}
    num_levels = FunctionSpaces.get_num_levels(space)
    num_elements = FunctionSpaces.get_num_elements(space)

    candidate_points = _hierarchical_candidate_points(
        space, level_points, num_levels, manifold_dim
    )

    # Per-level parametric breakpoints and element-index maps, used to find an active element's
    # extent in each direction.
    breakpoints = [
        Geometry.get_breakpoints(
            FunctionSpaces.get_parametric_geometry(FunctionSpaces.get_space(space, l))
        ) for l in 1:num_levels
    ]
    cart_maps = [
        Geometry.get_cart_num_elements(
            FunctionSpaces.get_parametric_geometry(FunctionSpaces.get_space(space, l))
        ) for l in 1:num_levels
    ]
    domain_min = ntuple(d -> breakpoints[1][d][1], manifold_dim)

    element_local_points = Vector{NTuple{manifold_dim, Vector{Float64}}}(undef, num_elements)
    element_point_ids = Vector{Vector{Int}}(undef, num_elements)
    num_points = 0
    for element in 1:num_elements
        level, level_id = FunctionSpaces.convert_to_element_level_and_level_id(
            space, element
        )
        cart = Tuple(cart_maps[level][level_id])
        # Per direction, keep the candidate points the element owns and map them to [0, 1].
        local_per_dim = ntuple(manifold_dim) do d
            bp = breakpoints[level][d]
            left, right = bp[cart[d]], bp[cart[d] + 1]
            owns_lower = left <= domain_min[d] + 1e-12
            selected = filter(candidate_points[level][d]) do g
                (left + 1e-12 < g <= right + 1e-12) ||
                    (owns_lower && abs(g - left) <= 1e-12)
            end
            return [(g - left) / (right - left) for g in selected]
        end
        # Row indices for this element's tensor grid (dimension 1 fastest), all distinct
        # because the ownership rule assigns each point to a single element.
        ids = collect((num_points + 1):(num_points + prod(length, local_per_dim)))
        num_points += length(ids)
        element_local_points[element] = local_per_dim
        element_point_ids[element] = ids
    end

    return CollocationPoints{manifold_dim}(
        element_local_points, element_point_ids, num_points, false
    )
end

############################################################################################
#                                    Internal helpers                                      #
############################################################################################
# Candidate collocation points per level, per direction. Defaults to the Greville abscissae of
# each level's space; otherwise validates and normalises the user-supplied set.
function _hierarchical_candidate_points(space, ::Nothing, num_levels::Int, manifold_dim::Int)
    return [
        FunctionSpaces.get_greville_points(FunctionSpaces.get_space(space, l)) for
        l in 1:num_levels
    ]
end
function _hierarchical_candidate_points(
    space, level_points, num_levels::Int, manifold_dim::Int
)
    if length(level_points) != num_levels
        throw(
            ArgumentError(
                "`level_points` must have one entry per level: expected $(num_levels), got \
                $(length(level_points)).",
            ),
        )
    end
    return [
        ntuple(d -> collect(Float64, level_points[l][d]), manifold_dim) for l in 1:num_levels
    ]
end

# The global (linear) index that a tuple of per-direction basis indices maps to. This must
# match the global basis numbering used by `Forms.evaluate`, so that collocation rows
# (point ids) coincide with basis ids for Greville collocation.
_basis_linear_indices(space::FunctionSpaces.TensorProductSpace) =
    FunctionSpaces.get_lin_num_basis(space)
_basis_linear_indices(space::FunctionSpaces.BSplineSpace) =
    LinearIndices((FunctionSpaces.get_num_basis(space),))
function _basis_linear_indices(space::FunctionSpaces.AbstractFESpace)
    throw(
        ArgumentError(
            "GrevilleCollocation is not implemented for spaces of type $(typeof(space)).",
        ),
    )
end

# Build a `CollocationPoints` from a tensor-product set of parametric points (one vector per
# direction). `point_lin` maps a tuple of per-direction point indices to a global row index;
# `bijective` records whether those row indices coincide with basis indices.
function _collocation_from_tensor_points(
    space::FunctionSpaces.AbstractFESpace{manifold_dim},
    points_per_dim::NTuple{manifold_dim, Vector{Float64}},
    point_lin,
    bijective::Bool,
    num_points::Int,
) where {manifold_dim}
    parametric_geometry = FunctionSpaces.get_parametric_geometry(space)
    breakpoints = Geometry.get_breakpoints(parametric_geometry)
    cart_elements = Geometry.get_cart_num_elements(parametric_geometry)
    num_elements = Geometry.get_num_elements(parametric_geometry)

    # Per direction, find for each point its element interval and its local coordinate.
    element_of_point = ntuple(manifold_dim) do dim
        bp = breakpoints[dim]
        num_intervals = length(bp) - 1
        map(g -> clamp(searchsortedlast(bp, g), 1, num_intervals), points_per_dim[dim])
    end
    local_of_point = ntuple(manifold_dim) do dim
        bp = breakpoints[dim]
        map(
            i -> begin
                e = element_of_point[dim][i]
                (points_per_dim[dim][i] - bp[e]) / (bp[e + 1] - bp[e])
            end,
            eachindex(points_per_dim[dim]),
        )
    end

    element_local_points = Vector{NTuple{manifold_dim, Vector{Float64}}}(
        undef, num_elements
    )
    element_point_ids = Vector{Vector{Int}}(undef, num_elements)
    for element_id in 1:num_elements
        element_cart = Tuple(cart_elements[element_id])
        # Indices (per direction) of the points lying in this element.
        point_idxs = ntuple(
            dim -> findall(==(element_cart[dim]), element_of_point[dim]), manifold_dim
        )
        element_local_points[element_id] = ntuple(
            dim -> local_of_point[dim][point_idxs[dim]], manifold_dim
        )

        num_per_dim = ntuple(dim -> length(point_idxs[dim]), manifold_dim)
        ids = Vector{Int}(undef, prod(num_per_dim))
        # Row indices ordered to match the `CartesianPoints` iteration order (dimension 1
        # fastest), i.e. column-major over `num_per_dim`.
        for (linear_idx, cart) in enumerate(CartesianIndices(num_per_dim))
            point_cart = ntuple(dim -> point_idxs[dim][cart[dim]], manifold_dim)
            ids[linear_idx] = point_lin[point_cart...]
        end
        element_point_ids[element_id] = ids
    end

    return CollocationPoints{manifold_dim}(
        element_local_points, element_point_ids, num_points, bijective
    )
end
