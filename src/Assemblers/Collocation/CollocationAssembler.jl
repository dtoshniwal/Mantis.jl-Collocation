"""
    assemble(
        collocation_form::CollocationForm{manifold_dim, LHS, RHS, I},
        dirichlet_bcs::Dict{Int, Float64}=Dict{Int, Float64}();
        lhs_type::Type=spa.SparseMatrixCSC{Float64, Int},
        rhs_type::Type=Vector{Float64},
    )

Assemble the left- and right-hand sides of a collocation discretisation for the given
[`CollocationForm`](@ref) and Dirichlet boundary conditions.

Each collocation point contributes one row to the system: the strong-form left-hand side
blocks are evaluated pointwise at the collocation points (giving the matrix rows), and the
right-hand side forcing is evaluated at the same points. This shares the sparse-assembly
core (`build_array`, `zero_rows!`, `set_diagonal!`) with the Galerkin
[`assemble`](@ref) but replaces weighted integration with point evaluation.

# Arguments
- `collocation_form::CollocationForm`: The collocation discretisation to assemble.
- `dirichlet_bcs::Dict{Int, Float64}`: The Dirichlet boundary conditions, keyed by the
    boundary basis index. Imposed strongly for any collocation-point set, independently of
    whether the strong-form system is square or rectangular. When the points have a
    point-to-basis bijection (e.g. [`GrevilleCollocation`](@ref); see [`is_bijective`](@ref))
    each boundary row is replaced by its identity equation, keeping the system square.
    Otherwise the boundary degrees of freedom are lifted to the right-hand side and one
    constraint row `u_b = value` is appended per boundary basis function (see
    [`apply_collocation_dirichlet`](@ref)).
- `lhs_type::Type`: The type of the left-hand side array. Default is
    `SparseMatrixCSC{Float64, Int}`.
- `rhs_type::Type`: The type of the right-hand side array. Default is `Vector{Float64}`.

# Returns
- `lhs::lhs_type`: The assembled left-hand side array.
- `rhs::rhs_type`: The assembled right-hand side vector.
"""
function assemble(
    collocation_form::CollocationForm{manifold_dim, LHS, RHS, I},
    dirichlet_bcs::Dict{Int, Float64}=Dict{Int, Float64}();
    lhs_type::Type=spa.SparseMatrixCSC{Float64, Int},
    rhs_type::Type=Vector{Float64},
) where {
    manifold_dim,
    num_rows,
    lhs_num_cols,
    rhs_num_cols,
    LHS <: NTuple{num_rows, NTuple{lhs_num_cols, Union{Int, Forms.AbstractForm}}},
    RHS <: NTuple{num_rows, NTuple{rhs_num_cols, Union{Int, Forms.AbstractForm}}},
    I,
}
    points = get_points(collocation_form)
    # With a point-to-basis bijection (Greville) the boundary basis index coincides with the
    # boundary row, so the conditions can be imposed in place by row replacement. Otherwise the
    # conditions are imposed after assembly, by lifting and appending constraint rows.
    row_replacement_bc = is_bijective(points)

    lhs_expressions = get_lhs_expressions(collocation_form)
    rhs_expressions = get_rhs_expressions(collocation_form)
    row_offsets = get_row_offsets(collocation_form)
    trial_offsets = get_trial_offsets(collocation_form)
    lhs_rows, lhs_cols, lhs_vals = get_collocation_pre_allocation(
        collocation_form, "lhs", lhs_type
    )
    rhs_rows, rhs_cols, rhs_vals = get_collocation_pre_allocation(
        collocation_form, "rhs", rhs_type
    )
    lhs_counts, rhs_counts = 0, 0

    for elem_id in 1:get_num_elements(collocation_form)
        point_ids = get_point_ids(points, elem_id)
        isempty(point_ids) && continue
        xi = get_collocation_points(points, elem_id)
        for row in 1:num_rows
            for col in 1:lhs_num_cols
                lhs_rows, lhs_cols, lhs_vals, lhs_counts = add_collocation_contributions!(
                    lhs_rows,
                    lhs_cols,
                    lhs_vals,
                    lhs_counts,
                    lhs_expressions[row][col],
                    elem_id,
                    xi,
                    point_ids,
                    row_offsets[row],
                    trial_offsets[col],
                )
            end

            for col in 1:rhs_num_cols
                rhs_rows, rhs_cols, rhs_vals, rhs_counts = add_collocation_contributions!(
                    rhs_rows,
                    rhs_cols,
                    rhs_vals,
                    rhs_counts,
                    rhs_expressions[row][col],
                    elem_id,
                    xi,
                    point_ids,
                    row_offsets[row],
                    trial_offsets[col],
                )
            end
        end
    end

    lhs_size = get_lhs_size(collocation_form)
    rhs_size = get_rhs_size(collocation_form)

    if row_replacement_bc
        # Bijective (Greville) points: the boundary basis index is also the boundary row, so
        # the conditions are imposed in place on the built arrays by row replacement.
        zero_rows!(lhs_vals, rhs_vals, lhs_rows, rhs_rows, dirichlet_bcs)
        lhs = build_array(
            lhs_type, lhs_rows[1:lhs_counts], lhs_cols[1:lhs_counts], lhs_vals[1:lhs_counts],
            lhs_size,
        )
        rhs = build_array(
            rhs_type, rhs_rows[1:rhs_counts], rhs_cols[1:rhs_counts], rhs_vals[1:rhs_counts],
            rhs_size,
        )
        return set_diagonal!(lhs, rhs, dirichlet_bcs)
    elseif !isempty(dirichlet_bcs)
        # Non-bijective points: impose the conditions strongly while still in triplet (COO)
        # form, before any sparse matrix is materialised.
        lhs_triplets, rhs_triplets = apply_collocation_dirichlet(
            (lhs_rows, lhs_cols, lhs_vals, lhs_counts, lhs_size),
            (rhs_rows, rhs_cols, rhs_vals, rhs_counts, rhs_size),
            dirichlet_bcs,
        )
        lhs_rows, lhs_cols, lhs_vals, lhs_counts, lhs_size = lhs_triplets
        rhs_rows, rhs_cols, rhs_vals, rhs_counts, rhs_size = rhs_triplets
    end

    lhs = build_array(
        lhs_type, lhs_rows[1:lhs_counts], lhs_cols[1:lhs_counts], lhs_vals[1:lhs_counts],
        lhs_size,
    )
    rhs = build_array(
        rhs_type, rhs_rows[1:rhs_counts], rhs_cols[1:rhs_counts], rhs_vals[1:rhs_counts],
        rhs_size,
    )
    return lhs, rhs
end

"""
    apply_collocation_dirichlet(lhs_triplets, rhs_triplets, dirichlet_bcs::Dict{Int, T}) where {T}

Strongly impose the Dirichlet conditions `dirichlet_bcs` (a `basis index => value` map) on a
collocation system still held in triplet (coordinate / COO) form, *before* any matrix is
built. Each argument is a tuple `(rows, cols, vals, count, size)` describing the first `count`
nonzeros of the left- and right-hand sides respectively.

The conditions are imposed without forming the (possibly large, rectangular) matrix:

- For each boundary basis function `b` with prescribed value `v`, every left-hand-side entry
  in column `b` is dropped and its known contribution `value * v` is moved to the right-hand
  side, decoupling that degree of freedom from the strong-form rows.
- One constraint row `u_b = v` is appended per boundary basis function.

Because each boundary degree of freedom then appears only in its own constraint row, the
least-squares (or exact, if square) solution satisfies `u_b = v` exactly. The procedure does
not depend on whether the strong-form rows form a square or rectangular (over-collocated)
system, and the returned triplets keep one column per basis function, so the full coefficient
vector is recovered directly from `lhs \\ rhs`.

Returns the updated `(rows, cols, vals, count, size)` tuples for the left- and right-hand
sides.
"""
function apply_collocation_dirichlet(
    lhs_triplets, rhs_triplets, dirichlet_bcs::Dict{Int, T}
) where {T}
    lhs_rows, lhs_cols, lhs_vals, lhs_count, lhs_size = lhs_triplets
    rhs_rows, rhs_cols, rhs_vals, rhs_count, rhs_size = rhs_triplets

    boundary_indices = collect(keys(dirichlet_bcs))
    num_bc = length(boundary_indices)
    # Row indices of the appended constraint equations, one past the existing rows.
    constraint_row = Dict(b => lhs_size[1] + k for (k, b) in enumerate(boundary_indices))

    out_lhs_rows = Int[]
    out_lhs_cols = Int[]
    out_lhs_vals = T[]
    sizehint!(out_lhs_rows, lhs_count + num_bc)
    sizehint!(out_lhs_cols, lhs_count + num_bc)
    sizehint!(out_lhs_vals, lhs_count + num_bc)

    # Extra right-hand-side entries: lifted boundary contributions plus the constraint values.
    extra_rhs_rows = Int[]
    extra_rhs_vals = T[]

    for i in 1:lhs_count
        col = lhs_cols[i]
        value = get(dirichlet_bcs, col, nothing)
        if value === nothing
            push!(out_lhs_rows, lhs_rows[i])
            push!(out_lhs_cols, col)
            push!(out_lhs_vals, lhs_vals[i])
        else
            # Lift `lhs_vals[i] * u_col = lhs_vals[i] * value` to the right-hand side.
            push!(extra_rhs_rows, lhs_rows[i])
            push!(extra_rhs_vals, -lhs_vals[i] * value)
        end
    end

    for b in boundary_indices
        push!(out_lhs_rows, constraint_row[b])
        push!(out_lhs_cols, b)
        push!(out_lhs_vals, one(T))
        push!(extra_rhs_rows, constraint_row[b])
        push!(extra_rhs_vals, dirichlet_bcs[b])
    end

    out_rhs_rows = vcat(view(rhs_rows, 1:rhs_count), extra_rhs_rows)
    out_rhs_vals = vcat(view(rhs_vals, 1:rhs_count), extra_rhs_vals)
    out_rhs_cols = ones(Int, length(out_rhs_rows))

    augmented_rows = lhs_size[1] + num_bc
    new_lhs = (
        out_lhs_rows, out_lhs_cols, out_lhs_vals, length(out_lhs_vals),
        (augmented_rows, lhs_size[2]),
    )
    new_rhs = (
        out_rhs_rows, out_rhs_cols, out_rhs_vals, length(out_rhs_vals),
        (augmented_rows, rhs_size[2]),
    )
    return new_lhs, new_rhs
end

"""
    add_collocation_contributions!(
        rows::Vector{Int},
        cols::Vector{Int},
        vals::AbstractVector,
        counts::Int,
        expression,
        element_id::Int,
        xi::Points.AbstractPoints,
        point_ids::Vector{Int},
        row_offset::Int,
        col_offset::Int,
    )

Updates the row, column, and value vectors with contributions from the strong-form
`expression` evaluated at the collocation points `xi` of element `element_id`. The rows of
the contribution are the global collocation-point indices `point_ids`; the columns (for a
trial expression) are the global basis indices returned by `Forms.evaluate`.

A rank-0 `expression` (a forcing field) contributes one value per collocation point to a
right-hand side vector. A rank-1 `expression` (a trial space acted on by the strong-form
operator) contributes a `num_points × num_basis` block.
"""
function add_collocation_contributions!(
    rows::Vector{Int},
    cols::Vector{Int},
    vals::AbstractVector,
    counts::Int,
    expression,
    element_id::Int,
    xi::Points.AbstractPoints,
    point_ids::Vector{Int},
    row_offset::Int,
    col_offset::Int,
)
    if expression == 0
        return rows, cols, vals, counts
    end

    form_eval, basis_indices = Forms.evaluate(expression, element_id, xi)
    # 0-forms and the Laplacian of 0-forms are scalar, so the first (and only) component
    # holds the values.
    block_eval = form_eval[1]

    if Forms.get_expression_rank(expression) == 0
        # Forcing: one value per collocation point (right-hand side vector).
        for point_local_id in eachindex(block_eval)
            counts += 1
            rows[counts] = point_ids[point_local_id] + row_offset
            vals[counts] = block_eval[point_local_id]
        end
    else
        # Trial space acted on by the strong-form operator: num_points × num_basis block.
        col_ids = basis_indices[1]
        for ord in CartesianIndices(block_eval)
            counts += 1
            rows[counts] = point_ids[ord[1]] + row_offset
            cols[counts] = col_ids[ord[2]] + col_offset
            vals[counts] = block_eval[ord]
        end
    end

    return rows, cols, vals, counts
end

"""
    get_collocation_pre_allocation(
        collocation_form::CollocationForm, side::String, ::Type{A}
    ) where {T, A <: AbstractArray{T}}

Returns pre-allocated row, column, and value vectors for the left-hand side (`"lhs"`) or
right-hand side (`"rhs"`) of the collocation system. Mirrors [`get_pre_allocation`](@ref) but
is sized from the per-element collocation-point counts.
"""
function get_collocation_pre_allocation(
    collocation_form::CollocationForm, side::String, ::Type{A}
) where {T, A <: AbstractArray{T}}
    nnz_elem = get_estimated_nnz_per_elem(collocation_form)
    if side == "lhs"
        nvals = nnz_elem[1] * get_num_evaluation_elements(collocation_form)
    elseif side == "rhs"
        nvals = nnz_elem[2] * get_num_evaluation_elements(collocation_form)
    else
        throw(ArgumentError("Invalid side: $(side). Must be 'lhs' or 'rhs'."))
    end

    rows = Vector{Int}(undef, nvals)
    if side == "rhs" && A <: AbstractVector
        cols = ones(Int, nvals)
    else
        cols = Vector{Int}(undef, nvals)
    end

    vals = Vector{eltype(A)}(undef, nvals)

    return rows, cols, vals
end
