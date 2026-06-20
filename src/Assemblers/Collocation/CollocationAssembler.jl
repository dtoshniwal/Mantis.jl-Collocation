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
    boundary basis index. Only supported when the collocation points have a point-to-basis
    bijection (e.g. [`GrevilleCollocation`](@ref)); see [`is_bijective`](@ref).
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
    if !isempty(dirichlet_bcs) && !is_bijective(points)
        throw(
            ArgumentError(
                """Dirichlet boundary conditions are only supported for collocation point \
                sets with a point-to-basis bijection (e.g. `GrevilleCollocation`)."""
            ),
        )
    end

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

    zero_rows!(lhs_vals, rhs_vals, lhs_rows, rhs_rows, dirichlet_bcs)
    lhs = build_array(
        lhs_type,
        lhs_rows[1:lhs_counts],
        lhs_cols[1:lhs_counts],
        lhs_vals[1:lhs_counts],
        get_lhs_size(collocation_form),
    )
    rhs = build_array(
        rhs_type,
        rhs_rows[1:rhs_counts],
        rhs_cols[1:rhs_counts],
        rhs_vals[1:rhs_counts],
        get_rhs_size(collocation_form),
    )
    lhs, rhs = set_diagonal!(lhs, rhs, dirichlet_bcs)

    return lhs, rhs
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
