############################################################################################
#                                        Structure                                         #
############################################################################################
"""
    CollocationForm{manifold_dim, LHS, RHS, I}

Structure representing a collocation discretisation of a (strong-form) PDE. The left- and
right-hand sides are given as blocks of form expressions; unlike a [`WeakForm`](@ref), these
blocks are **bare forms** rather than real-valued operators: collocation evaluates the strong
form pointwise at the collocation points (held in `inputs`), so there is no integral
operator and no test space.

For example, the 0-form Poisson problem `δ(d(u⁰)) = f` is represented with a single left-hand
block `δ(d(u⁰))` and a single right-hand block `f⁰`.

# Fields
- `lhs_expressions::LHS`: The left-hand side blocks. Each entry is a `Forms.AbstractForm`
    (typically an `AbstractFormSpace`, i.e. expression rank 1) or `0`.
- `rhs_expressions::RHS`: The right-hand side blocks. Each entry is a `Forms.AbstractForm`
    (typically an `AbstractFormField`, i.e. a forcing) or `0`.
- `inputs::I`: The [`CollocationInputs`](@ref) holding the trial forms, forcings, and the
    collocation points that define the rows of the system.

# Type parameters
- `manifold_dim::Int`: The dimension of the manifold.
- `LHS`, `RHS`: Types of the left- and right-hand side block tuples.
- `I`: Type of the inputs (a `CollocationInputs{manifold_dim}`).
"""
struct CollocationForm{manifold_dim, LHS, RHS, I}
    lhs_expressions::LHS
    rhs_expressions::RHS
    inputs::I

    function CollocationForm(
        lhs_expressions::LHS, rhs_expressions::RHS, inputs::I
    ) where {
        manifold_dim,
        lhs_num_rows,
        lhs_num_cols,
        rhs_num_rows,
        rhs_num_cols,
        LHS <: NTuple{lhs_num_rows, NTuple{lhs_num_cols, Union{Int, Forms.AbstractForm}}},
        RHS <: NTuple{rhs_num_rows, NTuple{rhs_num_cols, Union{Int, Forms.AbstractForm}}},
        I <: CollocationInputs{manifold_dim},
    }
        if lhs_num_rows != rhs_num_rows
            throw(
                ArgumentError(
                    """The number of rows on the left-hand side must match the number of \
                    rows on the right-hand side. The left-hand side has $(lhs_num_rows) \
                    rows and the right-hand side has $(rhs_num_rows) rows."""
                ),
            )
        end

        num_trial = get_num_trial(inputs)
        if num_trial != lhs_num_cols
            throw(
                ArgumentError(
                    """The number of columns on the left-hand side must match the number of \
                    trial forms. The left-hand side has $(lhs_num_cols) columns and the \
                    inputs have $(num_trial) trial spaces."""
                ),
            )
        end

        return new{manifold_dim, LHS, RHS, I}(lhs_expressions, rhs_expressions, inputs)
    end
end

############################################################################################
#                                         Getters                                          #
############################################################################################

get_lhs_expressions(cf::CollocationForm) = cf.lhs_expressions
get_rhs_expressions(cf::CollocationForm) = cf.rhs_expressions
get_inputs(cf::CollocationForm) = cf.inputs
get_points(cf::CollocationForm) = get_points(get_inputs(cf))
get_trial_forms(cf::CollocationForm) = get_trial_forms(get_inputs(cf))
get_forcing(cf::CollocationForm, id::Int=1) = get_forcing(get_inputs(cf), id)
get_trial_sizes(cf::CollocationForm) = Forms.get_num_basis.(get_trial_forms(cf))
get_trial_size(cf::CollocationForm) = sum(get_trial_sizes(cf))

# Number of equation (row) blocks, and points-per-block. Each row block contributes one
# collocation equation per collocation point, so it has `get_num_points` rows.
num_row_blocks(cf::CollocationForm) = length(get_lhs_expressions(cf))
get_num_rows(cf::CollocationForm) = num_row_blocks(cf) * get_num_points(get_points(cf))

"""
    get_row_offsets(cf::CollocationForm)

Returns the global row offset of each equation (row) block. Each block spans
`get_num_points` rows (one collocation equation per point).
"""
function get_row_offsets(cf::CollocationForm)
    num_points = get_num_points(get_points(cf))
    return ntuple(block -> (block - 1) * num_points, num_row_blocks(cf))
end

"""
    get_trial_offsets(cf::CollocationForm)

Returns the global column offset of each trial (column) block.
"""
function get_trial_offsets(cf::CollocationForm)
    trial_cumsum = [0; cumsum(collect(get_trial_sizes(cf))[1:(end - 1)])...]
    return ntuple(block -> trial_cumsum[block], length(get_trial_forms(cf)))
end

get_lhs_size(cf::CollocationForm) = (get_num_rows(cf), get_trial_size(cf))

function get_rhs_size(cf::CollocationForm)
    if isnothing(get_forcing(cf))
        return get_num_rows(cf), get_trial_size(cf)
    end

    return get_num_rows(cf), 1
end

"""
    get_num_elements(cf::CollocationForm)

Returns the number of elements over which the collocation points are distributed.
"""
get_num_elements(cf::CollocationForm) = get_num_elements(get_points(cf))

get_num_evaluation_elements(cf::CollocationForm) = get_num_elements(cf)

"""
    get_estimated_nnz_per_elem(cf::CollocationForm)

Returns a (safe over-)estimate of the number of non-zero entries contributed per element by
the left- and right-hand sides. Used to pre-allocate the assembly buffers.
"""
function get_estimated_nnz_per_elem(cf::CollocationForm)
    points = get_points(cf)
    max_points_per_elem = maximum(
        (length(get_point_ids(points, e)) for e in 1:get_num_elements(points)); init=0
    )

    lhs_nnz = 0
    for lhs_row in get_lhs_expressions(cf), expression in lhs_row
        expression == 0 && continue
        lhs_nnz += max_points_per_elem * Forms.get_max_local_dim(expression)
    end

    rhs_nnz = 0
    for rhs_row in get_rhs_expressions(cf), expression in rhs_row
        expression == 0 && continue
        rhs_nnz += max_points_per_elem
    end

    return (lhs_nnz, rhs_nnz)
end
