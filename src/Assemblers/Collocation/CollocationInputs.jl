"""
    CollocationInputs{manifold_dim, TrF, F, P} <: AbstractInputs

Container for the trial spaces, forcing terms, and collocation points used in a
collocation-based assembly. Unlike [`WeakFormInputs`](@ref), there is no test space:
collocation enforces the strong form pointwise, so the rows of the linear system correspond
to collocation points rather than to test basis functions.

# Fields
- `trial_forms::TrF`: The trial forms (each a `Forms.AbstractFormSpace`).
- `forcings::F`: The forcing terms (each a `Forms.AbstractFormField`), or `(nothing,)`.
- `points::P`: The collocation points (an [`AbstractCollocationPoints`](@ref)). These define
    the rows of the system and are shared by all blocks.

# Type parameters
- `manifold_dim::Int`: The dimension of the manifold.
- `TrF`, `F`, `P`: Types of the trial forms, forcings, and collocation points.

# Constructors
- `CollocationInputs(trial_forms::NTuple, forcings::NTuple, points)`
- `CollocationInputs(trial_forms::NTuple, points)` — no forcing.
- `CollocationInputs(trial_form, forcing, points)` — single trial form and forcing.
- `CollocationInputs(trial_form, points)` — single trial form, no forcing.
"""
struct CollocationInputs{manifold_dim, TrF, F, P} <: AbstractInputs
    trial_forms::TrF
    forcings::F
    points::P

    function CollocationInputs(
        trial_forms::TrF, forcings::F, points::P
    ) where {
        manifold_dim,
        num_TrF,
        num_F,
        TrF <: NTuple{num_TrF, Forms.AbstractFormSpace{manifold_dim}},
        F <: NTuple{num_F, Forms.AbstractFormField{manifold_dim}},
        P <: AbstractCollocationPoints{manifold_dim},
    }
        return new{manifold_dim, TrF, F, P}(trial_forms, forcings, points)
    end

    function CollocationInputs(
        trial_forms::TrF, points::P
    ) where {
        manifold_dim,
        num_TrF,
        TrF <: NTuple{num_TrF, Forms.AbstractFormSpace{manifold_dim}},
        P <: AbstractCollocationPoints{manifold_dim},
    }
        return new{manifold_dim, TrF, Tuple{Nothing}, P}(trial_forms, (nothing,), points)
    end

    function CollocationInputs(
        trial_form::TrF, forcing::F, points::P
    ) where {
        manifold_dim,
        TrF <: Forms.AbstractFormSpace{manifold_dim},
        F <: Forms.AbstractFormField{manifold_dim},
        P <: AbstractCollocationPoints{manifold_dim},
    }
        return CollocationInputs((trial_form,), (forcing,), points)
    end

    function CollocationInputs(
        trial_form::TrF, points::P
    ) where {
        manifold_dim,
        TrF <: Forms.AbstractFormSpace{manifold_dim},
        P <: AbstractCollocationPoints{manifold_dim},
    }
        return CollocationInputs((trial_form,), points)
    end
end

get_trial_forms(ci::CollocationInputs) = ci.trial_forms
get_forcings(ci::CollocationInputs) = ci.forcings
get_points(ci::CollocationInputs) = ci.points
get_trial_form(ci::CollocationInputs, i::Int=1) = get_trial_forms(ci)[i]
get_forcing(ci::CollocationInputs, i::Int=1) = get_forcings(ci)[i]
get_num_trial(ci::CollocationInputs) = length(get_trial_forms(ci))
get_num_forcings(ci::CollocationInputs) = length(get_forcings(ci))
