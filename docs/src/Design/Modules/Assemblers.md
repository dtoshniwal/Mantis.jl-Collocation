```@meta
CurrentModule = Mantis.Assemblers
```
# [Assemblers](@id DocAssemblyModule)

The `Assemblers` module turns a *weak formulation*, written in the language of [Forms](@ref),
into the matrices and vectors of a discrete linear (or generalised eigenvalue) problem. It
connects the symbolic form expressions you write to the sparse arrays you solve.

## The assembly workflow

A typical solve in `Mantis` follows the same four steps regardless of the PDE:

1. **Collect the inputs.** A [`WeakFormInputs`](@ref) bundles the trial space, the test space,
   and (optionally) a forcing term. When trial and test spaces coincide (the usual
   Bubnov-Galerkin case) you only pass it once.
2. **Describe the weak form.** Write a function that reads the trial/test/forcing objects back
   out of the inputs (with `get_trial_form`, `get_test_form` and `get_forcing`) and returns
   the left- and right-hand side as nested tuples of integral expressions, one entry per block
   of the (possibly mixed) system.
3. **Build a [`WeakForm`](@ref).** This pairs the left/right-hand side expressions with the
   inputs.
4. **Assemble.** [`assemble`](@ref) evaluates every block over the mesh and returns the global
   system, optionally applying boundary conditions.

```julia
using Mantis

# ... geometry, B-spline space B, forcing f⁰, quadrature dΩ defined as usual ...
Λ⁰  = Forms.FormSpace(0, B, "ϕ")
wfi = Assemblers.WeakFormInputs(Λ⁰, f⁰)

function poisson(inputs::Assemblers.AbstractInputs, dΩ)
    v⁰ = Assemblers.get_test_form(inputs)
    u⁰ = Assemblers.get_trial_form(inputs)
    f⁰ = Assemblers.get_forcing(inputs)

    A = ∫(d(v⁰) ∧ ★(d(u⁰)), dΩ)   # stiffness block
    b = ∫(v⁰ ∧ ★(f⁰), dΩ)         # load block
    return ((A,),), ((b,),)        # one-by-one block structure
end

bc = Forms.set_dirichlet_boundary_conditions(Λ⁰, 0.0)

lhs, rhs  = poisson(wfi, dΩ)
weak_form = Assemblers.WeakForm(lhs, rhs, wfi)
A, b      = Assemblers.assemble(weak_form, bc)
ϕ⁰        = Forms.build_form_field(Λ⁰, vec(A \ b))
```

### Block structure

The left- and right-hand sides are returned as *tuples of tuples* of real-valued operators
([`AbstractRealValuedOperator`](@ref Mantis.Forms.AbstractRealValuedOperator)s, typically
[`Integral`](@ref Mantis.Forms.Integral)s). The outer tuple indexes block-rows and the inner
tuple indexes block-columns, so a scalar problem is `((A,),)` while a two-field mixed problem
(e.g. the mixed Hodge-Laplacian, or the Maxwell saddle-point system) is a ``2 \times 2``
arrangement such as `((A11, A12), (A21, A22))`. This is how `Mantis` represents the compatible
mixed problems of Finite Element Exterior Calculus. [`assemble`](@ref) stitches the blocks
into a single global matrix.

### Boundary conditions

Essential (Dirichlet) boundary conditions are applied *at assembly time*. You build a boundary
condition object with the [Forms](@ref) helper
[`set_dirichlet_boundary_conditions`](@ref Mantis.Forms.set_dirichlet_boundary_conditions) and
pass it to [`assemble`](@ref); natural
conditions need no special treatment because they are already contained in the weak form. The
keyword arguments `lhs_type` / `rhs_type` let you request dense outputs (e.g.
`Matrix{Float64}`), which is convenient for the generalised eigenvalue problems below.

## Standard weak formulations

`Mantis` ships ready-made weak forms for several model problems, so you do not have to rewrite
them each time. Each comes both as a *weak-form function* (returning the block expressions) and
as a *solver* convenience function.

The L² projection is [`L2_projection`](@ref), with solver [`solve_L2_projection`](@ref); see
the [L2 projection](@ref) example. The Hodge-Laplacian is the `hodge_laplace` family for
``0``-, ``1``- and ``n``-forms. The Maxwell eigenvalue problem is [`maxwell_eigenvalue`](@ref),
solved by [`solve_maxwell_eig`](@ref) (single-mesh and adaptive variants) against the
analytical reference [`get_analytical_maxwell_eig`](@ref); see the
[Maxwell eigenvalue problem](@ref) and [Adaptive refinement](@ref) examples.

These are good starting points when writing the weak form for a new problem.

## Collocation assembly

Besides the weak (Galerkin) workflow above, `Mantis` can discretise a PDE by collocation, in
which the strong form is enforced pointwise at a chosen set of points. There is no test space
and no integration. Each collocation point contributes one row to the system, found by
evaluating the strong-form operator there. The collocation points say where the equations are
imposed, much as a quadrature rule says where the integrals are evaluated in Galerkin assembly.

The pipeline mirrors the four steps above, with collocation-specific replacements:

1. **Choose the points.** [`GrevilleCollocation`](@ref) builds one Greville point per basis
   function. The system is then square, with the `i`-th point tied to the `i`-th basis
   function. This point-to-basis bijection (see [`is_bijective`](@ref)) lets Dirichlet conditions be
   imposed in place by row replacement. [`UserCollocation`](@ref) takes a tensor-product set of
   points supplied per direction, enabling non-Greville choices and over-collocation (more
   points than basis functions, giving a least-squares system).
   [`HierarchicalCollocation`](@ref) builds points for a locally refined hierarchical space by
   choosing candidate points per level and keeping, for each active element, the points of its
   level that lie inside it; see the [Adaptive collocation](@ref) example.
2. **Collect the inputs.** A [`CollocationInputs`](@ref) bundles the trial form(s), the
   forcing(s), and the points. Unlike [`WeakFormInputs`](@ref) there is no test space.
3. **Build a [`CollocationForm`](@ref).** Its blocks are *bare forms* rather than integral
   operators: the `0`-form Poisson problem `δ(d(u⁰)) = -f`, for instance, is the single
   left-hand block `δ(d(u⁰))` against the right-hand block `-f`.
4. **Assemble.** [`assemble`](@ref) evaluates the strong-form blocks at the points and returns
   the global system. Dirichlet conditions are passed the same way regardless of the point set,
   as a `boundary basis index => value` dictionary, and are imposed strongly whether the
   strong-form system is square or rectangular: by row replacement for Greville points, and
   otherwise by lifting and appending constraint rows (see [`apply_collocation_dirichlet`](@ref)).

See the [Collocation](@ref) example for a worked Poisson solve using both point types.

## All docstrings from Mantis.Assemblers
```@autodocs
Modules = [Main.Mantis.Assemblers]
```
