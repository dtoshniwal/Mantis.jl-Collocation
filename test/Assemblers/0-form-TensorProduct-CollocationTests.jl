module ZeroFormTensorProductCollocationTests

using Mantis

using Test
using LinearAlgebra

# Manufactured solution u = ∏ᵢ sin(ω xᵢ) and the matching collocation forcing.
#
# In Mantis the *pointwise* codifferential `δ(d(u⁰))` evaluates to the analyst Laplacian
# `+Δu = +∑ᵢ ∂²u/∂xᵢ²`. The Poisson problem `-Δu = f` is therefore collocated as
# `δ(d(u⁰)) = -f`. For u = ∏ᵢ sin(ω xᵢ) we have `-Δu = manifold_dim · ω² · u`, so the
# right-hand side field is `-f = -manifold_dim · ω² · u`.
function manufactured_forms(
    form_space::Forms.AbstractFormSpace{manifold_dim}
) where {manifold_dim}
    geometry = Forms.get_geometry(form_space)
    ω = 2.0 * pi
    function sol(x::Matrix{Float64})
        return [vec(prod(sin.(ω .* x); dims=2))]
    end
    function rhs(x::Matrix{Float64})
        y = prod(sin.(ω .* x); dims=2)
        return [vec(@. -manifold_dim * ω * ω * y)]
    end
    uₑ = Forms.AnalyticalFormField(0, sol, geometry, "u")
    f⁰ = Forms.AnalyticalFormField(0, rhs, geometry, "f")
    return uₑ, f⁰
end

# Solve `-Δu = f` by Greville collocation and return the discrete and exact solutions.
function solve_collocation_poisson(form_space, bc_value=0.0)
    uₑ, f⁰ = manufactured_forms(form_space)
    points = Assemblers.GrevilleCollocation(Forms.get_fe_space(form_space))
    inputs = Assemblers.CollocationInputs(form_space, f⁰, points)
    u⁰ = Assemblers.get_trial_form(inputs)
    collocation_form = Assemblers.CollocationForm(((δ(d(u⁰)),),), ((f⁰,),), inputs)
    bc = Forms.set_dirichlet_boundary_conditions(form_space, bc_value)
    A, b = Assemblers.assemble(collocation_form, bc)
    uₕ = Forms.build_form_field(form_space, vec(A \ b))
    return uₕ, uₑ, bc
end

# L2 error of the collocation solution on a `num_elements` mesh of degree-`p` C^{p-1} splines.
function collocation_l2_error(origin, box, num_elements, p)
    B = FunctionSpaces.create_bspline_space(origin, box, num_elements, p, p .- 1)
    Λ⁰ = Forms.FormSpace(0, B, "u")
    uₕ, uₑ, _ = solve_collocation_poisson(Λ⁰)
    geometry = Forms.get_geometry(Λ⁰)
    quad_degrees = (p isa Tuple ? p : (p,)) .+ 3
    canonical_qrule = Quadrature.tensor_product_rule(
        quad_degrees, Quadrature.gauss_legendre
    )
    dΩ = Quadrature.StandardQuadrature(
        canonical_qrule, Geometry.get_num_elements(geometry)
    )
    return Analysis.compute_error_total(uₕ, uₑ, dΩ, "L2")
end

@testset "GrevilleCollocation point construction" begin
    B = FunctionSpaces.create_bspline_space((0.0, 0.0), (1.0, 1.0), (4, 4), (3, 3), (2, 2))
    points = Assemblers.GrevilleCollocation(B)
    num_basis = FunctionSpaces.get_num_basis(B)

    # Square system: one collocation point per basis function.
    @test Assemblers.get_num_points(points) == num_basis
    @test Assemblers.get_num_elements(points) ==
        Geometry.get_num_elements(FunctionSpaces.get_parametric_geometry(B))
    @test Assemblers.is_bijective(points)

    # Each basis function's point lies in exactly one element; together they cover 1:num_basis.
    all_ids = Int[]
    for e in 1:Assemblers.get_num_elements(points)
        ids = Assemblers.get_point_ids(points, e)
        append!(all_ids, ids)
        # Local coordinates are in the canonical [0, 1] cell.
        xi = Assemblers.get_collocation_points(points, e)
        for i in 1:Points.get_num_points(xi)
            @test all(0.0 .<= xi[i] .<= 1.0)
        end
    end
    @test sort(all_ids) == collect(1:num_basis)
end

@testset "1D Poisson convergence" begin
    p = 3
    errors = [collocation_l2_error(0.0, 1.0, n, p) for n in (8, 16, 32, 64)]
    rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
    @test issorted(errors; rev=true)                 # error decreases under refinement
    @test errors[end] < 1e-3                          # accurate on the finest mesh
    @test rates[end] > p - 1.5                        # near the expected collocation rate
end

@testset "2D Poisson convergence" begin
    p = (3, 3)
    errors = [
        collocation_l2_error((0.0, 0.0), (1.0, 1.0), (n, n), p) for n in (8, 16, 32)
    ]
    rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
    @test issorted(errors; rev=true)
    @test errors[end] < 1e-2
    @test rates[end] > 1.5
end

@testset "Dirichlet boundary conditions" begin
    B = FunctionSpaces.create_bspline_space((0.0, 0.0), (1.0, 1.0), (4, 4), (3, 3), (2, 2))
    Λ⁰ = Forms.FormSpace(0, B, "u")
    bc_value = 0.5
    uₕ, _, bc = solve_collocation_poisson(Λ⁰, bc_value)
    # Each boundary basis coefficient is set exactly to the prescribed value, since the
    # corresponding collocation row is replaced by the identity equation.
    for idx in keys(bc)
        @test isapprox(uₕ.coefficients[idx], bc_value; atol=1e-12)
    end
end

@testset "UserCollocation over-collocation and strong boundary conditions" begin
    B = FunctionSpaces.create_bspline_space(0.0, 1.0, 8, 3, 2)
    num_basis = FunctionSpaces.get_num_basis(B)
    # Over-collocate: twice as many points as basis functions, so the strong-form system is
    # rectangular.
    pts = (collect(LinRange(0.0, 1.0, 2 * num_basis)),)
    user_points = Assemblers.UserCollocation(B, pts)

    @test !Assemblers.is_bijective(user_points)
    @test Assemblers.get_num_points(user_points) == 2 * num_basis

    Λ⁰ = Forms.FormSpace(0, B, "u")
    uₑ, f⁰ = manufactured_forms(Λ⁰)
    inputs = Assemblers.CollocationInputs(Λ⁰, f⁰, user_points)
    u⁰ = Assemblers.get_trial_form(inputs)
    collocation_form = Assemblers.CollocationForm(((δ(d(u⁰)),),), ((f⁰,),), inputs)

    # Without boundary conditions the strong-form system is rectangular: one row per
    # collocation point, one column per basis function.
    A, b = Assemblers.assemble(collocation_form)
    @test size(A) == (2 * num_basis, num_basis)

    # Boundary conditions are now imposed strongly for the non-bijective set too, exactly as
    # for Greville points: pass the `basis index => value` dictionary to `assemble`. The
    # appended constraint rows keep one column per basis function, and `\` solves the
    # (over-determined) system in the least-squares sense.
    bc_value = 0.5
    bc = Forms.set_dirichlet_boundary_conditions(Λ⁰, bc_value)
    A_bc, b_bc = Assemblers.assemble(collocation_form, bc)
    @test size(A_bc, 2) == num_basis
    @test size(A_bc, 1) == 2 * num_basis + length(bc)

    uₕ = Forms.build_form_field(Λ⁰, vec(A_bc \ b_bc))
    # Each boundary coefficient is set exactly to the prescribed value.
    for idx in keys(bc)
        @test isapprox(uₕ.coefficients[idx], bc_value; atol=1e-12)
    end

    # With homogeneous conditions the over-collocated least-squares solution is accurate.
    bc0 = Forms.set_dirichlet_boundary_conditions(Λ⁰, 0.0)
    A0, b0 = Assemblers.assemble(collocation_form, bc0)
    uₕ0 = Forms.build_form_field(Λ⁰, vec(A0 \ b0))
    geometry = Forms.get_geometry(Λ⁰)
    dΩ = Quadrature.StandardQuadrature(
        Quadrature.tensor_product_rule((6,), Quadrature.gauss_legendre),
        Geometry.get_num_elements(geometry),
    )
    @test Analysis.compute_error_total(uₕ0, uₑ, dΩ, "L2") < 1e-3
end

@testset "HierarchicalCollocation on a refined 1D mesh" begin
    # Solution with a peak near x = 1, so refining the right-most elements helps.
    aa = 20
    sol(x::Matrix{Float64}) = [vec(@. x[:, 1]^aa * (1 - x[:, 1]))]
    function rhs(x::Matrix{Float64})
        return [vec(@. aa * (aa - 1) * x[:, 1]^(aa - 2) - (aa + 1) * aa * x[:, 1]^(aa - 1))]
    end

    function solve(H)
        Λ⁰ = Forms.FormSpace(0, H, "u")
        geometry = Forms.get_geometry(Λ⁰)
        f⁰ = Forms.AnalyticalFormField(0, rhs, geometry, "f")
        uₑ = Forms.AnalyticalFormField(0, sol, geometry, "u")
        points = Assemblers.HierarchicalCollocation(H)
        inputs = Assemblers.CollocationInputs(Λ⁰, f⁰, points)
        u⁰ = Assemblers.get_trial_form(inputs)
        cf = Assemblers.CollocationForm(((δ(d(u⁰)),),), ((f⁰,),), inputs)
        bc = Forms.set_dirichlet_boundary_conditions(Λ⁰, 0.0)
        A, b = Assemblers.assemble(cf, bc)
        uₕ = Forms.build_form_field(Λ⁰, vec(A \ b))
        dΩ = Quadrature.StandardQuadrature(
            Quadrature.tensor_product_rule((8,), Quadrature.gauss_legendre),
            Geometry.get_num_elements(geometry),
        )
        return Analysis.compute_error_total(uₕ, uₑ, dΩ, "L2"), points
    end

    function refine_last_two(H)
        level = findlast(
            l -> !isempty(FunctionSpaces.get_level_element_ids(H, l)),
            1:FunctionSpaces.get_num_levels(H),
        )
        ids = sort(FunctionSpaces.get_level_element_ids(H, level))
        marked = [Int[] for _ in 1:FunctionSpaces.get_num_levels(H)]
        marked[level] = ids[(end - 1):end]
        return FunctionSpaces.refine_space(H, marked)
    end

    B = FunctionSpaces.create_bspline_space((0.0,), (1.0,), (5,), (3,), (2,))
    H = FunctionSpaces.HierarchicalFiniteElementSpace(B, (2,), true, false)

    errors = Float64[]
    for _ in 0:2
        error, points = solve(H)
        push!(errors, error)
        # The default candidate points are the per-level Greville abscissae; with the
        # boundary-ownership rule each active basis gets exactly one collocation point.
        @test Assemblers.get_num_points(points) == FunctionSpaces.get_num_basis(H)
        @test !Assemblers.is_bijective(points)
        H = refine_last_two(H)
    end

    # Refining the two right-most elements drives the error down at every step.
    @test issorted(errors; rev=true)
    @test errors[end] < errors[1] / 5

    # Custom level points are accepted, and a wrong number of levels is rejected.
    custom = [
        FunctionSpaces.get_greville_points(FunctionSpaces.get_space(H, l)) for
        l in 1:FunctionSpaces.get_num_levels(H)
    ]
    @test Assemblers.get_num_points(
        Assemblers.HierarchicalCollocation(H; level_points=custom)
    ) == FunctionSpaces.get_num_basis(H)
    @test_throws ArgumentError Assemblers.HierarchicalCollocation(H; level_points=custom[1:1])
end

end
